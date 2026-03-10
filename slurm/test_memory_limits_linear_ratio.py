#!/usr/bin/env python3
"""
Memory Stress Test for MOBO Geometries — Linear-Ratio Configuration

Tests memory usage across the full MOBO parameter space (steel_ratio ×
steel_slope × scint_slope) to establish safe per-job memory limits.

Key insight from KLMWS_geo.cpp (linear_ratio branch):
  steel_slope and scint_slope ARE read by the C++ geometry code and modify
  per-layer slice thickness via a linear ramp:
    s_thick = s_thick_orig * (1 - slope + (layer-1) * 2*slope / (N-1))
  Factor ranges from (1-slope) at layer 1 to (1+slope) at layer N.

  Total scintillator volume is CONSERVED regardless of slope (linear function
  averages to 1.0 over all layers), but the DISTRIBUTION across layers affects
  particle absorption.  Thin steel in EARLY layers (high positive steel_slope)
  allows more particles to penetrate the full detector → more hits → more RAM.

Primary memory driver: HcalSteelThickness (= 75.5 * steel_ratio).
Secondary effect:      steel_slope — high positive slope → thin early steel
                       → more particle throughput → slightly higher memory.

Grid (16 configs = 4 steel_ratio × 4 steel_slope):
  steel_ratio : 0.3, 0.5, 0.7, 0.9  (→ HcalSteelThickness 22.65, 37.75, 52.85, 67.95 mm)
  steel_slope : -0.8, -0.3, 0.3, 0.8  (affects early-layer absorption distribution)
  scint_slope : 0.0 (fixed; does not affect absorption or hit count significantly)

Fixed geometry constants (not varied):
  HcalScintillatorNbLayers = 14
  HcalScintillatorThickness = 75.5 * (1 - steel_ratio) mm  (set by editxml.py)

Total: 16 configs × 3 particles × 3 reps = 144 SLURM jobs

Worst-case hypothesis:
  steel_ratio = 0.3 → HcalSteelThickness = 22.65 mm (minimum total absorption)
  steel_slope = +0.8 → layer 1 steel ≈ 22.65 × 0.2 = 4.53 mm (thin early absorption,
                        maximising particle throughput into all 14 layers)

Usage:
    source /hpc/group/vossenlab/rck32/eic/work_eic/setup.sh
    python3 /hpc/group/vossenlab/rck32/eic/work_eic/slurm/test_memory_limits_linear_ratio.py
    python3 slurm/test_memory_limits_linear_ratio.py --quick        # worst-case only
    python3 slurm/test_memory_limits_linear_ratio.py --analyze-only # re-analyze today's jobs
"""

import os
import sys
import subprocess
import shutil
import time
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
from workflow_util import get_mem_limit

# ── Configuration ──────────────────────────────────────────────────────────

# Total layer thickness is always 75.5mm (= 55.5 + 20 baseline).
# editxml.py 'steel_ratio' handler: steel_mm = 75.5 * ratio, scint_mm = 75.5 * (1-ratio)
TOTAL_PER_LAYER_MM = 75.5

# 4 steel_ratio values spanning the full MOBO range (0.3–0.9)
_STEEL_RATIOS   = [0.3, 0.5, 0.7, 0.9]

# 4 steel_slope values spanning the full MOBO range (-0.8–0.8)
_STEEL_SLOPES   = [-0.8, -0.3, 0.3, 0.8]

# test_id offset: 9300-series (9000=basic, 9200=preshower)
def _build_geometries():
    configs = []
    test_id = 9300
    for ratio in _STEEL_RATIOS:
        for slope in _STEEL_SLOPES:
            steel_mm = TOTAL_PER_LAYER_MM * ratio
            scint_mm = TOTAL_PER_LAYER_MM * (1 - ratio)
            slope_str = f"{slope:+.1f}".replace("+", "p").replace("-", "m").replace(".", "")
            name = f"lr_r{int(ratio*10):02d}_s{slope_str}"   # e.g. "lr_r01_sm09"
            configs.append((name, test_id, ratio, slope, steel_mm, scint_mm))
            test_id += 1
    return configs

ALL_GEOMETRIES = _build_geometries()

N_REPS = 3

QUICK_MODE   = "--quick"        in sys.argv
ANALYZE_ONLY = "--analyze-only" in sys.argv
for flag in ("--quick", "--analyze-only"):
    if flag in sys.argv:
        sys.argv.remove(flag)

if QUICK_MODE:
    # Worst case: minimum steel_ratio (lowest total absorption) + highest steel_slope
    # (thinnest steel in early layers → maximum particle throughput into all layers)
    GEOMETRIES = [g for g in ALL_GEOMETRIES if g[2] == 0.3 and g[3] == 0.8]
else:
    GEOMETRIES = ALL_GEOMETRIES

PARTICLES     = ["pi+", "mu-", "neutron"]
PARTICLE_SAFE = {"pi+": "pip", "mu-": "mum", "neutron": "neutron"}

# ── Environment ────────────────────────────────────────────────────────────

try:
    WORK_EIC  = os.environ['WORK_EIC']
    EPIC_HOME = os.environ['EPIC_HOME']
except KeyError as e:
    print(f"Missing env var {e}. Source work_eic/setup.sh first.")
    sys.exit(1)

SUBMIT_SCRIPT    = os.path.join(WORK_EIC, "slurm", "submit_workflow.py")
DETECTOR_INSTALL = os.path.join(EPIC_HOME, "install", "share", "epic")
KLMWS_TEMPLATE   = os.path.join(DETECTOR_INSTALL, "compact", "pid", "klmws.xml")
EPIC_TEMPLATE    = os.path.join(DETECTOR_INSTALL, "epic_klmws_only.xml")
SETUP_PATH       = os.path.join(EPIC_HOME, "install", "setup.sh")

# ── XML Generation ─────────────────────────────────────────────────────────

def create_test_xml(test_id, steel_mm, scint_mm, steel_slope, scint_slope=0.0):
    """
    Create klmws_{test_id}.xml and epic_klmws_only_{test_id}.xml.

    Sets HcalSteelThickness, HcalScintillatorThickness, steel_slope, and
    scint_slope.  In the linear_ratio branch of KLMWS_geo.cpp, all four
    constants are read and used to compute per-layer slice thicknesses:
      s_thick = s_thick_orig * (1 - slope + (layer-1) * 2*slope / (N-1))
    where s_thick_orig is HcalSteelThickness (Steel235 slices) or
    HcalScintillatorThickness (DR_Polystyrene slices).

    klmws.xml on the linear_ratio branch defines steel_slope and scint_slope
    as top-level <constant> elements (default 0), so this function updates them
    in-place.
    """
    klmws_dst = os.path.join(DETECTOR_INSTALL, "compact", "pid", f"klmws_{test_id}.xml")
    shutil.copyfile(KLMWS_TEMPLATE, klmws_dst)

    tree = ET.parse(klmws_dst)
    root = tree.getroot()

    for const in root.iter('constant'):
        name = const.get('name')
        if name == 'HcalSteelThickness':
            const.set('value', f'{steel_mm}*mm')
        elif name == 'HcalScintillatorThickness':
            const.set('value', f'{scint_mm}*mm')
        # steel_slope / scint_slope: write if constant already exists in template
        elif name == 'steel_slope':
            const.set('value', str(steel_slope))
        elif name == 'scint_slope':
            const.set('value', str(scint_slope))

    tree.write(klmws_dst)

    epic_dst = os.path.join(DETECTOR_INSTALL, f"epic_klmws_only_{test_id}.xml")
    shutil.copyfile(EPIC_TEMPLATE, epic_dst)
    tree = ET.parse(epic_dst)
    root = tree.getroot()
    for inc in root.iter('include'):
        ref = inc.get('ref', '')
        if 'klmws.xml' in ref and 'klmws_' not in ref:
            inc.set('ref', ref.replace('klmws.xml', f'klmws_{test_id}.xml'))
    tree.write(epic_dst)

    return epic_dst


def cleanup_test_xmls():
    for geo in GEOMETRIES:
        test_id = geo[1]
        for path in [
            os.path.join(DETECTOR_INSTALL, "compact", "pid", f"klmws_{test_id}.xml"),
            os.path.join(DETECTOR_INSTALL, f"epic_klmws_only_{test_id}.xml"),
        ]:
            if os.path.exists(path):
                os.remove(path)
                print(f"  Cleaned up {os.path.basename(path)}")


# ── Job Submission ─────────────────────────────────────────────────────────

def submit_all_tests():
    run_names_map = {}
    processes     = []

    log_dir = os.path.join(WORK_EIC, "slurm", "mem_test_logs_linear_ratio")
    os.makedirs(log_dir, exist_ok=True)

    for geo_name, test_id, ratio, slope, steel_mm, scint_mm in GEOMETRIES:
        compact_file = create_test_xml(test_id, steel_mm, scint_mm, steel_slope=slope)
        print(f"  Created XML: {geo_name}  ratio={ratio:.1f}  steel={steel_mm:.2f}mm  scint={scint_mm:.2f}mm  slope={slope:+.1f}")

        for particle in PARTICLES:
            psafe = PARTICLE_SAFE[particle]
            run_name_pref = f"mst_lr_{geo_name}_{psafe}"
            run_names_map[(geo_name, particle)] = run_name_pref

            cmd = [
                sys.executable, SUBMIT_SCRIPT,
                "--compactFile",     compact_file,
                "--setupPath",       SETUP_PATH,
                "--run_name_pref",   run_name_pref,
                "--runNum",          str(test_id),
                "--particle",        particle,
                "--num_simulations", str(N_REPS),
                "--skipTraining",
                "--no-classification",
                "--chPath",          EPIC_HOME,
                "--geo_config",      "linear_ratio",
            ]

            log_path = os.path.join(log_dir, f"{run_name_pref}.log")
            log_file = open(log_path, 'w')
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
            processes.append((geo_name, particle, test_id, proc, log_file))
            print(f"    Launched {run_name_pref}  (PID {proc.pid})")

    return run_names_map, processes


def wait_for_all(processes):
    total = len(processes)
    print(f"\nWaiting for {total} submit_workflow.py processes...")
    print(f"(Each manages {N_REPS} sim+process+analyze SLURM jobs, then monitors until done)")

    while True:
        still = [(g, p, t, proc, lf) for g, p, t, proc, lf in processes if proc.poll() is None]
        done  = total - len(still)
        if not still:
            break
        print(f"  {done}/{total} done, {len(still)} running... (sleeping 30s)")
        time.sleep(30)

    failed = []
    for geo_name, particle, test_id, proc, log_file in processes:
        log_file.close()
        if proc.returncode != 0:
            failed.append((geo_name, particle, proc.returncode))

    if failed:
        print(f"\nWARNING: {len(failed)} submit_workflow.py processes had errors:")
        for geo, part, rc in failed:
            print(f"  {geo} / {part}: exit {rc}")
    else:
        print(f"\nAll {total} submit_workflow.py processes finished.")

    return failed


# ── Memory Analysis ────────────────────────────────────────────────────────

def limit_to_mb(limit_str):
    return float(limit_str.replace('G', '')) * 1024


def mem_str_to_mb(val):
    if not val or val.strip() in ('', '0'):
        return 0.0
    val = val.strip()
    if val.endswith('K'):
        return float(val[:-1]) / 1024
    elif val.endswith('M'):
        return float(val[:-1])
    elif val.endswith('G'):
        return float(val[:-1]) * 1024
    try:
        return float(val) / 1024
    except ValueError:
        return 0.0


def analyze_memory(run_names_map):
    print("\n" + "=" * 120)
    print("MEMORY STRESS TEST RESULTS — LINEAR-RATIO CONFIGURATION")
    print(f"  {len(GEOMETRIES)} configs × {len(PARTICLES)} particles × {N_REPS} reps  |  "
          f"MOBO space: steel_ratio 0.3–0.9, steel_slope ±0.8")
    print("=" * 120)
    print()
    print("NOTE: HcalSteelThickness (= 75.5 * steel_ratio) is the primary memory driver.")
    print("      steel_slope has a secondary effect: high positive slope → thin early-layer steel")
    print("      → more particle throughput into all layers → more hits → more RAM.")
    print("      Total scintillator/steel volume is conserved regardless of slope (linear ramp averages to 1×).")
    print()

    result = subprocess.run(
        ["sacct", "-S", time.strftime("%Y-%m-%d"), "--parsable2",
         "--format=JobID,JobName%80,State%20,MaxRSS", "--noheader"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"sacct failed: {result.stderr}")
        return

    parent_info = {}
    batch_rss   = {}

    for line in result.stdout.strip().split('\n'):
        if not line.strip():
            continue
        parts = line.split('|')
        if len(parts) < 4:
            continue
        job_id, job_name, state, max_rss = parts[0], parts[1], parts[2], parts[3]

        if '.batch' in job_id:
            parent_id = job_id.split('.')[0]
            rss = mem_str_to_mb(max_rss)
            if rss > 0:
                batch_rss[parent_id] = rss
        elif '.' not in job_id and 'mst_lr_' in job_name:
            parent_info[job_id] = (job_name.strip(), state.strip())

    stats = defaultdict(list)  # (geo_name, particle) -> [(rss_mb, state)]

    for job_id, (job_name, state) in parent_info.items():
        m = re.match(r'mst_lr_(.+?)_(pip|mum|neutron)_', job_name)
        if not m:
            continue
        geo_name = m.group(1)
        psafe    = m.group(2)
        particle = {"pip": "pi+", "mum": "mu-", "neutron": "neutron"}[psafe]
        rss_mb   = batch_rss.get(job_id, 0.0)
        stats[(geo_name, particle)].append((rss_mb, state))

    hdr = (f"{'Config':<20} | {'Ratio':>5} | {'Steel mm':>8} | {'Scint mm':>8} | {'Slope':>5} | "
           f"{'Particle':<8} | {'Limit':>5} | {'N':>2} | "
           f"{'Mean MB':>8} | {'Max MB':>8} | {'Max GB':>6} | {'Headroom':>9} | OOM")
    print(hdr)
    print("-" * 120)

    any_concern = False
    # Track max observed per particle across ALL configs
    recommendations = {}   # particle -> max_gb seen

    for particle in PARTICLES:
        for geo in GEOMETRIES:
            geo_name, test_id, ratio, slope, steel_mm, scint_mm = geo
            key = (geo_name, particle)
            entries = stats.get(key, [])
            compact_file = os.path.join(DETECTOR_INSTALL, f"epic_klmws_only_{test_id}.xml")
            limit_str  = get_mem_limit(compact_file, particle, geo_config="linear_ratio")
            limit_mb   = limit_to_mb(limit_str)
            oom_count  = sum(1 for e in entries if 'OUT_OF_MEMORY' in e[1])
            rss_vals   = [e[0] for e in entries if e[0] > 0]

            if rss_vals:
                mean_mb      = sum(rss_vals) / len(rss_vals)
                max_mb       = max(rss_vals)
                max_gb       = max_mb / 1024
                headroom_pct = (1 - max_mb / limit_mb) * 100
                flag         = " (!)" if headroom_pct < 20 else ""
                headroom_str = f"{headroom_pct:.0f}%{flag}"
                if oom_count > 0 or headroom_pct < 20:
                    any_concern = True
                prev = recommendations.get(particle, {})
                # track max per steel tier for tier-specific recommendations
                tier = "thin" if steel_mm < 35 else "thick"
                prev_max = prev.get(tier, 0.0)
                prev[tier] = max(prev_max, max_gb)
                recommendations[particle] = prev
            else:
                mean_mb      = 0
                max_mb       = 0
                max_gb       = 0
                headroom_str = "N/A"

            n_str = str(len(entries)) if entries else "0"
            print(f"{geo_name:<20} | {ratio:>5.1f} | {steel_mm:>6.2f}mm | {scint_mm:>6.2f}mm | {slope:>+5.1f} | "
                  f"{PARTICLE_SAFE[particle]:<8} | {limit_str:>5} | {n_str:>2} | "
                  f"{mean_mb:>8.0f} | {max_mb:>8.0f} | {max_gb:>6.2f} | "
                  f"{headroom_str:>9} | {oom_count}")

        print()

    # ── Tier-based recommendations ──────────────────────────────────────────
    print("=" * 120)
    print("RECOMMENDED LIMITS per tier (max observed × 1.25, rounded to nearest GB):")
    print("  Tier boundaries: thin < 35mm ≤ thick  (steel_ratio split ≈ 0.46)")
    print()
    for particle in PARTICLES:
        tiers = recommendations.get(particle, {})
        for tier in ("thin", "thick"):
            max_gb = tiers.get(tier, 0.0)
            if max_gb == 0:
                print(f"  {particle:<8} [{tier:<5}]: no data")
                continue
            recommended = max_gb * 1.25
            recommended_gb = int(recommended) + (1 if recommended % 1 > 0 else 0)
            print(f"  {particle:<8} [{tier:<5}]: max observed = {max_gb:.2f} GB  →  recommended ≥ {recommended_gb}G")
        print()

    print()
    if any_concern:
        print("WARNING: Some configurations have <20% headroom or OOM failures — see (!) flags above.")
        print("         Update _get_mem_limit_linear_ratio() in workflow_util.py with the values above.")
    else:
        print("All configurations have adequate headroom (≥20%). Update workflow_util.py with the "
              "tighter recommended values to reduce wasted allocation.")
    print("=" * 120)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 120)
    print("MEMORY STRESS TEST FOR MOBO GEOMETRIES — LINEAR-RATIO CONFIGURATION")
    print("=" * 120)
    n_jobs = len(GEOMETRIES) * len(PARTICLES) * N_REPS
    print(f"  {len(GEOMETRIES)} configs × {len(PARTICLES)} particles × {N_REPS} reps = {n_jobs} SLURM jobs")
    if QUICK_MODE:
        print("  (QUICK MODE: worst-case geometry only — steel_ratio=0.3, steel_slope=+0.8)")
    if ANALYZE_ONLY:
        print("  (ANALYZE-ONLY: skipping submission, running sacct on today's mst_lr_* jobs)")
    print()

    if ANALYZE_ONLY:
        analyze_memory({})
        return

    print("[1/4] Generating test XMLs and submitting jobs...")
    run_names_map, processes = submit_all_tests()
    print(f"\n  Launched {len(processes)} submit_workflow.py processes")
    print()

    try:
        print("[2/4] Waiting for all SLURM jobs to finish...")
        failed = wait_for_all(processes)
        print()

        print("[3/4] Waiting 15s for sacct to finalize...")
        time.sleep(15)

        print("[4/4] Analyzing memory usage...")
        analyze_memory(run_names_map)

    finally:
        print("\nCleaning up test XML files...")
        cleanup_test_xmls()
        log_dir = os.path.join(WORK_EIC, "slurm", "mem_test_logs_linear_ratio")
        print(f"Submission logs in: {log_dir}")
        print("Done!")


if __name__ == "__main__":
    main()
