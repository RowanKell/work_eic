#!/usr/bin/env python3
"""
Memory Stress Test for MOBO Geometries

Tests memory usage across the full MOBO parameter space (num_layers × steel_ratio)
to establish safe per-job memory limits and minimize wasted allocation.

16 configurations = 4 layer counts × 4 steel ratios, covering thin/thick scint
and few/many layers. 3 jobs per (config, particle) combo.

Total: 16 geometries × 3 particles × 3 jobs = 144 jobs

MOBO parameter space:
  num_layers:  5–18  (INT)
  steel_ratio: 0.265–0.934  (total slab = 75.5 mm)
    steel_mm = 75.5 * steel_ratio
    scint_mm = 75.5 * (1 - steel_ratio)
  thick_scint threshold: scint_mm > 30.0

Current memory limits in submit_workflow.py:
  thick_scint + pi+/neutron: 32G
  thick_scint + mu-:         16G
  thin  scint (any):          8G

Usage:
    source /hpc/group/vossenlab/rck32/eic/work_eic/setup.sh
    python3 /hpc/group/vossenlab/rck32/eic/work_eic/slurm/test_memory_limits.py
    python3 /hpc/group/vossenlab/rck32/eic/work_eic/slurm/test_memory_limits.py --quick
    python3 /hpc/group/vossenlab/rck32/eic/work_eic/slurm/test_memory_limits.py --analyze-only
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

# ── Configuration ──────────────────────────────────────────────────────────

TOTAL = 75.5  # mm, total slab thickness (steel + scint)

# 4 layer counts × 4 steel ratios = 16 configs
# test_id offset: 9100-series to avoid clashing with old 9000-series runs
_LAYERS   = [5, 9, 14, 18]
_RATIOS   = [0.265, 0.45, 0.65, 0.934]  # spans thick→thin scint

def _build_geometries():
    configs = []
    test_id = 9100
    for layers in _LAYERS:
        for ratio in _RATIOS:
            steel = round(TOTAL * ratio, 2)
            scint = round(TOTAL * (1 - ratio), 2)
            thick = "thick" if scint > 30.0 else "thin"
            name = f"{layers}L_r{int(ratio*1000):04d}"  # e.g. 14L_r0265
            configs.append((name, test_id, layers, steel, scint, thick))
            test_id += 1
    return configs

ALL_GEOMETRIES = _build_geometries()

N_REPS = 3  # simulation jobs per (config, particle)

QUICK_MODE    = "--quick"        in sys.argv
ANALYZE_ONLY  = "--analyze-only" in sys.argv
for flag in ("--quick", "--analyze-only"):
    if flag in sys.argv:
        sys.argv.remove(flag)

if QUICK_MODE:
    # Worst-case only: 18L + thickest scint (ratio=0.265)
    GEOMETRIES = [g for g in ALL_GEOMETRIES if g[2] == 18 and g[4] > 50][:1]
else:
    GEOMETRIES = ALL_GEOMETRIES

PARTICLES    = ["pi+", "mu-", "neutron"]
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

def create_test_xml(test_id, layers, steel_mm, scint_mm):
    """Create klmws_{test_id}.xml and epic_klmws_only_{test_id}.xml."""
    klmws_dst = os.path.join(DETECTOR_INSTALL, "compact", "pid", f"klmws_{test_id}.xml")
    shutil.copyfile(KLMWS_TEMPLATE, klmws_dst)

    tree = ET.parse(klmws_dst)
    root = tree.getroot()
    for const in root.iter('constant'):
        name = const.get('name')
        if name == 'HcalScintillatorNbLayers':
            const.set('value', str(int(layers)))
        elif name == 'HcalSteelThickness':
            const.set('value', f'{steel_mm}*mm')
        elif name == 'HcalScintillatorThickness':
            const.set('value', f'{scint_mm}*mm')
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
    processes = []

    log_dir = os.path.join(WORK_EIC, "slurm", "mem_test_logs")
    os.makedirs(log_dir, exist_ok=True)

    for geo_name, test_id, layers, steel, scint, thick in GEOMETRIES:
        compact_file = create_test_xml(test_id, layers, steel, scint)
        print(f"  Created XML: {geo_name}  layers={layers}  steel={steel}mm  scint={scint}mm  ({thick})")

        for particle in PARTICLES:
            psafe = PARTICLE_SAFE[particle]
            run_name_pref = f"mst_{geo_name}_{psafe}"
            run_names_map[(geo_name, particle)] = run_name_pref

            cmd = [
                sys.executable, SUBMIT_SCRIPT,
                "--compactFile",    compact_file,
                "--setupPath",      SETUP_PATH,
                "--run_name_pref",  run_name_pref,
                "--runNum",         str(test_id),
                "--particle",       particle,
                "--num_simulations", str(N_REPS),
                "--skipTraining",
                "--no-classification",
                "--chPath",         EPIC_HOME,
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

def get_expected_limit(scint_mm, particle):
    """Mirror the current memory limit logic from submit_workflow.py."""
    if scint_mm > 30.0:
        if particle in ("pi+", "neutron", "kaon0L", "proton"):
            return "32G"
        else:
            return "16G"
    return "8G"


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
    print("\n" + "=" * 110)
    print("MEMORY STRESS TEST RESULTS")
    print(f"  {len(GEOMETRIES)} configs × {len(PARTICLES)} particles × {N_REPS} reps  |  MOBO space: layers 5–18, steel_ratio 0.265–0.934")
    print("=" * 110)

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
        elif '.' not in job_id and 'mst_' in job_name:
            parent_info[job_id] = (job_name.strip(), state.strip())

    stats = defaultdict(list)  # (geo_name, particle) -> [(rss_mb, state)]

    for job_id, (job_name, state) in parent_info.items():
        m = re.match(r'mst_(.+?)_(pip|mum|neutron)_', job_name)
        if not m:
            continue
        geo_name = m.group(1)
        psafe    = m.group(2)
        particle = {"pip": "pi+", "mum": "mu-", "neutron": "neutron"}[psafe]
        rss_mb   = batch_rss.get(job_id, 0.0)
        stats[(geo_name, particle)].append((rss_mb, state))

    # ── Print grouped by particle for easy limit-setting ──
    hdr = (f"{'Config':<16} | {'L':>2} | {'Scint':>6} | {'Steel':>6} | "
           f"{'Particle':<8} | {'Limit':>5} | {'N':>2} | "
           f"{'Mean MB':>8} | {'Max MB':>8} | {'Max GB':>6} | {'Headroom':>9} | OOM")
    print(hdr)
    print("-" * 110)

    any_concern = False
    recommendations = {}  # particle -> max_gb_seen across all configs

    for particle in PARTICLES:
        for geo in GEOMETRIES:
            geo_name, test_id, layers, steel, scint, thick = geo
            key = (geo_name, particle)
            entries    = stats.get(key, [])
            limit_str  = get_expected_limit(scint, particle)
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
                if oom_count > 0:
                    any_concern = True
                    flag = " OOM"
                if headroom_pct < 20:
                    any_concern = True
                # Track max usage per particle for recommendation
                prev = recommendations.get(particle, 0.0)
                recommendations[particle] = max(prev, max_gb)
            else:
                mean_mb      = 0
                max_mb       = 0
                max_gb       = 0
                headroom_str = "N/A"

            n_str = str(len(entries)) if entries else "0"
            print(f"{geo_name:<16} | {layers:>2} | {scint:>5.1f}mm | {steel:>5.1f}mm | "
                  f"{PARTICLE_SAFE[particle]:<8} | {limit_str:>5} | {n_str:>2} | "
                  f"{mean_mb:>8.0f} | {max_mb:>8.0f} | {max_gb:>6.2f} | "
                  f"{headroom_str:>9} | {oom_count}")

        print()  # blank line between particle groups

    # ── Recommendations ──
    print("=" * 110)
    print("RECOMMENDED LIMITS (max observed + 25% safety margin, rounded up to nearest 4G):")
    print()
    for particle in PARTICLES:
        max_gb = recommendations.get(particle, 0.0)
        if max_gb == 0:
            print(f"  {particle:<8}: no data")
            continue
        recommended = max_gb * 1.25
        # round up to nearest 4G
        recommended = int(((recommended + 3.999) // 4) * 4)
        current     = get_expected_limit(
            # use a thick-scint value to get the upper bound
            55.5 if particle != "mu-" else 55.5, particle
        )
        print(f"  {particle:<8}: max observed = {max_gb:.2f} GB  →  recommended ≥ {recommended}G  (current limit: {current})")

    print()
    if any_concern:
        print("WARNING: Some configurations have <20% headroom or OOM failures — see (!) flags above.")
    else:
        print("All configurations have adequate headroom (≥20%). Current limits are safe.")
    print("=" * 110)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 110)
    print("MEMORY STRESS TEST FOR MOBO GEOMETRIES")
    print("=" * 110)
    n_jobs = len(GEOMETRIES) * len(PARTICLES) * N_REPS
    print(f"  {len(GEOMETRIES)} configs × {len(PARTICLES)} particles × {N_REPS} reps = {n_jobs} SLURM jobs")
    if QUICK_MODE:
        print("  (QUICK MODE: worst-case geometry only)")
    if ANALYZE_ONLY:
        print("  (ANALYZE-ONLY: skipping submission, running sacct on today's mst_* jobs)")
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
        log_dir = os.path.join(WORK_EIC, "slurm", "mem_test_logs")
        print(f"Submission logs in: {log_dir}")
        print("Done!")


if __name__ == "__main__":
    main()
