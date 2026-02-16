#!/usr/bin/env python3
"""
Memory Stress Test for MOBO Geometries

Tests dynamic memory allocation across the MOBO parameter space to verify
that no geometry/particle combination will cause OUT_OF_MEMORY failures.

Generates test XML files with varying scint thickness and layer counts,
submits 5 jobs per geometry/particle combo via submit_workflow.py (debug_mode),
monitors completion, then uses sacct to check MaxRSS and report headroom.

Test geometries (8 configs):
  normal_14L/18L:   steel=55.5mm, scint=20.0mm  -> 8G limit
  boundary_14L/18L: steel=40.0mm, scint=30.0mm  -> 8G limit (at boundary)
  thick_14L/18L:    steel=18.0mm, scint=40.0mm   -> 16G/10G limit
  worst_14L/18L:    steel=10.0mm, scint=50.0mm   -> 16G/10G limit

Total: 8 geometries x 3 particles x 5 jobs = 120 jobs

Usage:
    source /hpc/group/vossenlab/rck32/eic/work_eic/setup.sh
    python3 /hpc/group/vossenlab/rck32/eic/work_eic/slurm/test_memory_limits.py
    python3 /hpc/group/vossenlab/rck32/eic/work_eic/slurm/test_memory_limits.py --quick  # worst_18L only
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

ALL_GEOMETRIES = [
    # (name, test_id, layers, steel_mm, scint_mm)
    ("normal_14L",   9000, 14, 55.5, 20.0),
    ("normal_18L",   9001, 18, 55.5, 20.0),
    ("boundary_14L", 9002, 14, 40.0, 30.0),
    ("boundary_18L", 9003, 18, 40.0, 30.0),
    ("thick_14L",    9004, 14, 18.0, 40.0),
    ("thick_18L",    9005, 18, 18.0, 40.0),
    ("worst_14L",    9006, 14, 10.0, 50.0),
    ("worst_18L",    9007, 18, 10.0, 50.0),
]

QUICK_MODE = "--quick" in sys.argv
if QUICK_MODE:
    GEOMETRIES = [ALL_GEOMETRIES[-1]]  # worst_18L only
    sys.argv.remove("--quick")
else:
    GEOMETRIES = ALL_GEOMETRIES

PARTICLES = ["pi+", "mu-", "neutron"]
PARTICLE_SAFE = {"pi+": "pip", "mu-": "mum", "neutron": "neutron"}

# ── Environment ────────────────────────────────────────────────────────────

try:
    WORK_EIC = os.environ['WORK_EIC']
    EPIC_HOME = os.environ['EPIC_HOME']
except KeyError as e:
    print(f"Missing env var {e}. Source work_eic/setup.sh first.")
    sys.exit(1)

SUBMIT_SCRIPT = os.path.join(WORK_EIC, "slurm", "submit_workflow.py")

# DD4hep resolves ${DETECTOR_PATH} from install/setup.sh which points to
# the install tree, so test XMLs must go there.
DETECTOR_INSTALL = os.path.join(EPIC_HOME, "install", "share", "epic")
KLMWS_TEMPLATE = os.path.join(DETECTOR_INSTALL, "compact", "pid", "klmws.xml")
EPIC_TEMPLATE = os.path.join(DETECTOR_INSTALL, "epic_klmws_only.xml")
SETUP_PATH = os.path.join(EPIC_HOME, "install", "setup.sh")

# ── XML Generation ─────────────────────────────────────────────────────────

def create_test_xml(test_id, layers, steel_mm, scint_mm):
    """Create klmws_{test_id}.xml and epic_klmws_only_{test_id}.xml with given params.
    Returns the path to the top-level compact file."""
    # 1. Copy klmws.xml -> klmws_{test_id}.xml, edit parameters
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

    # 2. Copy epic_klmws_only.xml -> epic_klmws_only_{test_id}.xml, update include ref
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
    """Remove all generated test XML files."""
    for _, test_id, _, _, _ in GEOMETRIES:
        for path in [
            os.path.join(DETECTOR_INSTALL, "compact", "pid", f"klmws_{test_id}.xml"),
            os.path.join(DETECTOR_INSTALL, f"epic_klmws_only_{test_id}.xml"),
        ]:
            if os.path.exists(path):
                os.remove(path)
                print(f"  Cleaned up {os.path.basename(path)}")


# ── debug_mode toggle ─────────────────────────────────────────────────────

def read_debug_mode():
    """Read current debug_mode value from submit_workflow.py."""
    with open(SUBMIT_SCRIPT, 'r') as f:
        content = f.read()
    match = re.search(r'debug_mode\s*=\s*(True|False)', content)
    if match:
        return match.group(1) == 'True'
    return None


def set_debug_mode(value: bool):
    """Set debug_mode in submit_workflow.py. Returns previous value."""
    with open(SUBMIT_SCRIPT, 'r') as f:
        content = f.read()

    match = re.search(r'(debug_mode\s*=\s*)(True|False)', content)
    if not match:
        print("  WARNING: Could not find debug_mode in submit_workflow.py")
        return None

    old_value = match.group(2) == 'True'
    new_str = f'{match.group(1)}{value}'
    new_content = content[:match.start()] + new_str + content[match.end():]

    with open(SUBMIT_SCRIPT, 'w') as f:
        f.write(new_content)

    if old_value != value:
        print(f"  Changed debug_mode: {old_value} -> {value}")
    else:
        print(f"  debug_mode already {value}")

    return old_value


# ── Job Submission ─────────────────────────────────────────────────────────

def submit_all_tests():
    """Generate XMLs and launch all submit_workflow.py processes in parallel.
    Returns (run_names_map, processes) where:
      run_names_map: dict of (geo_name, particle) -> run_name_pref
      processes: list of (geo_name, particle, test_id, Popen, log_file)
    """
    run_names_map = {}
    processes = []

    log_dir = os.path.join(WORK_EIC, "slurm", "mem_test_logs")
    os.makedirs(log_dir, exist_ok=True)

    for geo_name, test_id, layers, steel, scint in GEOMETRIES:
        compact_file = create_test_xml(test_id, layers, steel, scint)
        print(f"  Created XML: {geo_name} (id={test_id}, layers={layers}, steel={steel}mm, scint={scint}mm)")

        for particle in PARTICLES:
            psafe = PARTICLE_SAFE[particle]
            run_name_pref = f"mem_test_{geo_name}_{psafe}"
            run_names_map[(geo_name, particle)] = run_name_pref

            cmd = [
                sys.executable, SUBMIT_SCRIPT,
                "--compactFile", compact_file,
                "--setupPath", SETUP_PATH,
                "--run_name_pref", run_name_pref,
                "--runNum", str(test_id),
                "--particle", particle,
                "--skipTraining",
                "--no-classification",
                "--chPath", EPIC_HOME,
            ]

            log_path = os.path.join(log_dir, f"{run_name_pref}.log")
            log_file = open(log_path, 'w')
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
            processes.append((geo_name, particle, test_id, proc, log_file))
            print(f"    Launched {run_name_pref} (PID {proc.pid})")

    return run_names_map, processes


def wait_for_all(processes):
    """Wait for all submit_workflow.py processes to finish."""
    total = len(processes)
    print(f"\nWaiting for {total} submit_workflow.py processes to finish...")
    print("(Each runs 5 sim+process+analyze jobs, then monitors until done)")

    while True:
        still_running = [(g, p, t, proc, lf) for g, p, t, proc, lf in processes if proc.poll() is None]
        done = total - len(still_running)
        if not still_running:
            break
        print(f"  {done}/{total} done, {len(still_running)} running... (sleeping 30s)")
        time.sleep(30)

    # Close log files and check exit codes
    failed = []
    for geo_name, particle, test_id, proc, log_file in processes:
        log_file.close()
        if proc.returncode != 0:
            failed.append((geo_name, particle, proc.returncode))

    if failed:
        print(f"\nWARNING: {len(failed)} submit_workflow.py processes exited with errors:")
        for geo, part, rc in failed:
            print(f"  {geo} / {part}: exit code {rc}")
    else:
        print(f"\nAll {total} processes completed successfully.")

    return failed


# ── Memory Analysis ────────────────────────────────────────────────────────

def get_expected_limit(scint_mm, particle):
    """Mirror the memory limit logic from submit_workflow.py."""
    if scint_mm > 30.0:
        if particle in ("pi+", "neutron", "kaon0L", "proton"):
            return "16G"
        else:  # mu-
            return "10G"
    return "8G"


def mem_str_to_mb(val):
    """Convert sacct MaxRSS string (e.g., '4500K', '12M', '1.5G') to MB."""
    if not val or val.strip() == '' or val.strip() == '0':
        return 0.0
    val = val.strip()
    if val.endswith('K'):
        return float(val[:-1]) / 1024
    elif val.endswith('M'):
        return float(val[:-1])
    elif val.endswith('G'):
        return float(val[:-1]) * 1024
    try:
        return float(val) / 1024  # assume KB
    except ValueError:
        return 0.0


def limit_to_mb(limit_str):
    """Convert '8G' / '16G' to MB."""
    return float(limit_str.replace('G', '')) * 1024


def analyze_memory(run_names_map):
    """Use sacct to gather MaxRSS for all mem_test_* jobs and produce summary table."""
    print("\n" + "=" * 100)
    print("MEMORY STRESS TEST RESULTS")
    print("=" * 100)

    # Build comma-separated list of all run_name prefixes for sacct --name filter
    # sacct --name matches job names, but our job names have suffixes. Use -S today instead.
    result = subprocess.run(
        ["sacct", "-S", time.strftime("%Y-%m-%d"), "--parsable2",
         "--format=JobID,JobName%80,State%20,MaxRSS", "--noheader"],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"sacct failed: {result.stderr}")
        return

    # Parse: parent jobs give us job name + state, batch substeps give MaxRSS
    parent_info = {}  # job_id -> (job_name, state)
    batch_rss = {}    # parent_job_id -> MaxRSS in MB

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
        elif '.' not in job_id and 'mem_test_' in job_name:
            parent_info[job_id] = (job_name.strip(), state.strip())

    # Group by (geo_name, particle)
    stats = defaultdict(list)  # (geo_name, particle) -> [(rss_mb, state)]

    for job_id, (job_name, state) in parent_info.items():
        # Parse: mem_test_normal_14L_pip_500events_run_9000_February_14_0
        m = re.match(r'mem_test_(.+?)_(pip|mum|neutron)_', job_name)
        if not m:
            continue
        geo_name = m.group(1)
        psafe = m.group(2)
        # Reverse lookup particle
        particle = {"pip": "pi+", "mum": "mu-", "neutron": "neutron"}[psafe]

        rss_mb = batch_rss.get(job_id, 0.0)
        stats[(geo_name, particle)].append((rss_mb, state))

    # Print table
    header = f"{'Geometry':<18} | {'Particle':<8} | {'Limit':>5} | {'Jobs':>4} | {'Mean(MB)':>9} | {'Max(MB)':>8} | {'Headroom':>9} | {'OOM':>3}"
    print(header)
    print("-" * 100)

    any_concern = False

    for geo_name, test_id, layers, steel, scint in GEOMETRIES:
        for particle in PARTICLES:
            key = (geo_name, particle)
            entries = stats.get(key, [])
            limit_str = get_expected_limit(scint, particle)
            limit_mb = limit_to_mb(limit_str)

            if not entries:
                print(f"{geo_name:<18} | {PARTICLE_SAFE[particle]:<8} | {limit_str:>5} |    0 |       N/A |      N/A |       N/A | N/A")
                continue

            rss_vals = [e[0] for e in entries if e[0] > 0]
            oom_count = sum(1 for e in entries if 'OUT_OF_MEMORY' in e[1])

            if rss_vals:
                mean_mb = sum(rss_vals) / len(rss_vals)
                max_mb = max(rss_vals)
                headroom_pct = (1 - max_mb / limit_mb) * 100
                flag = ""
                if headroom_pct < 15:
                    flag = " (!)"
                    any_concern = True
                headroom_str = f"{headroom_pct:.0f}%{flag}"
            else:
                mean_mb = 0
                max_mb = 0
                headroom_str = "N/A"

            print(f"{geo_name:<18} | {PARTICLE_SAFE[particle]:<8} | {limit_str:>5} | {len(entries):>4} | {mean_mb:>9.0f} | {max_mb:>8.0f} | {headroom_str:>9} | {oom_count:>3}")

            if oom_count > 0:
                any_concern = True

    print("=" * 100)
    if any_concern:
        print("WARNING: Some configurations have <15% headroom or OOM failures!")
        print("Review the table above and consider increasing memory limits.")
    else:
        print("All configurations have adequate memory headroom. Safe to run MOBO.")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 100)
    print("MEMORY STRESS TEST FOR MOBO GEOMETRIES")
    print("=" * 100)
    n_geom = len(GEOMETRIES)
    n_part = len(PARTICLES)
    n_jobs = n_geom * n_part * 5
    print(f"Testing {n_geom} geometries x {n_part} particles x 5 jobs = {n_jobs} total jobs")
    if QUICK_MODE:
        print("(QUICK MODE: testing worst_18L only)")
    print()

    # Step 1: Ensure debug_mode = True
    print("[1/5] Setting debug_mode = True in submit_workflow.py...")
    original_debug = set_debug_mode(True)
    print()

    try:
        # Step 2: Generate XMLs and submit
        print("[2/5] Generating test XMLs and submitting jobs...")
        run_names_map, processes = submit_all_tests()
        print(f"\n  Launched {len(processes)} submit_workflow.py processes")
        print()

        # Step 3: Wait for completion
        print("[3/5] Waiting for all processes to complete...")
        failed = wait_for_all(processes)
        print()

        # Step 4: Let sacct finalize
        print("[4/5] Waiting 15s for sacct to update...")
        time.sleep(15)

        # Step 5: Analyze
        print("[5/5] Analyzing memory usage...")
        analyze_memory(run_names_map)

    finally:
        # Restore debug_mode to its original value
        print(f"\nRestoring debug_mode = {original_debug} in submit_workflow.py...")
        if original_debug is not None:
            set_debug_mode(original_debug)
        else:
            set_debug_mode(False)

        # Cleanup test XMLs
        print("\nCleaning up test XML files...")
        cleanup_test_xmls()

        # Note about logs
        log_dir = os.path.join(WORK_EIC, "slurm", "mem_test_logs")
        print(f"\nSubmission logs preserved in: {log_dir}")
        print("Done!")


if __name__ == "__main__":
    main()
