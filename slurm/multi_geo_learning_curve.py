#!/usr/bin/env python3
"""
Multi-Geometry GNN Learning Curve Study

Tests whether relative geometry rankings (by MOBO objectives) are preserved
when using fewer training events. If rankings are stable across numDfs values,
MOBO can use fewer events per trial to save compute.

Objectives tested (the 4 MOBO objectives):
  - Neutron energy RMSE: low (<2.75 GeV), high (>=2.75 GeV)
  - muID classifier AUC: low (<2.75 GeV), high (>=2.75 GeV)

Geometries:
  A: Baseline  (14 layers, steel_ratio=0.735, 55.5/20.0 mm)
  B: Few thick-steel layers (8 layers, steel_ratio=0.90, 67.95/7.55 mm)
  C: Many thin-steel layers  (17 layers, steel_ratio=0.40, 30.2/45.3 mm)

Phases:
  0: Create geometry XMLs for B and C
  1: Generate simulation data (reuse A from learning_curve.py)
  2a: Train neutron energy predictor at each numDfs
  2b: Train muID classifier at each numDfs
  3: Collect results, rank stability analysis, plots

Usage:
    source /hpc/group/vossenlab/rck32/eic/work_eic/setup.sh
    python3 slurm/multi_geo_learning_curve.py

    # Skip data generation:
    python3 slurm/multi_geo_learning_curve.py --skip-data

    # Skip training, just re-plot:
    python3 slurm/multi_geo_learning_curve.py --skip-data --skip-training

    # Skip XML creation:
    python3 slurm/multi_geo_learning_curve.py --skip-xml --skip-data
"""

import os
import sys
import re
import copy
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────

TOTAL_THICKNESS_MM = 75.5  # steel + scintillator total per layer

GEOMETRIES = {
    "A": {
        "num_layers": 14, "steel_ratio": 0.735,
        "steel_mm": 55.5, "scint_mm": 20.0,
        "label": "Baseline (14L, r=0.735)",
    },
    "B": {
        "num_layers": 8, "steel_ratio": 0.90,
        "steel_mm": round(TOTAL_THICKNESS_MM * 0.90, 2),
        "scint_mm": round(TOTAL_THICKNESS_MM * 0.10, 2),
        "label": "Few thick-steel (8L, r=0.90)",
    },
    "C": {
        "num_layers": 17, "steel_ratio": 0.40,
        "steel_mm": round(TOTAL_THICKNESS_MM * 0.40, 2),
        "scint_mm": round(TOTAL_THICKNESS_MM * 0.60, 2),
        "label": "Many thin-steel (17L, r=0.40)",
    },
}

GEO_IDS = list(GEOMETRIES.keys())

# Particles needed: neutron (energy), mu- and pi+ (classifier)
PARTICLES = ["neutron", "mu-", "pi+"]
PARTICLE_SAFE = {"pi+": "pip", "mu-": "mum", "neutron": "neutron"}

NUM_EVENTS_PER_SIM = 500

SKIP_DATA = "--skip-data" in sys.argv
SKIP_TRAINING = "--skip-training" in sys.argv
SKIP_XML = "--skip-xml" in sys.argv
TEST_MODE = "--test" in sys.argv

# In test mode: 1 geometry (B), neutron only, 5 sims (debug_mode), 1 numDfs — 6 total jobs
if TEST_MODE:
    NUMDF_VALUES = [5]
    NUM_SIMS = 5   # debug_mode=True in submit_workflow.py hardcodes 5 sims
else:
    NUMDF_VALUES = [10, 20, 30, 50]
    NUM_SIMS = 50

# ── Environment ────────────────────────────────────────────────────────────

try:
    WORK_EIC = os.environ['WORK_EIC']
    EPIC_HOME = os.environ['EPIC_HOME']
    ML_VENV_HOME = os.environ['ML_VENV_HOME']
    MAIL_USER = os.environ['MAIL_USER']
except KeyError as e:
    print(f"Missing env var {e}. Source work_eic/setup.sh first.")
    sys.exit(1)

SUBMIT_SCRIPT = os.path.join(WORK_EIC, "slurm", "submit_workflow.py")
TIMING_PATH = os.path.join(WORK_EIC, "macros", "Timing_estimation")
CHECK_STATUS = os.path.join(WORK_EIC, "slurm", "util", "checkSlurmStatus.sh")
EXCLUDE_NODES = "dcc-youlab-gpu-28,dcc-gehmlab-gpu-ferc-s-z25-18,dcc-brunellab-gpu-[01-04],dcc-carlsonlab-gpu-[09-12],dcc-chsi-gpu-[05-08]"

# Install directory — where DETECTOR_PATH resolves inside eic-shell
INSTALL_DIR = os.path.join(EPIC_HOME, "install", "share", "epic")

# Output directories
RESULTS_DIR = os.path.join(TIMING_PATH, "results", "multi_geo_lc")
PLOTS_DIR = os.path.join(TIMING_PATH, "plots", "multi_geo_lc")
MODELS_DIR = os.path.join(TIMING_PATH, "models", "multi_geo_lc")
SHELLS_DIR = os.path.join(WORK_EIC, "slurm", "shells")

# Baseline learning curve results (geometry A reuse)
BASELINE_RESULTS_DIR = os.path.join(TIMING_PATH, "results", "learning_curve")

current_date = datetime.now().strftime("%B_%d")


# ── Utilities ──────────────────────────────────────────────────────────────

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def csv_prefix_for(geo_id, particle):
    """Path prefix for CSVs: append '{i}.csv' to get each file."""
    psafe = PARTICLE_SAFE[particle]
    if geo_id == "A":
        # Reuse baseline learning_curve data
        return os.path.join(TIMING_PATH, "data", "df",
                            f"learning_curve_{psafe}_{NUM_EVENTS_PER_SIM}events_run_0_")
    else:
        return os.path.join(TIMING_PATH, "data", "df",
                            f"mgeo_{geo_id}_{psafe}_{NUM_EVENTS_PER_SIM}events_run_0_")


def compact_file_for(geo_id):
    """Path to the top-level compact XML for a geometry.
    Uses the install directory since DETECTOR_PATH resolves there inside eic-shell."""
    if geo_id == "A":
        return os.path.join(INSTALL_DIR, "epic_klmws_only.xml")
    else:
        return os.path.join(INSTALL_DIR, f"epic_klmws_only_geo{geo_id}.xml")


def get_job_status(job_id):
    """Check SLURM job status. Returns 0=running, 1=completed, -1=failed."""
    result = subprocess.run([CHECK_STATUS, str(job_id)], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8').split()
    if len(output) != 1:
        return 0
    status = output[0]
    if status == "0":
        return 0
    elif status == "1":
        return 1
    elif status == "-1":
        return -1
    return 0


def set_debug_mode(enabled):
    """Set debug_mode in submit_workflow.py. Returns previous value."""
    with open(SUBMIT_SCRIPT, 'r') as f:
        content = f.read()
    match = re.search(r'debug_mode\s*=\s*(True|False)', content)
    if not match:
        print("  WARNING: Could not find debug_mode in submit_workflow.py")
        return None
    prev = match.group(1)
    target = "True" if enabled else "False"
    if prev != target:
        new_content = content[:match.start(1)] + target + content[match.end(1):]
        with open(SUBMIT_SCRIPT, 'w') as f:
            f.write(new_content)
    print(f"  debug_mode = {target} (was {prev})")
    return prev


def verify_debug_mode():
    """Ensure debug_mode matches what we need: True for test mode, False for production."""
    want = TEST_MODE
    with open(SUBMIT_SCRIPT, 'r') as f:
        content = f.read()
    match = re.search(r'debug_mode\s*=\s*(True|False)', content)
    if not match:
        print("  WARNING: Could not find debug_mode in submit_workflow.py")
        return
    current = match.group(1) == 'True'
    if current != want:
        set_debug_mode(want)
    else:
        print(f"  debug_mode = {match.group(1)} (OK)")


def wait_for_jobs(job_ids, label="jobs"):
    """Poll until all jobs in dict finish. Returns list of failed keys."""
    total = len(job_ids)
    if total == 0:
        return []
    print(f"\n  Waiting for {total} {label} to complete...")

    completed = set()
    failed = []

    while len(completed) < total:
        for key, job_id in job_ids.items():
            if key in completed:
                continue
            status = get_job_status(job_id)
            if status == 1:
                completed.add(key)
            elif status == -1:
                completed.add(key)
                failed.append(key)

        if len(completed) < total:
            print(f"  {len(completed)}/{total} done... (sleeping 30s)")
            time.sleep(30)

    if failed:
        print(f"\n  WARNING: {len(failed)} {label} failed: {failed}")
    else:
        print(f"\n  All {total} {label} completed successfully.")

    return failed


# ── Phase 0: Create Geometry XMLs ─────────────────────────────────────────

def create_geometry_xmls():
    """Create custom geometry XML files for geometries B and C.
    Files are placed in the install directory since DETECTOR_PATH resolves there
    inside eic-shell (not the source tree)."""
    print("\n[Phase 0] Creating geometry XMLs")

    # Use install dir as base — that's where DETECTOR_PATH points inside eic-shell
    base_klmws = os.path.join(INSTALL_DIR, "compact", "pid", "klmws.xml")
    base_compact = os.path.join(INSTALL_DIR, "epic_klmws_only.xml")

    for geo_id in ["B", "C"]:
        geo = GEOMETRIES[geo_id]
        klmws_out = os.path.join(INSTALL_DIR, "compact", "pid", f"klmws_geo{geo_id}.xml")
        compact_out = os.path.join(INSTALL_DIR, f"epic_klmws_only_geo{geo_id}.xml")

        # Create modified klmws XML
        tree = ET.parse(base_klmws)
        root = tree.getroot()

        for const in root.iter('constant'):
            name = const.get('name')
            if name == 'HcalSteelThickness':
                const.set('value', f"{geo['steel_mm']}*mm")
            elif name == 'HcalScintillatorThickness':
                const.set('value', f"{geo['scint_mm']}*mm")
            elif name == 'HcalScintillatorNbLayers':
                const.set('value', str(geo['num_layers']))

        tree.write(klmws_out)
        print(f"  Created {klmws_out}")
        print(f"    layers={geo['num_layers']}, steel={geo['steel_mm']}mm, scint={geo['scint_mm']}mm")

        # Create top-level compact XML pointing to the geo-specific klmws
        shutil.copyfile(base_compact, compact_out)
        ctree = ET.parse(compact_out)
        croot = ctree.getroot()

        for inc in croot.iter('include'):
            ref = inc.get('ref', '')
            if ref.endswith('klmws.xml') and 'klmws_geo' not in ref:
                new_ref = ref.replace('klmws.xml', f'klmws_geo{geo_id}.xml')
                inc.set('ref', new_ref)

        ctree.write(compact_out)
        print(f"  Created {compact_out}")


# ── Phase 1: Data Generation ──────────────────────────────────────────────

def generate_data():
    """Generate simulation data for geometries B and C. Reuse A."""
    new_geos = ["B"] if TEST_MODE else ["B", "C"]
    data_particles = ["neutron"] if TEST_MODE else PARTICLES

    n_jobs = len(new_geos) * len(data_particles) * NUM_SIMS
    print(f"\n[Phase 1] Generating data ({len(new_geos)} geos × {len(data_particles)} particles × {NUM_SIMS} sims = {n_jobs} jobs)")

    if not TEST_MODE:
        # Verify baseline data exists for geometry A
        print("  Checking geometry A (baseline) CSVs...")
        for particle in PARTICLES:
            prefix = csv_prefix_for("A", particle)
            n_found = sum(1 for i in range(NUM_SIMS) if os.path.isfile(f"{prefix}{i}.csv"))
            print(f"    {particle}: {n_found}/{NUM_SIMS} CSVs")
            if n_found < NUM_SIMS * 0.8:
                print(f"    WARNING: Only {n_found} CSVs for {particle}. "
                      "Run learning_curve.py first to generate baseline data.")

    log_dir = os.path.join(WORK_EIC, "slurm", "multi_geo_lc_logs")
    ensure_dir(log_dir)

    processes = []
    for geo_id in new_geos:
        compact = compact_file_for(geo_id)
        for particle in data_particles:
            psafe = PARTICLE_SAFE[particle]
            rn_pref = f"mgeo_{geo_id}_{psafe}"

            cmd = [
                sys.executable, SUBMIT_SCRIPT,
                "--run_name_pref", rn_pref,
                "--particle", particle,
                "--compactFile", compact,
                "--skipTraining",
                "--no-classification",
                "--runNum", "0",
            ]

            log_path = os.path.join(log_dir, f"data_gen_{geo_id}_{psafe}.log")
            log_file = open(log_path, 'w')
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
            processes.append((geo_id, particle, proc, log_file))
            print(f"  Launched data gen: geo {geo_id}, {particle} (PID {proc.pid})")

    # Wait for all to finish
    print(f"\n  Waiting for {len(processes)} data generation processes...")
    while True:
        running = [(g, p, proc, lf) for g, p, proc, lf in processes if proc.poll() is None]
        done = len(processes) - len(running)
        if not running:
            break
        print(f"  {done}/{len(processes)} done, {len(running)} running... (sleeping 30s)")
        time.sleep(30)

    # Check results
    all_ok = True
    for geo_id, particle, proc, log_file in processes:
        log_file.close()
        if proc.returncode != 0:
            print(f"  ERROR: Data gen for geo {geo_id} {particle} failed (rc={proc.returncode})")
            all_ok = False

    if not all_ok:
        print("  Some data generation processes failed. Check logs in multi_geo_lc_logs/")
        sys.exit(1)

    # Verify CSVs
    for geo_id in new_geos:
        for particle in data_particles:
            prefix = csv_prefix_for(geo_id, particle)
            n_found = sum(1 for i in range(NUM_SIMS) if os.path.isfile(f"{prefix}{i}.csv"))
            print(f"  geo {geo_id} {particle}: {n_found}/{NUM_SIMS} CSVs")

    print("  Phase 1 complete.")


# ── Phase 2a: Neutron Energy Training ─────────────────────────────────────

def submit_energy_jobs():
    """Submit neutron energy prediction training jobs for each (geo, numDfs).
    For geometry A, reuse existing results. Returns job_ids dict and shell list."""

    train_geos = ["B"] if TEST_MODE else GEO_IDS
    print(f"\n[Phase 2a] Submitting neutron energy training jobs (geos: {train_geos})")

    ensure_dir(RESULTS_DIR)
    ensure_dir(PLOTS_DIR)
    ensure_dir(SHELLS_DIR)

    out_folder = os.path.join(WORK_EIC, "slurm", "output", f"output{current_date}")
    error_folder = os.path.join(WORK_EIC, "slurm", "error", f"error{current_date}")
    ensure_dir(out_folder)
    ensure_dir(error_folder)

    if not TEST_MODE:
        # Check what baseline results already exist for geo A
        print("  Checking geometry A baseline results...")
        for n in NUMDF_VALUES:
            baseline_file = os.path.join(BASELINE_RESULTS_DIR, f"neutron_n{n}.txt")
            if os.path.isfile(baseline_file):
                print(f"    neutron_n{n}.txt: exists (will reuse)")
            else:
                print(f"    neutron_n{n}.txt: MISSING (will need to train)")

    job_ids = {}
    shell_scripts = []

    for geo_id in train_geos:
        prefix = csv_prefix_for(geo_id, "neutron")

        for n in NUMDF_VALUES:
            # For geo A, check if baseline result exists
            if geo_id == "A":
                baseline_file = os.path.join(BASELINE_RESULTS_DIR, f"neutron_n{n}.txt")
                if os.path.isfile(baseline_file):
                    continue  # Skip, will read from baseline

            job_name = f"mgeo_{geo_id}_neutron_n{n}"
            results_file = os.path.join(RESULTS_DIR, f"energy_{geo_id}_neutron_n{n}.txt")
            model_dir = os.path.join(MODELS_DIR, f"{geo_id}_neutron_n{n}")
            ensure_dir(model_dir)

            shell_path = os.path.join(SHELLS_DIR, f"mgeo_energy_{geo_id}_n{n}.sh")
            shell_scripts.append(shell_path)

            with open(shell_path, 'w') as f:
                f.write(f"""#!/bin/bash
#SBATCH --chdir={EPIC_HOME}
#SBATCH --job-name={job_name}
#SBATCH --output={out_folder}/%x.out
#SBATCH --error={error_folder}/%x.err
#SBATCH -p scavenger-gpu
#SBATCH --time=00:45:00
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --mem=60G
#SBATCH --gpus=1
#SBATCH --mail-user={MAIL_USER}
#SBATCH --mail-type=FAIL
#SBATCH --exclude={EXCLUDE_NODES}
set -eo pipefail

echo "Multi-geo energy training: geo {geo_id}, numDfs={n}"
source {ML_VENV_HOME}/bin/activate
python3 {TIMING_PATH}/train_GNN.py \
  --numDfs {n} \
  --runNum 0 \
  --inputDataPref "{prefix}" \
  --modelPath "{model_dir}/" \
  --resultsFilePath "{results_file}" \
  --lossPlotPath "{PLOTS_DIR}/" \
  --testPlotPath "{PLOTS_DIR}/" \
  --resultsPlotPath "{PLOTS_DIR}/" \
  --runName "{job_name}" \
  --particle "neutron" \
  --writeHighEnergyObjective \
  --writeLowEnergyObjective \
  --no-deleteDfs
echo "Training complete"
""")

            result = subprocess.run(["sbatch", shell_path], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  ERROR submitting {job_name}: {result.stderr}")
                continue
            job_id = result.stdout.strip().split()[-1]
            job_ids[(geo_id, n)] = job_id
            print(f"  Submitted {job_name} -> job {job_id}")

    return job_ids, shell_scripts


# ── Phase 2b: Classifier Training ─────────────────────────────────────────

def submit_classifier_jobs():
    """Submit muID classifier training jobs for each (geo, numDfs).
    For geometry A, reuse existing results. Returns job_ids dict and shell list."""

    if TEST_MODE:
        print(f"\n[Phase 2b] Skipping classifier in test mode")
        return {}, []

    print(f"\n[Phase 2b] Submitting classifier training jobs")

    ensure_dir(RESULTS_DIR)
    ensure_dir(SHELLS_DIR)

    out_folder = os.path.join(WORK_EIC, "slurm", "output", f"output{current_date}")
    error_folder = os.path.join(WORK_EIC, "slurm", "error", f"error{current_date}")
    ensure_dir(out_folder)
    ensure_dir(error_folder)

    # Check baseline classifier results for geo A
    print("  Checking geometry A baseline classifier results...")
    for n in NUMDF_VALUES:
        baseline_file = os.path.join(BASELINE_RESULTS_DIR, f"classifier_n{n}.txt")
        if os.path.isfile(baseline_file):
            print(f"    classifier_n{n}.txt: exists (will reuse)")
        else:
            print(f"    classifier_n{n}.txt: MISSING (will need to train)")

    job_ids = {}
    shell_scripts = []

    for geo_id in GEO_IDS:
        mu_prefix = csv_prefix_for(geo_id, "mu-")
        pi_prefix = csv_prefix_for(geo_id, "pi+")

        for n in NUMDF_VALUES:
            # For geo A, check if baseline result exists
            if geo_id == "A":
                baseline_file = os.path.join(BASELINE_RESULTS_DIR, f"classifier_n{n}.txt")
                if os.path.isfile(baseline_file):
                    continue

            job_name = f"mgeo_{geo_id}_cls_n{n}"
            results_file = os.path.join(RESULTS_DIR, f"classifier_{geo_id}_n{n}.txt")
            model_dir = os.path.join(MODELS_DIR, f"classifier_{geo_id}_n{n}")
            ensure_dir(model_dir)

            shell_path = os.path.join(SHELLS_DIR, f"mgeo_cls_{geo_id}_n{n}.sh")
            shell_scripts.append(shell_path)

            with open(shell_path, 'w') as f:
                f.write(f"""#!/bin/bash
#SBATCH --chdir={EPIC_HOME}
#SBATCH --job-name={job_name}
#SBATCH --output={out_folder}/%x.out
#SBATCH --error={error_folder}/%x.err
#SBATCH -p scavenger-gpu
#SBATCH --time=00:45:00
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --gpus=1
#SBATCH --mail-user={MAIL_USER}
#SBATCH --mail-type=FAIL
#SBATCH --exclude={EXCLUDE_NODES}
set -eo pipefail

echo "Multi-geo classifier: geo {geo_id}, numDfs={n}"
source {ML_VENV_HOME}/bin/activate
python3 {TIMING_PATH}/train_GNN_classifier.py \
  --inputDataPrefMu "{mu_prefix}" \
  --inputDataPrefPi "{pi_prefix}" \
  --numDfs {n} \
  --runNum 0 \
  --modelPath "{model_dir}/" \
  --resultsFilePath "{results_file}" \
  --testPlotPath "{PLOTS_DIR}/" \
  --runName "{job_name}" \
  --no-deleteDfs
echo "Classifier training complete"
""")

            result = subprocess.run(["sbatch", shell_path], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  ERROR submitting {job_name}: {result.stderr}")
                continue
            job_id = result.stdout.strip().split()[-1]
            job_ids[(geo_id, n)] = job_id
            print(f"  Submitted {job_name} -> job {job_id}")

    return job_ids, shell_scripts


# ── Phase 3: Collect Results ───────────────────────────────────────────────

def collect_all_results():
    """Collect all energy and classifier results.
    Returns:
        energy: dict of (geo_id, n) -> (low_rmse, high_rmse)
        classifier: dict of (geo_id, n) -> (low_auc, high_auc)
    """
    energy = {}
    classifier = {}

    for geo_id in GEO_IDS:
        for n in NUMDF_VALUES:
            # Energy results
            if geo_id == "A":
                # Try baseline first, then multi_geo_lc
                efile = os.path.join(BASELINE_RESULTS_DIR, f"neutron_n{n}.txt")
                if not os.path.isfile(efile):
                    efile = os.path.join(RESULTS_DIR, f"energy_A_neutron_n{n}.txt")
            else:
                efile = os.path.join(RESULTS_DIR, f"energy_{geo_id}_neutron_n{n}.txt")

            if os.path.isfile(efile):
                with open(efile, 'r') as f:
                    lines = f.read().strip().split('\n')
                if len(lines) >= 2:
                    try:
                        energy[(geo_id, n)] = (float(lines[0]), float(lines[1]))
                    except ValueError:
                        print(f"  Bad data in {efile}: {lines}")

            # Classifier results
            if geo_id == "A":
                cfile = os.path.join(BASELINE_RESULTS_DIR, f"classifier_n{n}.txt")
                if not os.path.isfile(cfile):
                    cfile = os.path.join(RESULTS_DIR, f"classifier_A_n{n}.txt")
            else:
                cfile = os.path.join(RESULTS_DIR, f"classifier_{geo_id}_n{n}.txt")

            if os.path.isfile(cfile):
                with open(cfile, 'r') as f:
                    lines = [l.strip() for l in f.read().strip().split('\n') if l.strip()]
                if len(lines) >= 2:
                    try:
                        # Classifier appends, take last 2 lines
                        classifier[(geo_id, n)] = (float(lines[-2]), float(lines[-1]))
                    except ValueError:
                        print(f"  Bad data in {cfile}: {lines}")

    return energy, classifier


# ── Phase 3b: Rank Stability Analysis ─────────────────────────────────────

def rank_stability_analysis(energy, classifier):
    """Analyze whether geometry rankings are preserved across numDfs values.
    Returns a dict of metric -> {n -> {geo_id -> rank}}."""

    metrics = {}

    # Build per-metric value tables
    for metric_name, source, idx in [
        ("low_RMSE", energy, 0),
        ("high_RMSE", energy, 1),
        ("low_AUC", classifier, 0),
        ("high_AUC", classifier, 1),
    ]:
        rankings = {}
        for n in NUMDF_VALUES:
            values = {}
            for geo_id in GEO_IDS:
                key = (geo_id, n)
                if key in source:
                    values[geo_id] = source[key][idx]

            if len(values) < 2:
                continue

            # Rank: for RMSE lower is better (rank 1), for AUC higher is better (rank 1)
            reverse = "AUC" in metric_name
            sorted_geos = sorted(values.keys(), key=lambda g: values[g], reverse=reverse)
            rankings[n] = {g: rank + 1 for rank, g in enumerate(sorted_geos)}

        metrics[metric_name] = rankings

    return metrics


def print_rank_table(metrics, energy, classifier):
    """Print rank stability analysis."""
    print("\n" + "=" * 100)
    print("RANK STABILITY ANALYSIS")
    print("=" * 100)
    print(f"Reference: numDfs={NUMDF_VALUES[-1]} ({NUMDF_VALUES[-1] * NUM_EVENTS_PER_SIM} events)")
    print()

    ref_n = NUMDF_VALUES[-1]
    all_stable = True

    for metric_name, rankings in metrics.items():
        print(f"  {metric_name}:")
        ref_ranks = rankings.get(ref_n, {})

        header = f"    {'numDfs':>6} | {'Events':>7}"
        for geo_id in GEO_IDS:
            header += f" | Geo {geo_id:>1} rank"
        header += " | Stable?"
        print(header)
        print("    " + "-" * (len(header) - 4))

        for n in NUMDF_VALUES:
            if n not in rankings:
                continue
            ranks = rankings[n]
            line = f"    {n:>6} | {n * NUM_EVENTS_PER_SIM:>7}"
            for geo_id in GEO_IDS:
                r = ranks.get(geo_id, "-")
                line += f" | {str(r):>10}"

            # Check if ranks match reference
            stable = (n == ref_n) or all(
                ranks.get(g) == ref_ranks.get(g) for g in GEO_IDS if g in ranks and g in ref_ranks
            )
            line += f" | {'YES' if stable else 'NO':>7}"
            if not stable:
                all_stable = False
            print(line)
        print()

    # Summary
    print("-" * 100)
    if all_stable:
        # Find minimum numDfs where all metrics are stable
        min_stable = NUMDF_VALUES[0]
        for n in NUMDF_VALUES[:-1]:
            stable_at_n = True
            for metric_name, rankings in metrics.items():
                ref_ranks = rankings.get(ref_n, {})
                cur_ranks = rankings.get(n, {})
                if not all(cur_ranks.get(g) == ref_ranks.get(g)
                           for g in GEO_IDS if g in cur_ranks and g in ref_ranks):
                    stable_at_n = False
                    break
            if stable_at_n:
                min_stable = n
                break

        print(f"  RESULT: Rankings are STABLE across all numDfs values.")
        print(f"  Minimum numDfs with consistent rankings: {min_stable} "
              f"({min_stable * NUM_EVENTS_PER_SIM} events)")
        print(f"  -> MOBO could safely use numDfs={min_stable} to save compute.")
    else:
        print(f"  RESULT: Rankings are NOT fully stable across numDfs values.")
        print(f"  Some rank inversions detected. Consider using numDfs={ref_n} for accuracy.")

    print("=" * 100)


# ── Phase 4: Plots ─────────────────────────────────────────────────────────

def plot_results(energy, classifier, metrics):
    """Plot objective values vs numDfs for each geometry."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plots")
        return

    ensure_dir(PLOTS_DIR)

    colors = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green"}
    markers = {"A": "o", "B": "s", "C": "^"}

    # Plot 1: 2×2 grid of objectives vs events
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    plot_configs = [
        (axes[0, 0], "low_RMSE", energy, 0, "Low E RMSE (<2.75 GeV)", "RMSE (GeV)", False),
        (axes[0, 1], "high_RMSE", energy, 1, "High E RMSE (>=2.75 GeV)", "RMSE (GeV)", False),
        (axes[1, 0], "low_AUC", classifier, 0, "Low E muID AUC (<2.75 GeV)", "ROC AUC", True),
        (axes[1, 1], "high_AUC", classifier, 1, "High E muID AUC (>=2.75 GeV)", "ROC AUC", True),
    ]

    for ax, metric_name, source, idx, title, ylabel, is_auc in plot_configs:
        for geo_id in GEO_IDS:
            x_vals, y_vals = [], []
            for n in NUMDF_VALUES:
                key = (geo_id, n)
                if key in source:
                    x_vals.append(n * NUM_EVENTS_PER_SIM)
                    y_vals.append(source[key][idx])
            if x_vals:
                ax.plot(x_vals, y_vals, marker=markers[geo_id], color=colors[geo_id],
                        label=GEOMETRIES[geo_id]["label"], linewidth=2, markersize=8)

        ax.set_xlabel("Number of Events", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        if is_auc:
            all_vals = [source[(g, n)][idx] for g in GEO_IDS for n in NUMDF_VALUES
                        if (g, n) in source]
            if all_vals:
                ax.set_ylim(bottom=max(0.5, min(all_vals) - 0.05))

    fig.suptitle("Multi-Geometry Learning Curve: Do Rankings Stay Consistent?",
                 fontsize=16, y=1.01)
    fig.tight_layout()

    plot_path = os.path.join(PLOTS_DIR, "multi_geo_lc_objectives.pdf")
    fig.savefig(plot_path, bbox_inches='tight')
    png_path = os.path.join(PLOTS_DIR, "multi_geo_lc_objectives.png")
    fig.savefig(png_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved objectives plot to {plot_path}")

    # Plot 2: Rank visualization
    ref_n = NUMDF_VALUES[-1]
    fig2, axes2 = plt.subplots(1, 4, figsize=(18, 5))

    for i, (metric_name, ax) in enumerate(zip(
            ["low_RMSE", "high_RMSE", "low_AUC", "high_AUC"], axes2)):
        rankings = metrics.get(metric_name, {})
        for geo_id in GEO_IDS:
            x_vals, r_vals = [], []
            for n in NUMDF_VALUES:
                if n in rankings and geo_id in rankings[n]:
                    x_vals.append(n * NUM_EVENTS_PER_SIM)
                    r_vals.append(rankings[n][geo_id])
            if x_vals:
                ax.plot(x_vals, r_vals, marker=markers[geo_id], color=colors[geo_id],
                        label=GEOMETRIES[geo_id]["label"], linewidth=2, markersize=10)

        ax.set_xlabel("Number of Events", fontsize=11)
        ax.set_ylabel("Rank (1=best)", fontsize=11)
        ax.set_title(metric_name, fontsize=13)
        ax.set_yticks([1, 2, 3])
        ax.set_ylim(0.5, 3.5)
        ax.invert_yaxis()
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig2.suptitle("Geometry Rankings vs Training Data Size", fontsize=15, y=1.02)
    fig2.tight_layout()

    rank_path = os.path.join(PLOTS_DIR, "multi_geo_lc_ranks.pdf")
    fig2.savefig(rank_path, bbox_inches='tight')
    fig2.savefig(rank_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    plt.close(fig2)
    print(f"  Saved rank plot to {rank_path}")


# ── Print Summary Table ───────────────────────────────────────────────────

def print_summary_table(energy, classifier):
    """Print objective values for all geometries."""
    print("\n" + "=" * 100)
    print("MULTI-GEOMETRY LEARNING CURVE RESULTS")
    print("=" * 100)

    print("\n  Neutron Energy RMSE:")
    print(f"  {'Geo':>3} | {'numDfs':>6} | {'Events':>7} | {'Low RMSE':>10} | {'High RMSE':>10}")
    print("  " + "-" * 55)
    for geo_id in GEO_IDS:
        for n in NUMDF_VALUES:
            key = (geo_id, n)
            if key in energy:
                low, high = energy[key]
                print(f"  {geo_id:>3} | {n:>6} | {n * NUM_EVENTS_PER_SIM:>7} | {low:>10.4f} | {high:>10.4f}")
            else:
                print(f"  {geo_id:>3} | {n:>6} | {n * NUM_EVENTS_PER_SIM:>7} | {'N/A':>10} | {'N/A':>10}")
        print("  " + "-" * 55)

    print("\n  muID Classifier AUC:")
    print(f"  {'Geo':>3} | {'numDfs':>6} | {'Events':>7} | {'Low AUC':>10} | {'High AUC':>10}")
    print("  " + "-" * 55)
    for geo_id in GEO_IDS:
        for n in NUMDF_VALUES:
            key = (geo_id, n)
            if key in classifier:
                low, high = classifier[key]
                print(f"  {geo_id:>3} | {n:>6} | {n * NUM_EVENTS_PER_SIM:>7} | {low:>10.4f} | {high:>10.4f}")
            else:
                print(f"  {geo_id:>3} | {n:>6} | {n * NUM_EVENTS_PER_SIM:>7} | {'N/A':>10} | {'N/A':>10}")
        print("  " + "-" * 55)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 100)
    if TEST_MODE:
        print("MULTI-GEOMETRY LEARNING CURVE STUDY [TEST MODE]")
        print("  Test: geo B only, neutron only, 2 sims, 1 training job")
    else:
        print("MULTI-GEOMETRY LEARNING CURVE STUDY")
    print("=" * 100)
    print()
    for geo_id, geo in GEOMETRIES.items():
        print(f"  Geometry {geo_id}: {geo['label']}")
        print(f"    layers={geo['num_layers']}, steel={geo['steel_mm']}mm, "
              f"scint={geo['scint_mm']}mm (ratio={geo['steel_ratio']})")
    print(f"\n  numDfs sweep: {NUMDF_VALUES}")
    print(f"  Sims per particle: {NUM_SIMS}")
    print(f"  Events per sim: {NUM_EVENTS_PER_SIM}")
    print()

    prev_debug = None
    try:
        # Verify/set debug_mode
        if not SKIP_DATA:
            print("Checking submit_workflow.py settings...")
            prev_debug = set_debug_mode(TEST_MODE)

        # Phase 0: XML creation
        if SKIP_XML:
            print("\n[Phase 0] Skipping XML creation (--skip-xml)")
        else:
            create_geometry_xmls()

        # Phase 1: Data generation
        if SKIP_DATA:
            print("\n[Phase 1] Skipping data generation (--skip-data)")
            check_geos = ["B"] if TEST_MODE else GEO_IDS
            check_particles = ["neutron"] if TEST_MODE else PARTICLES
            for geo_id in check_geos:
                for particle in check_particles:
                    prefix = csv_prefix_for(geo_id, particle)
                    n_found = sum(1 for i in range(NUM_SIMS) if os.path.isfile(f"{prefix}{i}.csv"))
                    print(f"  geo {geo_id} {particle}: {n_found}/{NUM_SIMS} CSVs")
        else:
            generate_data()

        # Phase 2: Training
        if SKIP_TRAINING:
            print("\n[Phase 2] Skipping training (--skip-training)")
        else:
            energy_job_ids, energy_shells = submit_energy_jobs()
            cls_job_ids, cls_shells = submit_classifier_jobs()

            energy_failed = wait_for_jobs(energy_job_ids, "energy training jobs")
            cls_failed = wait_for_jobs(cls_job_ids, "classifier training jobs")

            # Cleanup shell scripts
            for script in energy_shells + cls_shells:
                p = Path(script)
                if p.is_file():
                    p.unlink()

            if energy_failed:
                print(f"\n  {len(energy_failed)} energy jobs failed.")
            if cls_failed:
                print(f"\n  {len(cls_failed)} classifier jobs failed.")

        # Phase 3: Collect and analyze
        print("\n[Phase 3] Collecting results...")
        energy, classifier = collect_all_results()

        if not energy and not classifier:
            print("  No results found! Check training job outputs.")
            sys.exit(1)

        print(f"  Found {len(energy)} energy results, {len(classifier)} classifier results")

        print_summary_table(energy, classifier)

        if not TEST_MODE:
            metrics = rank_stability_analysis(energy, classifier)
            print_rank_table(metrics, energy, classifier)

            # Phase 4: Plots
            print("\n[Phase 4] Generating plots...")
            plot_results(energy, classifier, metrics)
        else:
            print("\n  [TEST MODE] Skipping rank analysis and plots (only 1 geometry)")

        # Cleanup generated XMLs from install directory
        for geo_id in ["B", "C"]:
            klmws = os.path.join(INSTALL_DIR, "compact", "pid", f"klmws_geo{geo_id}.xml")
            compact = os.path.join(INSTALL_DIR, f"epic_klmws_only_geo{geo_id}.xml")
            for f in [klmws, compact]:
                if os.path.isfile(f):
                    os.remove(f)
                    print(f"  Cleaned up {f}")

        print("\nDone!")

    finally:
        # Restore debug_mode if we changed it
        if prev_debug is not None:
            with open(SUBMIT_SCRIPT, 'r') as f:
                content = f.read()
            match = re.search(r'debug_mode\s*=\s*(True|False)', content)
            if match and match.group(1) != prev_debug:
                new_content = content[:match.start(1)] + prev_debug + content[match.end(1):]
                with open(SUBMIT_SCRIPT, 'w') as f:
                    f.write(new_content)
                print(f"  Restored debug_mode = {prev_debug} in submit_workflow.py")


if __name__ == "__main__":
    main()
