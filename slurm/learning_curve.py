#!/usr/bin/env python3
"""
GNN Learning Curve Study

Measures how GNN energy prediction and mu/pi classification (muID) performance
change with the number of training events, for pi+, mu-, and neutron on the
baseline geometry.

Approach:
  Phase 1:  Generate 50 sims × 500 events per particle (once)
  Phase 2a: Train GNN energy predictor with numDfs = 5..50 (per particle)
  Phase 2b: Train GNN classifier (mu-/pi+ ID) with numDfs = 5..50
  Phase 3:  Collect binned RMSE + binned AUC results and plot learning curves

Usage:
    source /hpc/group/vossenlab/rck32/eic/work_eic/setup.sh
    python3 /hpc/group/vossenlab/rck32/eic/work_eic/slurm/learning_curve.py

    # Skip data generation if CSVs already exist:
    python3 slurm/learning_curve.py --skip-data

    # Skip both data and training, just re-plot:
    python3 slurm/learning_curve.py --skip-data --skip-training
"""

import os
import sys
import re
import subprocess
import time
import numpy as np
from datetime import datetime
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────

PARTICLES = ["pi+", "mu-", "neutron"]
PARTICLE_SAFE = {"pi+": "pip", "mu-": "mum", "neutron": "neutron"}
NUMDF_VALUES = [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90]
#NUMDF_VALUES = [ 75, 90]
NUM_EVENTS_PER_SIM = 500

SKIP_DATA = "--skip-data" in sys.argv
SKIP_TRAINING = "--skip-training" in sys.argv
ONLY_CLASS = "--only-classification" in sys.argv
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

# Output directories
RESULTS_DIR = os.path.join(TIMING_PATH, "results", "learning_curve")
PLOTS_DIR = os.path.join(TIMING_PATH, "plots", "learning_curve")
MODELS_DIR = os.path.join(TIMING_PATH, "models", "learning_curve")
SHELLS_DIR = os.path.join(WORK_EIC, "slurm", "shells")

current_date = datetime.now().strftime("%B_%d")

# ── Utilities ──────────────────────────────────────────────────────────────

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def run_name_for(particle):
    """Run name prefix used by submit_workflow.py for data generation."""
    return f"learning_curve_{PARTICLE_SAFE[particle]}"


def csv_prefix_for(particle):
    """Path prefix for CSVs: append '{i}.csv' to get each file."""
    rn = run_name_for(particle)
    return os.path.join(TIMING_PATH, "data", "df", f"{rn}_{NUM_EVENTS_PER_SIM}events_run_0_")


def verify_debug_mode():
    """Ensure debug_mode = False in submit_workflow.py (need 50 sims, not 5)."""
    with open(SUBMIT_SCRIPT, 'r') as f:
        content = f.read()
    match = re.search(r'debug_mode\s*=\s*(True|False)', content)
    if not match:
        print("  WARNING: Could not find debug_mode in submit_workflow.py")
        return
    if match.group(1) == 'True':
        print("  ERROR: debug_mode is True. Set to False before running learning curve.")
        print("  (Need 50 sims per particle, not 5)")
        sys.exit(1)
    print("  debug_mode = False (OK)")


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


# ── Phase 1: Data Generation ──────────────────────────────────────────────

def generate_data():
    """Launch submit_workflow.py for each particle in parallel. Blocks until all finish."""
    print("\n[Phase 1] Generating data (50 sims × 500 events × 3 particles = 150 jobs)")

    log_dir = os.path.join(WORK_EIC, "slurm", "learning_curve_logs")
    ensure_dir(log_dir)

    processes = []
    for particle in PARTICLES:
        psafe = PARTICLE_SAFE[particle]
        rn_pref = run_name_for(particle)

        cmd = [
            sys.executable, SUBMIT_SCRIPT,
            "--run_name_pref", rn_pref,
            "--particle", particle,
            "--skipTraining",
            "--no-classification",
            "--runNum", "0",
        ]

        log_path = os.path.join(log_dir, f"data_gen_{psafe}.log")
        log_file = open(log_path, 'w')
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        processes.append((particle, proc, log_file))
        print(f"  Launched data generation for {particle} (PID {proc.pid})")

    # Wait for all to finish
    print(f"\n  Waiting for {len(processes)} data generation processes...")
    while True:
        running = [(p, proc, lf) for p, proc, lf in processes if proc.poll() is None]
        done = len(processes) - len(running)
        if not running:
            break
        print(f"  {done}/{len(processes)} done, {len(running)} running... (sleeping 30s)")
        time.sleep(30)

    # Check results
    all_ok = True
    for particle, proc, log_file in processes:
        log_file.close()
        if proc.returncode != 0:
            print(f"  ERROR: Data generation for {particle} failed (rc={proc.returncode})")
            all_ok = False

    if not all_ok:
        print("  Some data generation processes failed. Check logs in learning_curve_logs/")
        sys.exit(1)

    # Verify CSVs exist
    for particle in PARTICLES:
        prefix = csv_prefix_for(particle)
        n_found = sum(1 for i in range(50) if os.path.isfile(f"{prefix}{i}.csv"))
        print(f"  {particle}: {n_found}/50 CSVs found")
        if n_found < 40:
            print(f"  WARNING: Only {n_found} CSVs for {particle}. Some sims may have failed.")

    print("  Phase 1 complete.")


# ── Phase 2: Training Sweeps ──────────────────────────────────────────────

def submit_training_jobs():
    """Submit GNN training jobs for each (particle, numDfs) combination.
    Returns dict mapping (particle, numDfs) -> job_id."""

    print(f"\n[Phase 2] Submitting {len(PARTICLES) * len(NUMDF_VALUES)} training jobs")

    ensure_dir(RESULTS_DIR)
    ensure_dir(PLOTS_DIR)
    ensure_dir(SHELLS_DIR)

    out_folder = os.path.join(WORK_EIC, "slurm", "output", f"output{current_date}")
    error_folder = os.path.join(WORK_EIC, "slurm", "error", f"error{current_date}")
    ensure_dir(out_folder)
    ensure_dir(error_folder)

    job_ids = {}
    shell_scripts = []

    for particle in PARTICLES:
        psafe = PARTICLE_SAFE[particle]
        prefix = csv_prefix_for(particle)

        for n in NUMDF_VALUES:
            job_name = f"lc_{psafe}_n{n}"
            results_file = os.path.join(RESULTS_DIR, f"{psafe}_n{n}.txt")
            model_dir = os.path.join(MODELS_DIR, f"{psafe}_n{n}")
            ensure_dir(model_dir)

            shell_path = os.path.join(SHELLS_DIR, f"lc_train_{psafe}_n{n}.sh")
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
set -e

echo "Learning curve training: {particle}, numDfs={n}"
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
  --particle "{particle}" \
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
            job_ids[(particle, n)] = job_id
            print(f"  Submitted {job_name} -> job {job_id}")

    return job_ids, shell_scripts


def wait_for_training(job_ids):
    """Poll until all training jobs finish."""
    total = len(job_ids)
    print(f"\n  Waiting for {total} training jobs to complete...")

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

        n_done = len(completed)
        if n_done < total:
            print(f"  {n_done}/{total} done... (sleeping 30s)")
            time.sleep(30)

    if failed:
        print(f"\n  WARNING: {len(failed)} training jobs failed:")
        for particle, n in failed:
            print(f"    {PARTICLE_SAFE[particle]} n={n}")
    else:
        print(f"\n  All {total} training jobs completed successfully.")

    return failed


# ── Phase 2b: Classifier Training Sweeps ──────────────────────────────────

def submit_classifier_jobs():
    """Submit GNN classifier (mu-/pi+ ID) training jobs for each numDfs value.
    Uses mu- and pi+ CSVs with matching numDfs.
    Returns dict mapping numDfs -> job_id."""

    print(f"\n[Phase 2b] Submitting {len(NUMDF_VALUES)} classifier training jobs")

    ensure_dir(RESULTS_DIR)
    ensure_dir(PLOTS_DIR)
    ensure_dir(SHELLS_DIR)

    out_folder = os.path.join(WORK_EIC, "slurm", "output", f"output{current_date}")
    error_folder = os.path.join(WORK_EIC, "slurm", "error", f"error{current_date}")
    ensure_dir(out_folder)
    ensure_dir(error_folder)

    mu_prefix = csv_prefix_for("mu-")
    pi_prefix = csv_prefix_for("pi+")

    job_ids = {}
    shell_scripts = []

    for n in NUMDF_VALUES:
        job_name = f"lc_classifier_n{n}"
        results_file = os.path.join(RESULTS_DIR, f"classifier_n{n}.txt")
        model_dir = os.path.join(MODELS_DIR, f"classifier_n{n}")
        ensure_dir(model_dir)

        shell_path = os.path.join(SHELLS_DIR, f"lc_classifier_n{n}.sh")
        shell_scripts.append(shell_path)


        mem_alloc = 40
        if(n > 50):
            mem_alloc = 60
        if(n > 70):
            mem_alloc = 100
        if(n > 85):
            mem_alloc = 120
        
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
#SBATCH --mem={mem_alloc}G
#SBATCH --gpus=1
#SBATCH --mail-user={MAIL_USER}
#SBATCH --mail-type=FAIL
#SBATCH --exclude={EXCLUDE_NODES}
set -e

echo "Learning curve classifier: numDfs={n}"
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
        job_ids[n] = job_id
        print(f"  Submitted {job_name} -> job {job_id}")

    return job_ids, shell_scripts


def wait_for_classifier(job_ids):
    """Poll until all classifier training jobs finish."""
    total = len(job_ids)
    print(f"\n  Waiting for {total} classifier jobs to complete...")

    completed = set()
    failed = []

    while len(completed) < total:
        for n, job_id in job_ids.items():
            if n in completed:
                continue
            status = get_job_status(job_id)
            if status == 1:
                completed.add(n)
            elif status == -1:
                completed.add(n)
                failed.append(n)

        if len(completed) < total:
            print(f"  {len(completed)}/{total} done... (sleeping 30s)")
            time.sleep(30)

    if failed:
        print(f"\n  WARNING: {len(failed)} classifier jobs failed: n={failed}")
    else:
        print(f"\n  All {total} classifier jobs completed successfully.")

    return failed


# ── Phase 3: Collect Results & Plot ───────────────────────────────────────

def collect_results():
    """Read energy prediction results. Returns dict of (particle, numDfs) -> (low_rmse, high_rmse)."""
    results = {}
    for particle in PARTICLES:
        psafe = PARTICLE_SAFE[particle]
        for n in NUMDF_VALUES:
            results_file = os.path.join(RESULTS_DIR, f"{psafe}_n{n}.txt")
            if not os.path.isfile(results_file):
                print(f"  Missing: {psafe}_n{n}.txt")
                continue
            with open(results_file, 'r') as f:
                lines = f.read().strip().split('\n')
            if len(lines) >= 2:
                try:
                    low_rmse = float(lines[0])
                    high_rmse = float(lines[1])
                    results[(particle, n)] = (low_rmse, high_rmse)
                except ValueError:
                    print(f"  Bad data in {psafe}_n{n}.txt: {lines}")
            elif len(lines) == 1:
                try:
                    val = float(lines[0])
                    if val == -1:
                        print(f"  {psafe}_n{n}.txt: reported failure (-1)")
                    else:
                        results[(particle, n)] = (val, val)
                except ValueError:
                    pass
    return results


def collect_classifier_results():
    """Read classifier results. Returns dict of numDfs -> (low_auc, high_auc).
    Classifier writes results in append mode: \\n{low_auc}\\n{high_auc}"""
    results = {}
    for n in NUMDF_VALUES:
        results_file = os.path.join(RESULTS_DIR, f"classifier_n{n}.txt")
        if not os.path.isfile(results_file):
            print(f"  Missing: classifier_n{n}.txt")
            continue
        with open(results_file, 'r') as f:
            lines = [l.strip() for l in f.read().strip().split('\n') if l.strip()]
        if len(lines) >= 2:
            try:
                # Take the last two non-empty lines (classifier appends)
                low_auc = float(lines[-2])
                high_auc = float(lines[-1])
                results[n] = (low_auc, high_auc)
            except ValueError:
                print(f"  Bad data in classifier_n{n}.txt: {lines}")
    return results


def plot_learning_curve(results, classifier_results):
    """Plot RMSE and AUC vs number of events."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plot")
        return

    ensure_dir(PLOTS_DIR)

    has_classifier = bool(classifier_results)
    n_rows = 2 if has_classifier else 1
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 6 * n_rows))
    if n_rows == 1:
        axes = [axes]  # make indexable by row

    colors = {"pi+": "tab:blue", "mu-": "tab:red", "neutron": "tab:green"}
    markers = {"pi+": "o", "mu-": "s", "neutron": "^"}

    # Row 1: Energy RMSE
    ax_low, ax_high, ax_avg = axes[0]

    for particle in PARTICLES:
        x_vals, low_vals, high_vals = [], [], []
        for n in NUMDF_VALUES:
            key = (particle, n)
            if key in results:
                low, high = results[key]
                x_vals.append(n * NUM_EVENTS_PER_SIM)
                low_vals.append(low)
                high_vals.append(high)
        if not x_vals:
            continue
        avg_vals = [(l + h) / 2 for l, h in zip(low_vals, high_vals)]
        ax_low.plot(x_vals, low_vals, marker=markers[particle], color=colors[particle],
                    label=particle, linewidth=2, markersize=8)
        ax_high.plot(x_vals, high_vals, marker=markers[particle], color=colors[particle],
                     label=particle, linewidth=2, markersize=8)
        ax_avg.plot(x_vals, avg_vals, marker=markers[particle], color=colors[particle],
                    label=particle, linewidth=2, markersize=8)

    for ax, title in [(ax_low, "Low E RMSE (<2.75 GeV)"),
                      (ax_high, "High E RMSE (>=2.75 GeV)"),
                      (ax_avg, "Average RMSE")]:
        ax.set_xlabel("Number of Events", fontsize=14)
        ax.set_ylabel("RMSE (GeV)", fontsize=14)
        ax.set_title(title, fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

    # Row 2: Classifier AUC (if available)
    if has_classifier:
        ax_cls_low, ax_cls_high, ax_cls_avg = axes[1]

        x_vals, low_vals, high_vals = [], [], []
        for n in NUMDF_VALUES:
            if n in classifier_results:
                low, high = classifier_results[n]
                x_vals.append(n * NUM_EVENTS_PER_SIM)
                low_vals.append(low)
                high_vals.append(high)

        if x_vals:
            avg_vals = [(l + h) / 2 for l, h in zip(low_vals, high_vals)]
            ax_cls_low.plot(x_vals, low_vals, marker='D', color='tab:purple',
                           label='mu/pi classifier', linewidth=2, markersize=8)
            ax_cls_high.plot(x_vals, high_vals, marker='D', color='tab:purple',
                            label='mu/pi classifier', linewidth=2, markersize=8)
            ax_cls_avg.plot(x_vals, avg_vals, marker='D', color='tab:purple',
                           label='mu/pi classifier', linewidth=2, markersize=8)

        for ax, title in [(ax_cls_low, "Low E muID AUC (<2.75 GeV)"),
                          (ax_cls_high, "High E muID AUC (>=2.75 GeV)"),
                          (ax_cls_avg, "Average muID AUC")]:
            ax.set_xlabel("Number of Events per Particle", fontsize=14)
            ax.set_ylabel("ROC AUC", fontsize=14)
            ax.set_title(title, fontsize=16)
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=max(0.5, min(low_vals + high_vals) - 0.05) if x_vals else 0.5)

    fig.suptitle("GNN Learning Curve: Performance vs Training Data Size", fontsize=18, y=1.02)
    fig.tight_layout()

    plot_path = os.path.join(PLOTS_DIR, "learning_curve_summary.pdf")
    fig.savefig(plot_path, bbox_inches='tight')
    print(f"  Saved plot to {plot_path}")

    png_path = os.path.join(PLOTS_DIR, "learning_curve_summary.png")
    fig.savefig(png_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def print_summary_table(results, classifier_results=None):
    """Print a text summary table of results."""
    print("\n" + "=" * 90)
    print("LEARNING CURVE RESULTS — Energy Prediction (RMSE)")
    print("=" * 90)
    print(f"{'Particle':<10} | {'numDfs':>6} | {'Events':>7} | {'Low RMSE':>10} | {'High RMSE':>10} | {'Avg RMSE':>10}")
    print("-" * 90)

    for particle in PARTICLES:
        psafe = PARTICLE_SAFE[particle]
        for n in NUMDF_VALUES:
            key = (particle, n)
            if key in results:
                low, high = results[key]
                avg = (low + high) / 2
                print(f"{psafe:<10} | {n:>6} | {n * NUM_EVENTS_PER_SIM:>7} | {low:>10.4f} | {high:>10.4f} | {avg:>10.4f}")
            else:
                print(f"{psafe:<10} | {n:>6} | {n * NUM_EVENTS_PER_SIM:>7} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10}")
        print("-" * 90)

    # Classifier AUC table
    if classifier_results:
        print("\n" + "=" * 90)
        print("LEARNING CURVE RESULTS — muID Classifier (ROC AUC)")
        print("=" * 90)
        print(f"{'numDfs':>6} | {'Events/particle':>15} | {'Low AUC':>10} | {'High AUC':>10} | {'Avg AUC':>10}")
        print("-" * 70)
        for n in NUMDF_VALUES:
            if n in classifier_results:
                low, high = classifier_results[n]
                avg = (low + high) / 2
                print(f"{n:>6} | {n * NUM_EVENTS_PER_SIM:>15} | {low:>10.4f} | {high:>10.4f} | {avg:>10.4f}")
            else:
                print(f"{n:>6} | {n * NUM_EVENTS_PER_SIM:>15} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10}")
        print("-" * 70)

    # Find the "knee" — point where improvement drops below 5% relative to previous
    print("\nDiminishing returns analysis (Energy RMSE):")
    for particle in PARTICLES:
        psafe = PARTICLE_SAFE[particle]
        prev_avg = None
        knee_n = None
        for n in NUMDF_VALUES:
            key = (particle, n)
            if key not in results:
                continue
            low, high = results[key]
            avg = (low + high) / 2
            if prev_avg is not None:
                improvement = (prev_avg - avg) / prev_avg * 100
                if improvement < 2.0 and knee_n is None:
                    knee_n = n
            prev_avg = avg
        if knee_n:
            print(f"  {psafe}: diminishing returns around {knee_n} sims ({knee_n * NUM_EVENTS_PER_SIM} events)")
        else:
            print(f"  {psafe}: performance still improving at {NUMDF_VALUES[-1]} sims — may need more data")

    # Classifier diminishing returns (AUC should increase, so look for <1% improvement)
    if classifier_results:
        print("\nDiminishing returns analysis (muID AUC):")
        prev_avg = None
        knee_n = None
        for n in NUMDF_VALUES:
            if n not in classifier_results:
                continue
            low, high = classifier_results[n]
            avg = (low + high) / 2
            if prev_avg is not None:
                improvement = (avg - prev_avg) / prev_avg * 100  # AUC increases
                if improvement < 1.0 and knee_n is None:
                    knee_n = n
            prev_avg = avg
        if knee_n:
            print(f"  classifier: diminishing returns around {knee_n} sims ({knee_n * NUM_EVENTS_PER_SIM} events/particle)")
        else:
            print(f"  classifier: performance still improving at {NUMDF_VALUES[-1]} sims — may need more data")

    print("=" * 90)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 90)
    print("GNN LEARNING CURVE STUDY")
    print("=" * 90)
    print(f"Particles: {', '.join(PARTICLES)}")
    print(f"numDfs sweep: {NUMDF_VALUES}")
    print(f"Events per sim: {NUM_EVENTS_PER_SIM}")
    print()

    # Verify settings
    print("Checking submit_workflow.py settings...")
    verify_debug_mode()

    # Phase 1: Data generation
    if SKIP_DATA:
        print("\n[Phase 1] Skipping data generation (--skip-data)")
        # Verify CSVs exist
        for particle in PARTICLES:
            prefix = csv_prefix_for(particle)
            n_found = sum(1 for i in range(50) if os.path.isfile(f"{prefix}{i}.csv"))
            print(f"  {particle}: {n_found}/50 CSVs found")
    else:
        generate_data()

    # Phase 2a: Energy prediction training sweeps
    if SKIP_TRAINING:
        print("\n[Phase 2a] Skipping energy training (--skip-training)")
        print("[Phase 2b] Skipping classifier training (--skip-training)")
    else:
        if(not ONLY_CLASS):
            job_ids, shell_scripts = submit_training_jobs()
        cls_job_ids, cls_shell_scripts = submit_classifier_jobs()

        if(not ONLY_CLASS):
            failed = wait_for_training(job_ids)
        cls_failed = wait_for_classifier(cls_job_ids)

        # Cleanup shell scripts
        if(not ONLY_CLASS):
            scripts = shell_scripts + cls_shell_scripts
        else:
            scripts = cls_shell_scripts
    
        for script in scripts:
            p = Path(script)
            if p.is_file():
                p.unlink()

        if(not ONLY_CLASS):
            if failed:
                print(f"\n  {len(failed)} energy training jobs failed. Results will be partial.")
        if cls_failed:
            print(f"\n  {len(cls_failed)} classifier jobs failed. Results will be partial.")


    # Phase 3: Collect and plot
    print("\n[Phase 3] Collecting results and plotting...")
    if(not ONLY_CLASS):
        results = collect_results()
    classifier_results = collect_classifier_results()

    if not classifier_results:
        if(ONLY_CLASS or not results):
            print("  No results found! Check training job outputs.")
        sys.exit(1)

    if(not ONLY_CLASS):
        print_summary_table(results, classifier_results)
        plot_learning_curve(results, classifier_results)

    else:
        print("Only ran training jobs, run again with --skip-training to print results")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
