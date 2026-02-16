# MOBO Preparation Studies — Plot Regeneration Guide

This document explains every script used to justify MOBO experiment design choices (training data volume, RAM allocation, batch sizes, result stability). Each section tells you **what the script does**, **what plots/outputs it produces**, and **exactly how to regenerate them**.

> **Prerequisites for all scripts:**
> ```bash
> source /hpc/group/vossenlab/rck32/eic/work_eic/setup.sh
> source /hpc/group/vossenlab/rck32/eic/epic_klm/install/setup.sh
> ```

---

## Table of Contents

1. [Learning Curve (Single Geometry)](#1-learning-curve-single-geometry)
2. [Multi-Geometry Learning Curve & Rank Stability](#2-multi-geometry-learning-curve--rank-stability)
3. [Memory Stress Test](#3-memory-stress-test)
4. [NF Batch Size Benchmark](#4-nf-batch-size-benchmark)
5. [GNN Hyperparameter Optimization](#5-gnn-hyperparameter-optimization)
6. [GPU Debugging & Diagnostics](#6-gpu-debugging--diagnostics)
7. [File Index](#7-file-index)

---

## 1. Learning Curve (Single Geometry)

**Script**: `slurm/learning_curve.py`
**Question answered**: *How many training dataframes (events) does the GNN need before performance plateaus?*

### What it does

Trains the GNN energy predictor and muID classifier on the baseline geometry with increasing amounts of training data (numDfs = 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90). Each numDf value means that many CSV files are loaded, where each CSV comes from a 500-event DDSim simulation. This reveals the diminishing-returns point where adding more data stops helping.

Three phases:
- **Phase 1**: Generate 50 simulations x 500 events for pi+, mu-, and neutron (150 SLURM jobs)
- **Phase 2a**: Train GNN energy predictor at each numDf value (one SLURM job per particle per numDf)
- **Phase 2b**: Train muID classifier (mu-/pi+ separation) at each numDf value
- **Phase 3**: Collect binned RMSE and binned AUC results, generate learning curve plots

### Outputs

| Output | Path |
|--------|------|
| **Summary learning curve plot** | `macros/Timing_estimation/plots/learning_curve/learning_curve_summary.pdf` |
| Per-particle per-numDf energy plots | `macros/Timing_estimation/plots/learning_curve/lc_{particle}_n{numDf}.pdf` |
| Per-numDf classifier ROC plots | `macros/Timing_estimation/plots/learning_curve/lc_classifier_n{numDf}.jpeg` |
| Raw RMSE results (text) | `macros/Timing_estimation/results/learning_curve/{particle}_n{numDf}.txt` |
| Raw AUC results (text) | `macros/Timing_estimation/results/learning_curve/classifier_n{numDf}.txt` |

The **summary plot** (`learning_curve_summary.pdf`) is the key paper figure — it shows all 4 MOBO objectives (low/high energy RMSE, low/high energy AUC) vs. number of training dataframes.

### How to regenerate

```bash
# Full run (generates data + trains + plots) — takes many hours
python3 /hpc/group/vossenlab/rck32/eic/work_eic/slurm/learning_curve.py

# If simulation data already exists, skip to training + plotting
python3 slurm/learning_curve.py --skip-data

# If training results exist, just regenerate plots
python3 slurm/learning_curve.py --skip-data --skip-training

# Only run classifier training (skip energy predictor)
python3 slurm/learning_curve.py --skip-data --only-classification
```

### Key configuration (in script)

```python
PARTICLES = ["pi+", "mu-", "neutron"]
NUMDF_VALUES = [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90]
NUM_EVENTS_PER_SIM = 500
# GPU jobs: 60G memory, 1 GPU, 45 min time limit, scavenger-gpu partition
```

### What to look for in the plots

- The numDf value where RMSE and AUC flatten out is the minimum data needed per MOBO trial
- If the curve is still dropping at numDf=50, you need more data per trial

---

## 2. Multi-Geometry Learning Curve & Rank Stability

**Script**: `slurm/multi_geo_learning_curve.py`
**Question answered**: *Are geometry rankings preserved when using fewer training events? Can MOBO use less data per trial without changing which geometry "wins"?*

### What it does

Tests 3 deliberately diverse detector geometries and checks whether their relative rankings on all 4 MOBO objectives remain stable as numDfs varies. If rankings are stable at low numDfs, MOBO can evaluate each trial with less compute.

| Geometry | Layers | Steel Ratio | Steel (mm) | Scint (mm) | Character |
|----------|--------|-------------|------------|------------|-----------|
| A (Baseline) | 14 | 0.735 | 55.5 | 20.0 | Standard |
| B | 8 | 0.90 | 67.95 | 7.55 | Few thick-steel layers |
| C | 17 | 0.40 | 30.2 | 45.3 | Many thin-steel layers |

Phases:
- **Phase 0**: Generate geometry XMLs for B and C (modifies `klmws.xml` template)
- **Phase 1**: Simulate data for B and C (reuses A data from learning_curve.py)
- **Phase 2a**: Train neutron energy predictor per geometry per numDf
- **Phase 2b**: Train muID classifier per geometry per numDf
- **Phase 3**: Rank stability analysis — are geometry rankings consistent across numDf values?

### Outputs

| Output | Path |
|--------|------|
| **Objectives plot** | `macros/Timing_estimation/plots/multi_geo_lc/multi_geo_lc_objectives.pdf` |
| **Rank stability plot** | `macros/Timing_estimation/plots/multi_geo_lc/multi_geo_lc_ranks.pdf` |
| Per-geometry per-numDf energy plots | `macros/Timing_estimation/plots/multi_geo_lc/mgeo_{geo}_{particle}_n{numDf}.pdf` |
| Raw results (text) | `macros/Timing_estimation/results/multi_geo_lc/energy_{geo}_{particle}_n{numDf}.txt` |
| Rank stability table | Console output (printed during Phase 3) |

The **rank stability plot** (`multi_geo_lc_ranks.pdf`) is the key paper figure — it shows geometry rank (1st/2nd/3rd) per objective at each numDf value. Stable lines = safe to use fewer events.

### How to regenerate

```bash
# Full run (XML creation + data generation + training + plotting)
python3 /hpc/group/vossenlab/rck32/eic/work_eic/slurm/multi_geo_learning_curve.py

# Skip data generation (reuse existing CSVs)
python3 slurm/multi_geo_learning_curve.py --skip-data

# Just re-plot from existing results
python3 slurm/multi_geo_learning_curve.py --skip-data --skip-training

# Skip XML geometry creation too
python3 slurm/multi_geo_learning_curve.py --skip-xml --skip-data

# Quick test mode (1 geometry, 1 numDf, fewer sims)
python3 slurm/multi_geo_learning_curve.py --test
```

### Important note

The geometry XML files must be installed (present in `epic_klm/install/share/epic/compact/pid/`). If you get `XercesC FATAL: unable to open primary document entity` errors, rebuild epic_klm:

```bash
cd /hpc/group/vossenlab/rck32/eic/epic_klm
eic-shell
cmake --build build && cmake --install build
```

### Key configuration (in script)

```python
NUMDF_VALUES_MGEO = [10, 20, 30, 50]  # production
# Test mode: 1 geometry, 5 sims, numDfs=[10]
```

---

## 3. Memory Stress Test

**Script**: `slurm/test_memory_limits.py`
**Question answered**: *What SLURM memory allocation does each geometry need? Will any geometry in the MOBO search space cause OOM?*

### What it does

Creates 8 test geometries spanning the MOBO parameter space (scintillator thickness 20–50mm, 14 or 18 layers), submits 5 SLURM jobs per geometry/particle combination (120 total), monitors completion, then uses `sacct` to extract peak memory usage (MaxRSS) and compute headroom vs. the allocated limit.

| Geometry | Layers | Steel (mm) | Scint (mm) | Memory Limit |
|----------|--------|------------|------------|--------------|
| normal_14L/18L | 14/18 | 55.5 | 20.0 | 8G |
| boundary_14L/18L | 14/18 | 40.0 | 30.0 | 8G (at threshold) |
| thick_14L/18L | 14/18 | 18.0 | 40.0 | 16G (pi+/n) / 10G (mu-) |
| worst_14L/18L | 14/18 | 10.0 | 50.0 | 16G (pi+/n) / 10G (mu-) |

### Outputs

| Output | Format |
|--------|--------|
| **Memory headroom table** | Console output (printed after all jobs complete) |
| Job submission logs | `slurm/mem_test_logs/` |
| Per-job sacct MaxRSS | Queried live via `sacct` |

The console table shows: geometry name, particle, mean MaxRSS, max MaxRSS, allocated limit, headroom %, and warnings for any config with <15% headroom or OOM failures.

**This is not a plot** — it produces a text table. To include in a paper, capture the console output or adapt the script to save a CSV/table.

### How to regenerate

```bash
# Full run (120 jobs, takes ~1 hour for all to complete)
python3 /hpc/group/vossenlab/rck32/eic/work_eic/slurm/test_memory_limits.py

# Quick mode (only worst-case geometry = worst_18L, 15 jobs)
python3 slurm/test_memory_limits.py --quick
```

### How dynamic memory allocation works in production

The memory limits tested here match the logic in `submit_workflow.py`:
- Scintillator thickness <= 30mm: all particles get **8G**
- Scintillator thickness > 30mm: pi+/neutron get **16G**, mu- gets **10G**

This threshold was validated by this stress test.

---

## 4. NF Batch Size Benchmark

**Script**: `macros/Timing_estimation/benchmark_batch_size.py`
**Question answered**: *What NF sampling batch size maximizes throughput without GPU OOM? How much faster can we go vs. the production batch size of 1000?*

### What it does

Tests NF (Normalizing Flow) sampling batch sizes from 64 to 512,000 on GPU. For each batch size, measures wall-clock time, throughput (samples/sec), and peak VRAM usage. Identifies the maximum batch size before OOM and the optimal throughput batch size.

Three modes:
- **`--slurm`**: Submits a full pipeline job (ddsim → process → benchmark with real data)
- **`--inputProcessedData`**: Uses existing processed JSON data
- **`--synthetic`**: Uses synthetic context tensors (quick GPU-only test)

### Outputs

| Output | Format |
|--------|--------|
| **Throughput table** | Console output (batch size, time, samples/sec, peak VRAM) |
| Summary statistics | Console: max batch size, optimal batch size, speedup vs production |

**This is also console output, not a saved plot.** To make a paper figure, you would need to capture the output and plot throughput vs. batch size externally, or modify the script to save a figure.

### Previous results (A5000, 24GB VRAM)

From the February 13 benchmark run:
- All batch sizes up to 512K succeeded (peak 11.3% VRAM)
- Optimal batch size: 200K → 87,063 samples/sec
- Production batch size (1K): 20,033 samples/sec
- **Speedup: 4.35x** by increasing batch size

### How to regenerate

```bash
# Synthetic-data-only benchmark (fastest, just needs a GPU node)
python3 macros/Timing_estimation/benchmark_batch_size.py --synthetic

# With real processed data
python3 macros/Timing_estimation/benchmark_batch_size.py \
    --inputProcessedData path/to/processed_data.json --thickness 2cm --trials 3

# Full pipeline via SLURM (ddsim → process → benchmark)
python3 macros/Timing_estimation/benchmark_batch_size.py --slurm --particle pi+
```

---

## 5. GNN Hyperparameter Optimization

**Script**: `macros/Timing_estimation/optimize.py`
**Question answered**: *What are the best GNN architecture hyperparameters?*

### What it does

Runs 100-trial Optuna hyperparameter optimization over the GNN architecture:

| Hyperparameter | Range |
|----------------|-------|
| MLP_hidden_dim | 16–100 |
| linear_capacity | 3–8 |
| n_linear_layers | 4–12 |
| n_conv_layers | 1–6 |
| learning_rate | 1e-4 to 5e-2 (log scale) |

Uses MedianPruner for early stopping of bad trials.

### Outputs

| Output | Path |
|--------|------|
| Pickled Optuna study | `macros/Timing_estimation/optimization/study_{N}/study.pkl` |
| Trial results text | `macros/Timing_estimation/optimization/study_{N}/study_{N}_optuna_results.txt` |
| Slice plot | `macros/Timing_estimation/optimization/study_{N}/optuna_slice_plot.html` |
| Parallel coordinate plot | `macros/Timing_estimation/optimization/study_{N}/optuna_parallel_coordinate.html` |
| Importance plot | `macros/Timing_estimation/optimization/study_{N}/optuna_importance.html` |
| Contour plot | `macros/Timing_estimation/optimization/study_{N}/optuna_contour.html` |
| Dataset size study plots | `macros/Timing_estimation/plots/dataset_size_study/study_{N}_*.pdf` |

### How to regenerate

```bash
# Activate ML venv first
source /hpc/group/vossenlab/rck32/ML_venv/bin/activate

# Run on a GPU node (this takes hours for 100 trials)
python3 macros/Timing_estimation/optimize.py
```

To reload and re-plot an existing study:
```python
import optuna, pickle
study = pickle.load(open("macros/Timing_estimation/optimization/study_3/study.pkl", "rb"))
optuna.visualization.plot_optimization_history(study).show()
```

---

## 6. GPU Debugging & Diagnostics

**Scripts**: `macros/Timing_estimation/debug_diagnostics.py`, `macros/Timing_estimation/debug_nf_test.py`
**Question answered**: *Are NF sampling results reproducible across GPU types? Which GPUs cause CUDA errors?*

### debug_diagnostics.py (utility module)

Provides diagnostic functions used by other scripts:
- `get_gpu_info()` — GPU name, architecture, VRAM, CUDA version, SLURM node
- `validate_context_tensor()` — NaN/Inf checks, per-column statistics
- `clamp_context_tensor()` — clamp context to training ranges
- `log_error_state()` — save diagnostic JSON + context tensor for replay

### debug_nf_test.py (standalone test)

Tests NF model loading and sampling across GPU types:

```bash
# Run on a GPU node
python3 macros/Timing_estimation/debug_nf_test.py --thickness 2cm

# Replay a saved error context
python3 macros/Timing_estimation/debug_nf_test.py --thickness 2cm --replay_dir debug_logs/
```

**Outputs**: Console report showing pass/fail for each test (loading method, nominal sampling, batch sampling, extreme context values, GPU diagnostics).

### Key findings (from CHANGELOG)

- `torch.compile(mode="reduce-overhead")` caused CUDA graph failures on shared GPUs — **removed**
- RTX 2080 Ti had ~14% failure rate due to CUDA illegal memory access (non-deterministic)
- A5000 had 43% CUBLAS_STATUS_ALLOC_FAILED rate with original `torch.load()` — fixed via CPU-first loading
- 8GB RTX 2080 nodes excluded from job submission (OOM with batch_size=50000 NF sampling)

---

## 7. File Index

Quick reference for all scripts and their locations relative to `work_eic/`:

### Scripts that generate paper-ready plots

| Script | Key Output |
|--------|------------|
| `slurm/learning_curve.py` | `plots/learning_curve/learning_curve_summary.pdf` |
| `slurm/multi_geo_learning_curve.py` | `plots/multi_geo_lc/multi_geo_lc_ranks.pdf`, `multi_geo_lc_objectives.pdf` |

### Scripts that produce console tables (capture for paper)

| Script | What it reports |
|--------|-----------------|
| `slurm/test_memory_limits.py` | Memory headroom per geometry/particle |
| `macros/Timing_estimation/benchmark_batch_size.py` | Throughput vs. batch size table |

### Supporting scripts

| Script | Role |
|--------|------|
| `macros/Timing_estimation/optimize.py` | Optuna hyperparameter optimization |
| `macros/Timing_estimation/debug_diagnostics.py` | GPU diagnostic utilities |
| `macros/Timing_estimation/debug_nf_test.py` | NF reproducibility test |
| `macros/Timing_estimation/train_GNN.py` | GNN energy predictor training (called by learning curve scripts) |
| `macros/Timing_estimation/train_GNN_classifier.py` | GNN muID classifier training (called by learning curve scripts) |
| `slurm/submit_workflow.py` | SLURM job orchestration (called by all study scripts) |

---

## Quick Regeneration Cheat Sheet

```bash
# Setup (always run first)
source /hpc/group/vossenlab/rck32/eic/work_eic/setup.sh
source /hpc/group/vossenlab/rck32/eic/epic_klm/install/setup.sh

# Re-plot learning curves from existing results (no SLURM needed)
python3 slurm/learning_curve.py --skip-data --skip-training

# Re-plot multi-geometry rank stability from existing results
python3 slurm/multi_geo_learning_curve.py --skip-data --skip-training

# Re-run memory stress test (submits 120 SLURM jobs)
python3 slurm/test_memory_limits.py

# Re-run batch size benchmark (needs GPU node)
python3 macros/Timing_estimation/benchmark_batch_size.py --synthetic

# Full from-scratch regeneration of everything (hours of compute)
python3 slurm/learning_curve.py
python3 slurm/multi_geo_learning_curve.py
python3 slurm/test_memory_limits.py
```
