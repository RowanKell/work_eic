#!/usr/bin/env python3
"""
Benchmark NF sampling batch sizes to find:
1. Maximum batch size before GPU OOM
2. Optimal batch size for fastest throughput (samples/sec)

Can run standalone with synthetic data, or with real processed data from the
ddsim+process pipeline. Use --slurm to submit a full pipeline job that runs
ddsim, process_root_file, then this benchmark with real data.

Usage:
    # Submit full pipeline benchmark (ddsim -> process -> benchmark):
    source /hpc/group/vossenlab/rck32/eic/work_eic/setup.sh
    python3 benchmark_batch_size.py --slurm

    # Run with existing processed data (on GPU node):
    python3 benchmark_batch_size.py --inputProcessedData path/to/data.json

    # Run with synthetic data only (on GPU node):
    python3 benchmark_batch_size.py --synthetic
"""

import argparse
import time
import sys
import os
import subprocess

import torch
import torch.cuda


def get_gpu_info():
    """Get GPU name and total VRAM."""
    if not torch.cuda.is_available():
        return "CPU", 0
    name = torch.cuda.get_device_name(0)
    vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    return name, vram_mb


def load_model(thickness="2cm"):
    """Load NF model using get_compiled_NF_model."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from time_res_util import get_compiled_NF_model
    model = get_compiled_NF_model(thickness=thickness, useGPU=True)
    return model


def load_real_context(input_path):
    """Load processed data and extract context tensors the same way
    newer_prepare_nn_input does, but without running the NF sampling."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from momentum_prediction_util import load_defaultdict
    from debug_diagnostics import validate_context_tensor, clamp_context_tensor

    processed_data = load_defaultdict(input_path)

    all_context = []
    num_pixel_list = ["num_pixels_high_z", "num_pixels_low_z"]

    for event_idx, event_data in processed_data.items():
        for stave_idx, stave_data in event_data.items():
            for layer_idx, layer_data in stave_data.items():
                for segment_idx, segment_data in layer_data.items():
                    for particle_id, particle_data in segment_data.items():
                        base_context = torch.tensor(
                            [particle_data['z_pos'], particle_data['hittheta'], particle_data['hitmomentum']],
                            dtype=torch.float32)
                        for SiPM_idx in range(2):
                            num_pixel_tag = num_pixel_list[SiPM_idx]
                            all_context.append(base_context.repeat(particle_data[num_pixel_tag], 1))

    all_context = torch.cat(all_context)

    # Validate and clamp the same way production does
    all_context, valid_mask = clamp_context_tensor(all_context, verbose=True)
    if not valid_mask.all():
        num_removed = int((~valid_mask).sum())
        print(f"Removed {num_removed} invalid context entries")
        all_context = all_context[valid_mask]

    return all_context


def benchmark_one(model, context_tensor, batch_size, num_trials=3):
    """Run NF sampling at a given batch size and return timing + VRAM stats.

    context_tensor: full context tensor on CPU (will be sliced to batch_size).
    """
    device = torch.device('cuda')

    # Use real context if available and large enough, otherwise tile it
    if len(context_tensor) >= batch_size:
        context = context_tensor[:batch_size].to(device)
    else:
        # Tile to reach batch_size
        repeats = (batch_size // len(context_tensor)) + 1
        context = context_tensor.repeat(repeats, 1)[:batch_size].to(device)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    times = []
    for t in range(num_trials):
        try:
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                _ = model.sample(num_samples=batch_size, context=context)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                torch.cuda.empty_cache()
                return {
                    "batch_size": batch_size,
                    "error": str(e)[:80],
                    "success": False,
                }
            raise

    peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
    mean_time = sum(times) / len(times)
    throughput = batch_size / mean_time

    del context
    torch.cuda.empty_cache()

    return {
        "batch_size": batch_size,
        "mean_time_s": mean_time,
        "throughput": throughput,
        "peak_vram_mb": peak_vram,
        "success": True,
    }


def run_benchmark(thickness="2cm", num_trials=3, input_data_path=None):
    """Run the full benchmark sweep."""
    gpu_name, total_vram = get_gpu_info()
    print(f"GPU: {gpu_name}")
    print(f"Total VRAM: {total_vram:.0f} MB")
    print(f"NF model thickness: {thickness}")
    print(f"Trials per batch size: {num_trials}")
    node = os.environ.get('SLURMD_NODENAME', 'local')
    print(f"Node: {node}")

    # Load context data
    if input_data_path:
        print(f"Loading real data from: {input_data_path}")
        context_tensor = load_real_context(input_data_path)
        print(f"Total context samples: {len(context_tensor):,}")
        data_source = "real"
    else:
        print("Using synthetic context data (z=0, theta=90, p=2.5)")
        context_tensor = torch.tensor([[0.0, 90.0, 2.5]], dtype=torch.float32)
        data_source = "synthetic"
    print()

    model = load_model(thickness)

    batch_sizes = [
        64, 128, 256, 512, 1000, 2000, 4000, 8000,
        16000, 32000, 64000, 100000, 128000, 200000, 256000, 512000,
    ]

    results = []

    print(f"{'Batch Size':>12} {'Time (s)':>10} {'Samples/s':>12} {'Peak VRAM':>12} {'Status':>10}")
    print("-" * 62)

    for bs in batch_sizes:
        result = benchmark_one(model, context_tensor, bs, num_trials=num_trials)
        results.append(result)

        if result["success"]:
            print(f"{bs:>12,} {result['mean_time_s']:>10.4f} {result['throughput']:>12,.0f} {result['peak_vram_mb']:>10,.0f} MB {'OK':>10}")
        else:
            print(f"{bs:>12,} {'':>10} {'':>12} {'':>12} {'OOM':>10}")
            print(f"  Error: {result['error']}")
            break

    # Summary
    print()
    print("=" * 62)
    print(f"SUMMARY (data source: {data_source})")
    print("=" * 62)

    successful = [r for r in results if r["success"]]
    if not successful:
        print("No batch sizes succeeded!")
        return results

    max_bs = successful[-1]
    print(f"Max successful batch size: {max_bs['batch_size']:,}")
    print(f"  Time: {max_bs['mean_time_s']:.4f}s, Throughput: {max_bs['throughput']:,.0f} samples/s")
    print(f"  Peak VRAM: {max_bs['peak_vram_mb']:,.0f} MB / {total_vram:,.0f} MB ({max_bs['peak_vram_mb']/total_vram*100:.1f}%)")

    best = max(successful, key=lambda r: r["throughput"])
    print(f"\nOptimal batch size (max throughput): {best['batch_size']:,}")
    print(f"  Time: {best['mean_time_s']:.4f}s, Throughput: {best['throughput']:,.0f} samples/s")
    print(f"  Peak VRAM: {best['peak_vram_mb']:,.0f} MB / {total_vram:,.0f} MB ({best['peak_vram_mb']/total_vram*100:.1f}%)")

    prod = next((r for r in successful if r["batch_size"] == 1000), None)
    if prod and best["batch_size"] != 1000:
        speedup = best["throughput"] / prod["throughput"]
        print(f"\nProduction batch size (1000): {prod['throughput']:,.0f} samples/s")
        print(f"Speedup from optimal: {speedup:.2f}x")

    if input_data_path:
        total_samples = len(load_real_context(input_data_path)) if data_source == "real" else 0
        if total_samples > 0 and best["success"]:
            est_time = total_samples / best["throughput"]
            prod_time = total_samples / prod["throughput"] if prod else 0
            print(f"\nEstimated time for full event ({total_samples:,} samples):")
            print(f"  At optimal batch size: {est_time:.2f}s")
            if prod_time > 0:
                print(f"  At production batch size (1000): {prod_time:.2f}s")

    return results


def submit_slurm(particle="pi+"):
    """Submit a full pipeline SLURM job: ddsim -> process -> benchmark."""
    try:
        workdir = os.environ['WORK_EIC']
        eic_shell_home = os.environ['EIC_SHELL_HOME']
        ml_venv = os.environ['ML_VENV_HOME']
        epic_home = os.environ['EPIC_HOME']
        mail_user = os.environ['MAIL_USER']
    except KeyError as e:
        print(f"Missing env var {e}. Source work_eic/setup.sh first.")
        sys.exit(1)

    from datetime import datetime
    current_date = datetime.now().strftime("%B_%d")
    out_dir = f"{workdir}/slurm/output/output{current_date}"
    err_dir = f"{workdir}/slurm/error/error{current_date}"
    root_dir = f"{workdir}/root_files/Clustering/{current_date}"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(err_dir, exist_ok=True)
    os.makedirs(root_dir, exist_ok=True)

    script_path = os.path.abspath(__file__)
    compact_file = f"{epic_home}/install/share/epic/epic_klmws_only.xml"
    setup_path = f"{epic_home}/install/setup.sh"
    steering_file = f"{workdir}/steering/scint_sensitive/sector.py"
    run_name = f"bench_batch_{particle.replace('+','p').replace('-','m')}"
    num_events = 500
    json_path = f"{workdir}/macros/Timing_estimation/data/processed_data/{run_name}.json"
    root_path = f"{root_dir}/{run_name}_{num_events}.edm4hep.root"

    slurm_script = f"{workdir}/slurm/shells/benchmark_batch_pipeline.sh"
    with open(slurm_script, 'w') as f:
        f.write(f"""#!/bin/bash
#SBATCH --chdir={epic_home}
#SBATCH --job-name=bench_batch_pipeline
#SBATCH --output={out_dir}/%x.out
#SBATCH --error={err_dir}/%x.err
#SBATCH -p scavenger-gpu
#SBATCH --time=01:00:00
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --mail-user={mail_user}
#SBATCH --mail-type=FAIL
#SBATCH --exclude=dcc-youlab-gpu-28,dcc-gehmlab-gpu-ferc-s-z25-18
set -eo pipefail

echo "=== GPU DIAGNOSTICS ==="
nvidia-smi
echo "SLURMD_NODENAME: $SLURMD_NODENAME"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "=== END GPU DIAGNOSTICS ==="

#########   DDSIM + PROCESS (inside eic-shell)   ##########
cat << EOF | {eic_shell_home}/eic-shell
set -e
source {workdir}/setup.sh
source {setup_path}

echo "=== Running ddsim ==="
/usr/local/bin/ddsim --compactFile {compact_file} -G --numberOfEvents {num_events} --steeringFile {steering_file} --outputFile {root_path} --part.userParticleHandler="" --gun.particle {particle}
echo "DDSIM completed"

echo "=== Running process_root_file ==="
python3 {workdir}/macros/Timing_estimation/process_root_file.py --filePathName {root_path} --processedDataPath {json_path} --geometryType 1 --compactFile {compact_file} --deleteROOTFile
echo "Process completed"
EOF

echo "=== eic-shell done, starting benchmark ==="

#########   BENCHMARK (native, with GPU)   ##########
source {ml_venv}/bin/activate
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

python3 {script_path} --inputProcessedData {json_path} --thickness 2cm --trials 3

# Cleanup
rm -f {json_path}
echo "=== Benchmark complete ==="
""")

    result = subprocess.run(["sbatch", slurm_script], capture_output=True, text=True)
    print(f"Submitted: {result.stdout.strip()}")
    print(f"Output: {out_dir}/bench_batch_pipeline.out")
    print(f"Error:  {err_dir}/bench_batch_pipeline.err")
    print(f"\nPipeline: ddsim ({num_events} {particle} events) -> process -> benchmark")


def main():
    parser = argparse.ArgumentParser(description="Benchmark NF sampling batch sizes")
    parser.add_argument("--thickness", type=str, default="2cm",
                        help="Scintillator thickness for NF model (1cm or 2cm)")
    parser.add_argument("--trials", type=int, default=3,
                        help="Number of trials per batch size for timing")
    parser.add_argument("--slurm", action="store_true",
                        help="Submit full pipeline SLURM job (ddsim -> process -> benchmark)")
    parser.add_argument("--inputProcessedData", type=str, default=None,
                        help="Path to processed JSON data (from process_root_file.py)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic context data instead of real data")
    parser.add_argument("--particle", type=str, default="pi+",
                        help="Particle type for SLURM pipeline (default: pi+)")
    args = parser.parse_args()

    if args.slurm:
        submit_slurm(particle=args.particle)
    else:
        run_benchmark(
            thickness=args.thickness,
            num_trials=args.trials,
            input_data_path=args.inputProcessedData,
        )


if __name__ == "__main__":
    main()
