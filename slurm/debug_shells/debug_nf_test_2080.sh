#!/bin/bash
#SBATCH --job-name=debug_nf_2080
#SBATCH --output=/hpc/group/vossenlab/rck32/eic/work_eic/slurm/output/debug_nf_2080_%j.out
#SBATCH --error=/hpc/group/vossenlab/rck32/eic/work_eic/slurm/error/debug_nf_2080_%j.err
#SBATCH -p scavenger-gpu
#SBATCH --time=00:10:00
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2080:1
#SBATCH --mem=8G
set -e

echo "=== GPU DIAGNOSTICS ==="
nvidia-smi
echo "SLURMD_NODENAME: $SLURMD_NODENAME"
echo "=== END GPU DIAGNOSTICS ==="

source /hpc/group/vossenlab/rck32/ML_venv/bin/activate
export CUDA_LAUNCH_BLOCKING=1
python3 /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/debug_nf_test.py --thickness 2cm
deactivate
echo "DONE"
