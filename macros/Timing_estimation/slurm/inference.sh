#!/bin/bash
#SBATCH --job-name=inference_20k_run_7
#SBATCH --output=/cwork/rck32/eic/work_eic/macros/Timing_estimation/slurm/output/outputAugust_7/%x_mu.out
#SBATCH --error=/cwork/rck32/eic/work_eic/macros/Timing_estimation/slurm/error/errorAugust_7/%x_mu.err
#SBATCH -p gpu-common
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem=40G
#SBATCH --mail-user=rck32@duke.edu
#SBATCH --mail-type=END
echo began job
source /cwork/rck32/ML_venv/bin/activate
python3 /cwork/rck32/eic/work_eic/macros/Timing_estimation/inference.py