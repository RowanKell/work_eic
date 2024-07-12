#!/bin/bash
#SBATCH --job-name=run_3_inference
#SBATCH --output=/cwork/rck32/eic/work_eic/macros/Timing_estimation/plotting_slurm/output/%x.out
#SBATCH --error=/cwork/rck32/eic/work_eic/macros/Timing_estimation/plotting_slurm/error/%x.err
#SBATCH -p scavenger-gpu
#SBATCH --account=vossenlab
#SBATCH --mem=20G
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mail-user=rck32@duke.edu
source /cwork/rck32/ML_venv/bin/activate
python3 /cwork/rck32/eic/work_eic/macros/Timing_estimation/inference.py
