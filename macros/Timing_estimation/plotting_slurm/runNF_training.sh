#!/bin/bash
#SBATCH --job-name=NF_training_july_8_12_flows_6_hl_64_hu_500_bs
#SBATCH --output=/cwork/rck32/eic/work_eic/macros/Timing_estimation/plotting_slurm/output/%x.out
#SBATCH --error=/cwork/rck32/eic/work_eic/macros/Timing_estimation/plotting_slurm/error/%x.err
#SBATCH -p gpu-common
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --mail-user=rck32@duke.edu
source /cwork/rck32/ML_venv/bin/activate
python3 /cwork/rck32/eic/work_eic/macros/Timing_estimation/NF_training.py
