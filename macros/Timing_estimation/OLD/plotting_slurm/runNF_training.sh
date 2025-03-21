#!/bin/bash
#SBATCH --job-name=NF_training_july_11_12_flows_6_hl_64_hu_2000_bs
#SBATCH --output=/cwork/rck32/eic/work_eic/macros/Timing_estimation/plotting_slurm/output/%x.out
#SBATCH --error=/cwork/rck32/eic/work_eic/macros/Timing_estimation/plotting_slurm/error/%x.err
#SBATCH -p scavenger-gpu
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --mail-user=rck32@duke.edu
source /cwork/rck32/ML_venv/bin/activate
python3 /cwork/rck32/eic/work_eic/macros/Timing_estimation/NF_training.py --useArgs --K 12 --hu 64 --hl 6 --bs 2000
