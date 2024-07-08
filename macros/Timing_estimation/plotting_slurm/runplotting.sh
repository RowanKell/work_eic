#!/bin/bash
#SBATCH --job-name=plot_full_uniform_fixed_p
#SBATCH --output=/cwork/rck32/eic/work_eic/macros/Timing_estimation/plotting_slurm/output/%x_mu.out
#SBATCH --error=/cwork/rck32/eic/work_eic/macros/Timing_estimation/plotting_slurm/error/%x.err
#SBATCH -p common
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --mail-user=rck32@duke.edu
source /cwork/rck32/ML_venv/bin/activate
python3 /cwork/rck32/eic/work_eic/macros/Timing_estimation/plotting_slurm/plotting.py
