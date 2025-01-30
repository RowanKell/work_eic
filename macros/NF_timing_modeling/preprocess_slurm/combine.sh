#!/bin/bash
#SBATCH --job-name=combine_processed_data
#SBATCH --output=/cwork/rck32/eic/work_eic/macros/Timing_estimation/preprocess_slurm/output/outputJuly_23/%x_mu.out
#SBATCH --error=/cwork/rck32/eic/work_eic/macros/Timing_estimation/preprocess_slurm/error/errorJuly_23/%x_mu.err
#SBATCH -p common
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --mail-user=rck32@duke.edu
echo began job
source /cwork/rck32/ML_venv/bin/activate
python3 /cwork/rck32/eic/work_eic/macros/Timing_estimation/combine_preprocessed_data.py