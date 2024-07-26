#!/bin/bash
#SBATCH --job-name=sample_timing
#SBATCH --output=/cwork/rck32/eic/work_eic/macros/Timing_estimation/sampling_slurm/output/outputJuly_25/%x_mu.out
#SBATCH --error=/cwork/rck32/eic/work_eic/macros/Timing_estimation/sampling_slurm/error/errorJuly_25/%x_mu.err
#SBATCH -p vossenlab-gpu
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --mail-user=rck32@duke.edu
#SBATCH --mail-type=END
echo began job
source /cwork/rck32/ML_venv/bin/activate
python3 /cwork/rck32/eic/work_eic/macros/Timing_estimation/sample.py --no-useArgs