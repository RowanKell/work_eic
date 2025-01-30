#!/bin/bash
#SBATCH --job-name=n_momentum_predict_50k
#SBATCH --output=/hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/slurm/output/outputSeptember_12/%x.out
#SBATCH --error=/hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/slurm/error/errorSeptember_12/%x.err
#SBATCH -p vossenlab-gpu
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --mail-user=rck32@duke.edu
#SBATCH --mail-type=END
echo began job
source /hpc/group/vossenlab/rck32/ML_venv/bin/activate
python3 /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/train_momentum_predictor.py