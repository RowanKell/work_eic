#!/bin/bash
#SBATCH --job-name=prepare_momentum_predict_5k
#SBATCH --output=/hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/slurm/output/outputSeptember_12/%x.out
#SBATCH --error=/hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/slurm/error/errorSeptember_12/%x.err
#SBATCH -p scavenger-gpu
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --mail-user=rck32@duke.edu
#SBATCH --mail-type=END
echo began job
source /hpc/group/vossenlab/rck32/ML_venv/bin/activate
python3 /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/process_data_for_momentum_NN.py