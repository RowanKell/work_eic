#!/bin/bash
#SBATCH --chdir=/hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation
#SBATCH --job-name=mem_study
#SBATCH --output=/hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/slurm/output/outputOctober_17/%x_mu.out
#SBATCH --error=/hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/slurm/error/errorOctober_17/%x_mu.err
#SBATCH -p vossenlab-gpu
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem=10G
#SBATCH --mail-user=rck32@duke.edu
#SBATCH --mail-type=END
echo began job
source /hpc/group/vossenlab/rck32/ML_venv/bin/activate
python3 /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/mem_study_process_data_for_momentum_NN.py
