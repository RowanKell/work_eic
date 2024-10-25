#!/bin/bash
#SBATCH --chdir=/hpc/group/vossenlab/rck32/eic/epic_klm
#SBATCH --job-name=optimize_October_22_pim_run_5
#SBATCH --output=/hpc/group/vossenlab/rck32/eic/work_eic/slurm/output/outputOctober_22/%x_mu.out
#SBATCH --error=/hpc/group/vossenlab/rck32/eic/work_eic/slurm/error/errorOctober_22/%x_mu.err
#SBATCH -p vossenlab-gpu
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --gpus=1
#SBATCH --mail-user=rck32@duke.edu

echo began job
echo began optimiing NN for prediction
source /hpc/group/vossenlab/rck32/ML_venv/bin/activate
python3 /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/hyperparameter_study.py --inputTensorPath /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/data/momentum_prediction_pulse/October_21/input/ --outputTensorPath /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/data/momentum_prediction_pulse/October_21/output/ --basePath /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/hyper_param_studies --n_trials 200
