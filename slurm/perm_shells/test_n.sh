#!/bin/bash
#SBATCH --chdir=/hpc/group/vossenlab/rck32/eic/epic_klm
#SBATCH --job-name=test_predictor_October_23_n_model_n_data
#SBATCH --output=/hpc/group/vossenlab/rck32/eic/work_eic/slurm/output/outputOctober_28/%x_mu.out
#SBATCH --error=/hpc/group/vossenlab/rck32/eic/work_eic/slurm/error/errorOctober_28/%x_mu.err
#SBATCH -p scavenger-gpu
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --gpus=1
#SBATCH --mail-user=rck32@duke.edu

echo began job
echo began testing NN for prediction
source /hpc/group/vossenlab/rck32/ML_venv/bin/activate
python3 /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/testMomentumReco.py --inputTensorPath /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/data/momentum_prediction_pulse/October_22/input/ --outputTensorPath /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/data/momentum_prediction_pulse/October_22/output/ --plotPath October_29_n_model_n_run_1 --particle n --runInfo October_29_run_1_n_model_n_data_constant_02_7range --inputModelPath October_23_n_run_1/model.pth
