#!/bin/bash
#SBATCH --chdir=/hpc/group/vossenlab/rck32/eic/epic_klm
#SBATCH --job-name=test_predictor_October_23_K_L_model_K_L_data
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
python3 /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/testMomentumReco.py --inputTensorPath /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/data/momentum_prediction_pulse/October_23_K_L/input/ --outputTensorPath /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/data/momentum_prediction_pulse/October_23_K_L/output/ --plotPath October_29_K_L_model_K_L_run_1 --particle K_L --runInfo October_29_run_1_K_L_model_K_L_data_constant_0_2_7range --inputModelPath October_23_K_L_run_1/model.pth
