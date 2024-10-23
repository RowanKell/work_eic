#!/bin/bash
#SBATCH --chdir=/hpc/group/vossenlab/rck32/eic/epic_klm
#SBATCH --job-name=prediction_sims_test
#SBATCH --output=/hpc/group/vossenlab/rck32/eic/work_eic/slurm/output/outputOctober_21/%x_mu.out
#SBATCH --error=/hpc/group/vossenlab/rck32/eic/work_eic/slurm/error/errorOctober_21/%x_mu.err
#SBATCH -p scavenger-gpu
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --gpus=1
#SBATCH --mail-user=rck32@duke.edu

echo began job

echo began postprocessing
source /hpc/group/vossenlab/rck32/ML_venv/bin/activate
python3 /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/process_data_for_momentum_NN_test.py --filePathName /hpc/group/vossenlab/rck32/eic/work_eic/root_files/momentum_prediction/October_21/pim_2000events_0_8_to_10GeV_90theta_origin_file_100.edm4hep.root --inputTensorPathName /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/data/momentum_prediction_pulse/October_21/test/test_in.pt --outputTensorPathName /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/data/momentum_prediction_pulse/October_21/test/test_out.pt
