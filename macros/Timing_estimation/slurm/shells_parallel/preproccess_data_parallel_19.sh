#!/bin/bash
#SBATCH --job-name=preproccess_data_parallel_19
#SBATCH --output=/cwork/rck32/eic/work_eic/macros/Timing_estimation/slurm/output/outputJuly_03/%x_mu.out
#SBATCH --error=/cwork/rck32/eic/work_eic/macros/Timing_estimation/slurm/error/errorJuly_03/%x_mu.err
#SBATCH -p common
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --mail-user=rck32@duke.edu
#SBATCH --mail-type=END
echo began job
source /cwork/rck32/ML_venv/bin/activate
python3 /cwork/rck32/eic/work_eic/macros/Timing_estimation/preprocess.py --outfile /cwork/rck32/eic/work_eic/macros/Timing_estimation/data/July_03/Run_0/Full_2000events_file_19.pt --parallel 1 --file_num 19

