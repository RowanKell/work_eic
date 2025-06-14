#!/bin/bash
#SBATCH --job-name=train_flow_2cm_1point8_time_constant_run_6
#SBATCH --output=/hpc/group/vossenlab/rck32/eic/work_eic/macros/NF_timing_modeling/slurm/output/outputJune_6/%x_mu.out
#SBATCH --error=/hpc/group/vossenlab/rck32/eic/work_eic/macros/NF_timing_modeling/slurm/error/errorJune_6/%x_mu.err
#SBATCH -p vossenlab-gpu
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --mail-user=rck32@duke.edu
#SBATCH --mail-type=END
echo began job
source /hpc/group/vossenlab/rck32/ML_venv/bin/activate
python3 /hpc/group/vossenlab/rck32/eic/work_eic/macros/NF_timing_modeling/train.py --useArgs --run_num 6 --num_epochs 18 --K 6 --hl 8 --hu 128 --lr 1e-4 --bs 15000 --num_files 600
