#!/bin/bash
#SBATCH --job-name=train_flow_5_55cm_run_3
#SBATCH --output=/hpc/group/vossenlab/rck32/eic/work_eic/macros/NF_timing_modeling/slurm/output/outputMarch_31/%x_mu.out
#SBATCH --error=/hpc/group/vossenlab/rck32/eic/work_eic/macros/NF_timing_modeling/slurm/error/errorMarch_31/%x_mu.err
#SBATCH -p vossenlab-gpu
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --mail-user=rck32@duke.edu
#SBATCH --mail-type=END
echo began job
source /hpc/group/vossenlab/rck32/ML_venv/bin/activate
python3 /hpc/group/vossenlab/rck32/eic/work_eic/macros/NF_timing_modeling/train.py --useArgs --run_num 3 --num_epochs 25 --K 12 --hl 26 --hu 256 --lr 5e-4 --bs 15000