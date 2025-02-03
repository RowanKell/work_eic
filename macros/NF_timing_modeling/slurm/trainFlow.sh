#!/bin/bash
#SBATCH --job-name=train_flow_2cm_run_14
#SBATCH --output=/hpc/group/vossenlab/rck32/eic/work_eic/macros/NF_timing_modeling/slurm/output/outputJanuary_30/%x_mu.out
#SBATCH --error=/hpc/group/vossenlab/rck32/eic/work_eic/macros/NF_timing_modeling/slurm/error/errorJanuary_30/%x_mu.err
#SBATCH -p vossenlab-gpu
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --mail-user=rck32@duke.edu
#SBATCH --mail-type=END
echo began job
source /hpc/group/vossenlab/rck32/ML_venv/bin/activate
python3 /hpc/group/vossenlab/rck32/eic/work_eic/macros/NF_timing_modeling/train.py --useArgs --run_num 14 --num_epochs 18 --K 4 --hl 10 --hu 56 --lr 2e-5 --bs 20000