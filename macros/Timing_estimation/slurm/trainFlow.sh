#!/bin/bash
#SBATCH --job-name=train_flow_1
#SBATCH --output=/cwork/rck32/eic/work_eic/macros/Timing_estimation/slurm/output/outputJuly_24/%x_mu.out
#SBATCH --error=/cwork/rck32/eic/work_eic/macros/Timing_estimation/slurm/error/errorJuly_24/%x_mu.err
#SBATCH -p vossenlab-gpu
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --mail-user=rck32@duke.edu
#SBATCH --mail-type=END
echo began job
source /cwork/rck32/ML_venv/bin/activate
python3 /cwork/rck32/eic/work_eic/macros/Timing_estimation/train.py --useArgs --run_num 1 --num_epochs 6 --K 2 --hl 10 --hu 128 --lr 7e-6