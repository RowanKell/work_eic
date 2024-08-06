#!/bin/bash
#SBATCH --job-name=quick_train_flow_10
#SBATCH --output=/cwork/rck32/eic/work_eic/macros/Timing_estimation/slurm/output/outputAugust_5/%x_mu.out
#SBATCH --error=/cwork/rck32/eic/work_eic/macros/Timing_estimation/slurm/error/errorAugust_5/%x_mu.err
#SBATCH -p scavenger-gpu
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --mail-user=rck32@duke.edu
#SBATCH --mail-type=END
echo began job
source /cwork/rck32/ML_venv/bin/activate
python3 /cwork/rck32/eic/work_eic/macros/Timing_estimation/train.py --useArgs --run_num 10 --num_epochs 4 --K 1 --hl 4 --hu 64 --lr 2e-5