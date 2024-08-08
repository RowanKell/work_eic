#!/bin/bash
#SBATCH --job-name=train_flow_7
#SBATCH --output=/cwork/rck32/eic/work_eic/macros/Timing_estimation/slurm/output/outputAugust_6/%x_mu.out
#SBATCH --error=/cwork/rck32/eic/work_eic/macros/Timing_estimation/slurm/error/errorAugust_6/%x_mu.err
#SBATCH -p vossenlab-gpu
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --mail-user=rck32@duke.edu
#SBATCH --mail-type=END
echo began job
source /cwork/rck32/ML_venv/bin/activate
python3 /cwork/rck32/eic/work_eic/macros/Timing_estimation/train.py --useArgs --run_num 7 --num_epochs 18 --K 8 --hl 26 --hu 256 --lr 1e-5