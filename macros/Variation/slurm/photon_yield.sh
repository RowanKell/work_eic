#!/bin/bash
#SBATCH --job-name=photon_yield
#SBATCH --output=/hpc/group/vossenlab/rck32/eic/work_eic/macros/Variation/slurm/%x_mu.out
#SBATCH --error=/hpc/group/vossenlab/rck32/eic/work_eic/macros/Variation/slurm/%x_mu.err
#SBATCH -p common
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --mail-user=rck32@duke.edu
#SBATCH --mail-type=END
echo began job
source /hpc/group/vossenlab/rck32/ML_venv/bin/activate
python3 /hpc/group/vossenlab/rck32/eic/work_eic/macros/Variation/photon_yield.py 