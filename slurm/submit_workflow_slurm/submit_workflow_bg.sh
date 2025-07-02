#!/bin/bash
#SBATCH --chdir=/hpc/group/vossenlab/rck32/eic/epic_klm
#SBATCH --job-name=submit_wkfl
#SBATCH --output=/hpc/group/vossenlab/rck32/eic/work_eic/slurm/submit_workflow_slurm/%x_mu.out
#SBATCH --error=/hpc/group/vossenlab/rck32/eic/work_eic/slurm/submit_workflow_slurm/%x_mu.err
#SBATCH -p common
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --mail-user=slurm_eicklm@outlook.com
#SBATCH --mail-type=END

WORK_EIC='/hpc/group/vossenlab/rck32/eic/work_eic/'
source $WORK_EIC/setup.sh
python3 $WORK_EIC/slurm/submit_workflow.py --deleteDfs False