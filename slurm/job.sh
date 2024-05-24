#!/bin/bash  
#SBATCH --job-name=KLM_sim
#SBATCH --output=/hpc/group/vossenlab/rck32/eic/work_eic/slurm/output/outputMay20/%x.out
#SBATCH --error=/hpc/group/vossenlab/rck32/eic/work_eic/slurm/error/errorMay20/%x.err
#SBATCH -p vossenlab-gpu
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --mail-user=rck32@duke.edu
echo began job
cd /hpc/group/vossenlab/rck32/eic/epic_klm
cat << EOF | /hpc/group/vossenlab/rck32/eic/eic-shell
source install/setup.sh
/usr/local/bin/ddsim --steeringFile simulations/steering/npsim_local3.py --compactFile /hpc/group/vossenlab/rck32/eic/epic_klm/epic_klmws_only.xml -G -N 30 --gun.particle "mu-" --outputFile ../work_eic/root_files/May21/mu_5GeV_30events_run_1.edm4hep.root --part.userParticleHandler=""
EOF


