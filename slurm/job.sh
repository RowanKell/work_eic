#!/bin/bash  
#SBATCH --chdir=/cwork/rck32/eic/epic_klm
#SBATCH --job-name=KLM_sim
#SBATCH --output=/cwork/rck32/eic/work_eic/slurm/output/outputJune_17/%x.out
#SBATCH --error=/cwork/rck32/eic/work_eic/slurm/error/errorJune_17/%x.err
#SBATCH -p vossenlab-gpu
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --mail-user=rck32@duke.edu
echo began job
cat << EOF | /cwork/rck32/eic/eic-shell
source install/setup.sh
/usr/local/bin/ddsim --steeringFile ../work_eic/steering/variation.py --compactFile /cwork/rck32/eic/epic_klm/epic_klmws_only.xml -G -N 5 --gun.particle "mu-" --outputFile ../work_eic/root_files/June_17/variation_two_sensors/mu_5GeV_variation_5events_run_1.edm4hep.root --part.userParticleHandler=""
EOF


