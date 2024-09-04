#!/bin/bash  
#SBATCH --chdir=/hpc/group/vossenlab/rck32/eic/epic_klm
#SBATCH --job-name=sector_sensor_20_variable_momentum
#SBATCH --output=/hpc/group/vossenlab/rck32/eic/work_eic/slurm/output/outputSeptember_3/%x.out
#SBATCH --error=/hpc/group/vossenlab/rck32/eic/work_eic/slurm/error/errorSeptember_3/%x.err
#SBATCH -p common
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --mail-user=rck32@duke.edu
echo began job
cat << EOF | /hpc/group/vossenlab/rck32/eic/eic-shell
source install/setup.sh
/usr/local/bin/ddsim --steeringFile ../work_eic/steering/sensor_sensitive/variation.py --compactFile /hpc/group/vossenlab/rck32/eic/epic_klm/epic_klmws_only.xml -G -N 20 --gun.particle "mu-" --outputFile ../work_eic/root_files/September_3/sector_sensor/run_1_mum_0_8_10GeV_theta_90_20events.edm4hep.root --part.userParticleHandler=""
EOF