#!/bin/bash  
#SBATCH --chdir=/hpc/group/vossenlab/rck32/eic/epic_klm
#SBATCH --job-name=segment_sensor_50_variable_energy
#SBATCH --output=/hpc/group/vossenlab/rck32/eic/work_eic/slurm/output/outputSeptember_6/%x.out
#SBATCH --error=/hpc/group/vossenlab/rck32/eic/work_eic/slurm/error/errorSeptember_6/%x.err
#SBATCH -p common
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --mail-user=rck32@duke.edu
echo began job
cat << EOF | /hpc/group/vossenlab/rck32/eic/eic-shell
source install/setup.sh
/usr/local/bin/ddsim --steeringFile ../work_eic/steering/sensor_sensitive/variation_pos_keepALL.py --compactFile /hpc/group/vossenlab/rck32/eic/epic_klm/epic_klmws_only.xml -G -N 50 --gun.particle "mu-" --outputFile ../work_eic/root_files/Photon_yield_param/sept_6/test.edm4hep.root --part.userParticleHandler="" --gun.position "(1769.3,0,0)" --gun.thetaMin 90 --gun.thetaMax 90
EOF