#!/bin/bash  
#SBATCH --chdir=/hpc/group/vossenlab/rck32/eic/epic_klm
#SBATCH --job-name=one_segment_sensor_1000_mu_10GeV_10
#SBATCH --output=/hpc/group/vossenlab/rck32/eic/work_eic/slurm/output/outputJanuary_25/%x.out
#SBATCH --error=/hpc/group/vossenlab/rck32/eic/work_eic/slurm/error/errorJanuary_25/%x.err
#SBATCH -p scavenger
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --mail-user=rck32@duke.edu
echo began job
cat << EOF | /hpc/group/vossenlab/rck32/eic/eic-shell
source install/setup.sh
/usr/local/bin/ddsim --steeringFile ../work_eic/steering/sensor_sensitive/variation.py --compactFile /hpc/group/vossenlab/rck32/eic/epic_klm/epic_klmws_only.xml -G -N 1000 --gun.particle "mu-" --outputFile ../work_eic/root_files/time_res_one_segment_sensor/January_25/no_QE/mu_10GeV_1000events_no_QE_10.edm4hep.root --part.userParticleHandler="" --part.keepAllParticles True
EOF