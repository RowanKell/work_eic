#!/bin/bash  
#SBATCH --chdir=/hpc/group/vossenlab/rck32/eic/epic_klm
#SBATCH --job-name=testing_photon_yield
#SBATCH --output=/hpc/group/vossenlab/rck32/eic/work_eic/slurm/output/march_3_%x.out
#SBATCH --error=/hpc/group/vossenlab/rck32/eic/work_eic/slurm/error/march_3_%x.err
#SBATCH -p common
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --mail-user=rck32@duke.edu
echo began job
cat << EOF | /hpc/group/vossenlab/rck32/eic/eic-shell
source /hpc/group/vossenlab/rck32/eic/epic_klm/install/setup.sh
/usr/local/bin/ddsim --steeringFile /hpc/group/vossenlab/rck32/eic/work_eic/steering/sensor_sensitive/variation_pos_keepALL.py --compactFile /hpc/group/vossenlab/rck32/eic/epic_klm/epic_klmws_only.xml --runType "batch" -G -N 50 --gun.particle "mu-" --outputFile /hpc/group/vossenlab/rck32/eic/work_eic/root_files/test/scint_sense_opticalph_march3_one_layer_50events_mum.edm4hep.root --macroFile ddsim_shells/myvis.mac --part.userParticleHandler=""
EOF