#!/bin/bash  
#SBATCH --chdir=/hpc/group/vossenlab/rck32/eic/epic_klm
#SBATCH --job-name=sector_scint_5k_5GeV
#SBATCH --output=/hpc/group/vossenlab/rck32/eic/work_eic/slurm/output/outputAugust_28/%x.out
#SBATCH --error=/hpc/group/vossenlab/rck32/eic/work_eic/slurm/error/errorAugust_28/%x.err
#SBATCH -p common
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --mail-user=rck32@duke.edu
echo began job
cat << EOF | /hpc/group/vossenlab/rck32/eic/eic-shell
source install/setup.sh
/usr/local/bin/ddsim --steeringFile ../work_eic/steering/scint_sensitive/variation_scint.py --compactFile /hpc/group/vossenlab/rck32/eic/epic_klm/epic_klmws_only.xml -G -N 5000 --gun.particle "neutron" --outputFile ../work_eic/root_files/August_28/sector_scint/run_1_n_5GeV_theta_90_5kevents.edm4hep.root --part.userParticleHandler=""
EOF