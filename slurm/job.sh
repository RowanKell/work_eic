#!/bin/bash  
#SBATCH --chdir=/cwork/rck32/eic/epic_klm
#SBATCH --job-name=sector_scint
#SBATCH --output=/cwork/rck32/eic/work_eic/slurm/output/outputAugust_7/%x.out
#SBATCH --error=/cwork/rck32/eic/work_eic/slurm/error/errorAugust_7/%x.err
#SBATCH -p common
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --mail-user=rck32@duke.edu
echo began job
cat << EOF | /cwork/rck32/eic/eic-shell
source install/setup.sh
/usr/local/bin/ddsim --steeringFile ../work_eic/steering/scint_sensitive/variation_scint.py --compactFile /cwork/rck32/eic/epic_klm/epic_klmws_only.xml -G -N 500000 --gun.particle "neutron" --outputFile ../work_eic/root_files/August_7/sector_scint/run_1_n_0_8_10GeV_theta_90_500kevents.edm4hep.root --part.userParticleHandler=""
EOF


