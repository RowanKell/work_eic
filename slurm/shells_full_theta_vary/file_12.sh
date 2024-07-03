#!/bin/bash
#SBATCH --chdir=/cwork/rck32/eic/epic_klm
#SBATCH --job-name=file_12
#SBATCH --output=/cwork/rck32/eic/work_eic/slurm/output/outputJuly_03/%x_mu.out
#SBATCH --error=/cwork/rck32/eic/work_eic/slurm/error/errorJuly_03/%x_mu.err
#SBATCH -p common
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --mail-user=rck32@duke.edu
echo began job
cat << EOF | /cwork/rck32/eic/eic-shell
source install/setup.sh
/usr/local/bin/ddsim --steeringFile ../work_eic/steering/variation_pos.py --compactFile /cwork/rck32/eic/epic_klm/epic_klmws_only.xml -G -N 8000 --gun.particle "mu-" --outputFile ../work_eic/root_files/July_2/slurm/mu_vary_p_z_theta_no_save_all/vary_p_8000events_12.edm4hep.root --part.userParticleHandler="" --gun.position "(1769.3, 0.0, 167.400000)" --gun.thetaMin "0.01668" --gun.thetaMax "3.13048"
EOF

