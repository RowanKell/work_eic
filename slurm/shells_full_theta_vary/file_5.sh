#!/bin/bash
#SBATCH --chdir=/cwork/rck32/eic/epic_klm
#SBATCH --job-name=file_5
#SBATCH --output=/cwork/rck32/eic/work_eic/slurm/output/outputJuly_02/%x_mu.out
#SBATCH --error=/cwork/rck32/eic/work_eic/slurm/error/errorJuly_02/%x_mu.err
#SBATCH -p common
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --mail-user=rck32@duke.edu
echo began job
cat << EOF | /cwork/rck32/eic/eic-shell
source install/setup.sh
/usr/local/bin/ddsim --steeringFile ../work_eic/steering/variation_pos.py --compactFile /cwork/rck32/eic/epic_klm/epic_klmws_only.xml -G -N 4000 --gun.particle "mu-" --outputFile ../work_eic/root_files/July_2/slurm/mu_vary_z_theta_no_save_all/5GeV_4000events_5.edm4hep.root --part.userParticleHandler="" --gun.position "(1769.3, 0.0, -357.250000)" --gun.thetaMin "0.00890" --gun.thetaMax "3.11492"
EOF

