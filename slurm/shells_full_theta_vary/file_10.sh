#!/bin/bash
#SBATCH --chdir=/cwork/rck32/eic/epic_klm
#SBATCH --job-name=file_10_TEST
#SBATCH --output=/cwork/rck32/eic/work_eic/slurm/output/outputJuly_01/%x_mu.out
#SBATCH --error=/cwork/rck32/eic/work_eic/slurm/error/errorJuly_01/%x_mu.err
#SBATCH -p common
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --mail-user=rck32@duke.edu
echo began job
cat << EOF | /cwork/rck32/eic/eic-shell
source install/setup.sh
/usr/local/bin/ddsim --steeringFile ../work_eic/steering/variation_pos.py --compactFile /cwork/rck32/eic/epic_klm/epic_klmws_only.xml -G -N 500 --gun.particle "mu-" --outputFile ../work_eic/root_files/July_1/slurm/mu_vary_z_theta/5GeV_500events_TEST.edm4hep.root --part.userParticleHandler="" --gun.position "(1769.3, 0.0, 17.500000)" --gun.thetaMin "0.01335" --gun.thetaMax "3.12825"
EOF

