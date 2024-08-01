#!/bin/bash
#SBATCH --account=vossenlab
#SBATCH -p common
#SBATCH --mem=10G
#SBATCH --job-name=mu_time_res_1cm
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --chdir=/cwork/rck32/eic/epic_klm
#SBATCH --output=/cwork/rck32/eic/work_eic/slurm/output/outputJuly_31/%x.out
#SBATCH --error=/cwork/rck32/eic/work_eic/slurm/error/errorJuly_31/%x.err
cat << EOF | /cwork/rck32/eic/eic-shell
source install/setup.sh
/usr/local/bin/ddsim --steeringFile /cwork/rck32/eic/work_eic/steering/variation.py --compactFile /cwork/rck32/eic/epic_klm/epic_klmws_only.xml -G -N 10000 --gun.particle "mu-" --outputFile /cwork/rck32/eic/work_eic/root_files/July_31/run_2_1cm/mu_0_8_to_10GeV_10k.edm4hep.root --part.userParticleHandler=""
EOF
