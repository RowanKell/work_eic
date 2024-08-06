#!/bin/bash
#SBATCH --account=vossenlab
#SBATCH -p common
#SBATCH --mem=10G
#SBATCH --job-name=mu_time_res_1cm_7
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --chdir=/cwork/rck32/eic/epic_klm
#SBATCH --output=/cwork/rck32/eic/work_eic/slurm/output/outputAugust_1/%x.out
#SBATCH --error=/cwork/rck32/eic/work_eic/slurm/error/errorAugust_1/%x.err
echo "began job"
cat << EOF | /cwork/rck32/eic/eic-shell
source install/setup.sh
/usr/local/bin/ddsim --steeringFile /cwork/rck32/eic/work_eic/steering/sensor_sensitive/variation_z_pos.py --compactFile /cwork/rck32/eic/epic_klm/epic_klmws_only.xml -G -N 500 --gun.particle "mu-" --outputFile /cwork/rck32/eic/work_eic/root_files/August_1/run_1cm_optph/mu_5GeV_500_7.edm4hep.root --part.userParticleHandler=""
EOF
