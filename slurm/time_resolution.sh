#!/bin/bash
#SBATCH --account=vossenlab
#SBATCH -p common
#SBATCH --mem=30G
#SBATCH --job-name=mu_one_segment_5GeV_1_5m
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --chdir=/cwork/rck32/eic/epic_klm
#SBATCH --output=/cwork/rck32/eic/work_eic/slurm/output/outputJun_3/%x.out
#SBATCH --error=/cwork/rck32/eic/work_eic/slurm/error/errorJun_3/%x.err
cat << EOF | /cwork/rck32/eic/eic-shell
source install/setup.sh
/usr/local/bin/ddsim --steeringFile /cwork/rck32/eic/work_eic/steering/steer.py --compactFile /cwork/rck32/eic/epic_klm/epic_klmws_only.xml -G -N 1000 --gun.particle "mu-" --outputFile ../work_eic/root_files/time_res/one_segment_sensor_in_volume/mu_5GeV_1kevents_1_5m_1cm_3cm_28_layers.edm4hep.root --part.userParticleHandler=""
EOF
