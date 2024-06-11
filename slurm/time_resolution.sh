#!/bin/bash
#SBATCH --account=vossenlab
#SBATCH -p common
#SBATCH --mem=30G
#SBATCH --job-name=time_res_pi_one_segment_5GeV_1_5m
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --chdir=/cwork/rck32/eic/epic_klm
#SBATCH --output=/cwork/rck32/eic/work_eic/slurm/output/outputJun_6/%x.out
#SBATCH --error=/cwork/rck32/eic/work_eic/slurm/error/errorJun_6/%x.err
cat << EOF | /cwork/rck32/eic/eic-shell
source install/setup.sh
/usr/local/bin/ddsim --steeringFile /cwork/rck32/eic/work_eic/steering/steer.py --compactFile /cwork/rck32/eic/epic_klm/epic_klmws_only.xml -G -N 10000 --gun.particle "pi-" --outputFile ../work_eic/root_files/time_res/one_segment_sensor_in_volume/June_6/pi_5GeV_10kevents_1_5m_1cm_3cm.edm4hep.root --part.userParticleHandler=""
EOF
