#!/bin/bash
#SBATCH --account=vossenlab
#SBATCH -p common
#SBATCH --mem=30G
#SBATCH --job-name=pi_5GeV_one_sector
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --chdir=/cwork/rck32/eic/epic_klm
#SBATCH --output=/cwork/rck32/eic/work_eic/slurm/output/outputJun_12/%x.out
#SBATCH --error=/cwork/rck32/eic/work_eic/slurm/error/errorJun_12/%x.err
cat << EOF | /cwork/rck32/eic/eic-shell
source install/setup.sh
/usr/local/bin/ddsim --steeringFile /cwork/rck32/eic/work_eic/steering/energy_dep.py --compactFile /cwork/rck32/eic/epic_klm/epic_klmws_only.xml -G -N 5000 --gun.particle "pi-" --outputFile /cwork/rck32/eic/work_eic/root_files/June_12/One_sector/pi_5GeV_5k.edm4hep.root --part.userParticleHandler=""
EOF
