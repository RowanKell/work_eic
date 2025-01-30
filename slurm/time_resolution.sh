#!/bin/bash
#SBATCH --account=vossenlab
#SBATCH -p common
#SBATCH --mem=10G
#SBATCH --job-name=mu_time_res_scint_run_1
#SBATCH --cpus-per-task=1
#SBATCH --chdir=/hpc/group/vossenlab/rck32/eic/epic_klm
#SBATCH --output=/hpc/group/vossenlab/rck32/eic/work_eic/slurm/output/outputJanuary_28/%x.out
#SBATCH --error=/hpc/group/vossenlab/rck32/eic/work_eic/slurm/error/errorJanuary_28/%x.err
echo "began job"
cat << EOF | /hpc/group/vossenlab/rck32/eic/eic-shell
source install/setup.sh
/usr/local/bin/ddsim --steeringFile /hpc/group/vossenlab/rck32/eic/work_eic/steering/scint_sensitive/steer.py --compactFile /hpc/group/vossenlab/rck32/eic/epic_klm/epic_klmws_only.xml -G -N 5000 --gun.particle "mu-" --outputFile /hpc/group/vossenlab/rck32/eic/work_eic/root_files/time_res_one_segment_scint/January_28/run_1_mum_10GeV_theta_90_5kevents.edm4hep.root --part.userParticleHandler=""
EOF
