#!/bin/tcsh                                                                                                                                                                
#SBATCH --account=clas12
#SBATCH --partition=production
#SBATCH --mem-per-cpu=4000
#SBATCH --job-name=KLM_sim
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=/u/home/rojokell/work_dir/slurm/output/outputMay_20/%x.out                                                                               
#SBATCH --error=/u/home/rojokell/work_dir/slurm/error/errorMay_20/%x.err
module purge -f
source /u/home/rojokell/.cshrc

cd /u/home/rojokell/eic
./eic-shell
cd epic_klm
source install/setup.sh

ddsim --steeringFile simulations/steering/npsim_local3.py --compactFile $DETECTOR_PATH/epic_klmws_only.xml -G -N 10 --gun.particle "mu-" --outputFile ../work_eic/root_files/May9/mu_5GeV_10events_run_1.edm4hep.root --part.userParticleHandler=""


