#!/bin/bash
#SBATCH --chdir=/hpc/group/vossenlab/rck32/eic/epic_klm
#SBATCH --job-name=March10_mobo_test_0_50events_run_0_March_17_0
#SBATCH --output=/hpc/group/vossenlab/rck32/eic/work_eic/slurm/output/outputMarch_17/%x.out
#SBATCH --error=/hpc/group/vossenlab/rck32/eic/work_eic/slurm/error/errorMarch_17/%x.err
#SBATCH -p scavenger-gpu
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --gpus=1
#SBATCH --mail-user=rck32@duke.edu
#SBATCH --mail-type=FAIL

echo began job


cat << EOF | /hpc/group/vossenlab/rck32/eic/eic-shell
source install/setup.sh

#run ddsim, capture status

#########   DDSIM    ##########
echo "Running ddsim with steeringFile input"
/usr/local/bin/ddsim  --compactFile /hpc/home/rck32/groupdir/eic/dRICH-MOBO/MOBO-tools/epic_klm//epic_klmws_only_0.xml -G --numberOfEvents 50 --steeringFile /hpc/group/vossenlab/rck32/eic/work_eic/steering/scint_sensitive/sector.py --outputFile /hpc/group/vossenlab/rck32/eic/work_eic/root_files/Clustering/March_17/March10_mobo_test_0_50events_run_0_50_0.edm4hep.root  --part.userParticleHandler=""
echo "DDSIM completed successfully"
echo began process root file
#########   PROCESS  ##########
python3 /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/process_root_file.py --filePathName /hpc/group/vossenlab/rck32/eic/work_eic/root_files/Clustering/March_17/March10_mobo_test_0_50events_run_0_50_0.edm4hep.root  --processedDataPath /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/data/processed_data/March10_mobo_test_0_50events_run_0_0.json --geometryType 1 --deleteROOTFile
EOF

echo "Beginning Analysis with analyze_data_old.py"    
source /hpc/group/vossenlab/rck32/ML_venv/bin/activate

#########   ANALYZE    ##########
python3 /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/analyze_data.py --inputProcessedData /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/data/processed_data/March10_mobo_test_0_50events_run_0_0.json --outputDataframePathName /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/data/df/March10_mobo_test_0_50events_run_0_0.csv --useCFD --batchSize 10000 --deleteJSON

deactivate
echo ENDING JOB
