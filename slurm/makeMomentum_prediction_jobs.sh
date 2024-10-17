#!/bin/bash
current_date=$(date +"%B_%d")
workdir="/hpc/group/vossenlab/rck32/eic/work_eic"

# hipodir="/lustre19/expphy/cache/clas12/rg-a/production/montecarlo/clasdis/fall2018/torus+1/v1/bkg50nA_10604MeV"
slurm_output="${workdir}/root_files/Slurm"
#USER SET VALUES
num_events=50

out_folder="/hpc/group/vossenlab/rck32/eic/work_eic/slurm/output/output${current_date}"
error_folder="/hpc/group/vossenlab/rck32/eic/work_eic/slurm/error/error${current_date}"

rootname="prediction_sims_oct_7_pim_run_1_"
root_file_dir="/hpc/group/vossenlab/rck32/eic/work_eic/root_files/momentum_prediction/October_7"
processdir="/hpc/group/vossenlab/rck32/eic/epic_klm/"
runJobs="${workdir}/slurm/runJobs.sh"
touch $runJobs
chmod +x $runJobs
echo " " > $runJobs
i=0

TensorPathName="/hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/data/momentum_prediction_pulse/October_7"
inputTensorPathName="${TensorPathName}/input_pim_${num_events}events"
outputTensorPathName="${TensorPathName}/output_pim_${num_events}events"

if [ ! -d "$out_folder" ]; then
  mkdir -p "$out_folder"
fi

if [ ! -d "$error_folder" ]; then
  mkdir -p "$error_folder"
fi
if [ ! -d "$TensorPathName" ]; then
  mkdir -p "$TensorPathName"
fi

if [ ! -d "$root_file_dir" ]; then
  mkdir -p "$root_file_dir"
fi

for num in $(seq 0 1)
do
    file="${workdir}/slurm/shells/${rootname}${i}.sh"
    touch $file
    content="#!/bin/bash\n" 
    content+="#SBATCH --chdir=/hpc/group/vossenlab/rck32/eic/epic_klm\n"
    content+="#SBATCH --job-name=${rootname}${i}\n"
    content+="#SBATCH --output=${out_folder}/%x_mu.out\n"
    content+="#SBATCH --error=${error_folder}/%x_mu.err\n"
    content+="#SBATCH -p vossenlab-gpu\n"
    content+="#SBATCH --account=vossenlab\n"
    content+="#SBATCH --cpus-per-task=1\n"
    content+="#SBATCH --mem=50G\n"
    content+="#SBATCH --gpus=1\n"
    content+="#SBATCH --mail-user=rck32@duke.edu\n"
    content+="echo began job\n"
    content+="cat << EOF | /hpc/group/vossenlab/rck32/eic/eic-shell\n"
    content+="source install/setup.sh\n"
    content+="/usr/local/bin/ddsim --steeringFile /hpc/group/vossenlab/rck32/eic/work_eic/steering/scint_sensitive/variation_scint.py --compactFile /hpc/group/vossenlab/rck32/eic/epic_klm/epic_klmws_only.xml -G -N ${num_events} --gun.particle \"pi-\" --outputFile ${root_file_dir}/pim_${num_events}events_0_8_to_10GeV_90theta_origin_file_${i}.edm4hep.root --part.userParticleHandler=\"\"\n"
    content+="EOF\n"
    content+="echo finished ddsim\n"
    content+="exit"
    content+="echo began postprocessing\n"
    content+="source /hpc/group/vossenlab/rck32/ML_venv/bin/activate\n"
    content+="python3 /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/process_data_for_momentum_NN.py --filePathName ${root_file_dir}/pim_${num_events}events_0_8_to_10GeV_90theta_origin_file_${i}.edm4hep.root --inputTensorPathName ${inputTensorPathName}_${i}.pt --outputTensorPathName ${outputTensorPathName}_${i}.pt"
    echo -e "$content" > $file 
    echo "sbatch shells/${rootname}${i}.sh" >> $runJobs
    i=$((i+1))
done

trainFile="/hpc/group/vossenlab/rck32/eic/work_eic/slurm/shells/train_momentum.sh"

touch $trainFile
train_content="#!/bin/bash\n" 
train_content+="#SBATCH --chdir=/hpc/group/vossenlab/rck32/eic/epic_klm\n"
train_content+="#SBATCH --job-name=${rootname}${i}\n"
train_content+="#SBATCH --output=${out_folder}/%x_mu.out\n"
train_content+="#SBATCH --error=${error_folder}/%x_mu.err\n"
train_content+="#SBATCH -p vossenlab-gpu\n"
train_content+="#SBATCH --account=vossenlab\n"
train_content+="#SBATCH --cpus-per-task=1\n"
train_content+="#SBATCH --mem=50G\n"
train_content+="#SBATCH --gpus=1\n"
train_content+="#SBATCH --mail-user=rck32@duke.edu\n"
train_content+=""

