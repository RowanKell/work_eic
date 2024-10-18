#!/bin/bash
current_date=$(date +"%B_%d")
rootdir="/hpc/group/vossenlab/rck32/eic/work_eic/root_files/momentum_prediction/October_7"
workdir="/hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation"

#USER SET VALUES
slurm_dir="/hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/slurm"
out_folder="${slurm_dir}/output/output${current_date}"
error_folder="${slurm_dir}/error/error${current_date}"

runJobs="${slurm_dir}/runJobs.sh"
touch $runJobs
chmod +x $runJobs
echo " " > $runJobs
i=0

#root_file_name="${rootdir}/n_5kevents_0_8_to_10GeV_90theta_origin_file_"
root_file_name="${rootdir}/pim_50events_0_8_to_10GeV_90theta_origin_file_"


if [ ! -d "$out_folder" ]; then
  mkdir -p "$out_folder"
fi

if [ ! -d "$error_folder" ]; then
  mkdir -p "$error_folder"
fi
shell_name="m_prediction_"
for num in $(seq 0 1)
do
    file="${slurm_dir}/shells/${shell_name}${num}.sh"
    input_tensor_file="${workdir}/data/momentum_prediction_pulse/October_17/file_${num}_pim_50_0_8_to_10GeV_nn_inputs.pt"
    output_tensor_file="${workdir}/data/momentum_prediction_pulse/October_17/file_${num}_pim_50_0_8_to_10GeV_nn_outputs.pt"
    touch $file
    content="#!/bin/bash\n" 
    content+="#SBATCH --chdir=${workdir}\n"
    content+="#SBATCH --job-name=${shell_name}${num}\n"
    content+="#SBATCH --output=${out_folder}/%x_mu.out\n"
    content+="#SBATCH --error=${error_folder}/%x_mu.err\n"
    content+="#SBATCH -p vossenlab-gpu\n"
    content+="#SBATCH --account=vossenlab\n"
    content+="#SBATCH --cpus-per-task=1\n"
    content+="#SBATCH --gpus=1\n"
    content+="#SBATCH --mem=100G\n"
    content+="#SBATCH --mail-user=rck32@duke.edu\n"
    content+="#SBATCH --mail-type=END\n"
    content+="echo began job\n"
    content+="source /hpc/group/vossenlab/rck32/ML_venv/bin/activate\n"
    content+="python3 /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/process_data_for_momentum_NN.py --filePathName ${root_file_name}${num}.edm4hep.root --inputTensorPathName ${input_tensor_file} --outputTensorPathName ${output_tensor_file}"
    echo -e "$content" > $file 
    echo "sbatch shells/${shell_name}${num}.sh" >> $runJobs
    i=$((i+1))
done
