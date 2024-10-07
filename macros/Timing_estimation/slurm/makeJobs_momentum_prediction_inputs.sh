#!/bin/bash
current_date=$(date +"%B_%d")
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


if [ ! -d "$out_folder" ]; then
  mkdir -p "$out_folder"
fi

if [ ! -d "$error_folder" ]; then
  mkdir -p "$error_folder"
fi
shell_name="m_prediction_"
for num in $(seq 36 50)
do
    file="${slurm_dir}/shells/${shell_name}${num}.sh"
    touch $file
    content="#!/bin/bash\n" 
    content+="#SBATCH --chdir=${workdir}\n"
    content+="#SBATCH --job-name=${shell_name}${num}\n"
    content+="#SBATCH --output=${out_folder}/%x_mu.out\n"
    content+="#SBATCH --error=${error_folder}/%x_mu.err\n"
    content+="#SBATCH -p scavenger-gpu\n"
    content+="#SBATCH --account=vossenlab\n"
    content+="#SBATCH --cpus-per-task=1\n"
    content+="#SBATCH --gpus=1\n"
    content+="#SBATCH --mem=100G\n"
    content+="#SBATCH --mail-user=rck32@duke.edu\n"
    content+="#SBATCH --mail-type=END\n"
    content+="echo began job\n"
    content+="source /hpc/group/vossenlab/rck32/ML_venv/bin/activate\n"
    content+="python3 /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/process_data_for_momentum_NN.py --fileNum ${num}"
    echo -e "$content" > $file 
    echo "sbatch shells/${shell_name}${num}.sh" >> $runJobs
    i=$((i+1))
done
