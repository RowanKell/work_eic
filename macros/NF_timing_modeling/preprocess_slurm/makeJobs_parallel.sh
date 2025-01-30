#!/bin/bash
current_date=$(date +"%B_%d")
workdir="/hpc/group/vossenlab/rck32/eic/work_eic/macros/NF_timing_modeling/"

slurm_output="${workdir}/root_files/Slurm"
daydir="/hpc/group/vossenlab/rck32/eic/work_eic/macros/NF_timing_modeling/data/${current_date}"
#USER SET VALUES
outputdir="${daydir}/Run_1/"

out_folder="/hpc/group/vossenlab/rck32/eic/work_eic/macros/NF_timing_modeling/preprocess_slurm/output/output${current_date}"
error_folder="/hpc/group/vossenlab/rck32/eic/work_eic/macros/NF_timing_modeling/preprocess_slurm/error/error${current_date}"

infiledir="/hpc/group/vossenlab/rck32/eic/work_eic/root_files/Photon_yield_param/no_QE_2cm/"

job_name="preprocess_600_z_vals"
processdir="/hpc/group/vossenlab/rck32/eic/work_eic/macros/NF_timing_modeling/data/"
runJobs="${workdir}/preprocess_slurm/runJobs.sh"
touch $runJobs
chmod +x $runJobs
echo " " > $runJobs
echo $daydir
i=0

if [ ! -d "$daydir" ]; then
  mkdir -p "$daydir"
fi

if [ ! -d "$outputdir" ]; then
  mkdir -p "$outputdir"
fi

if [ ! -d "$out_folder" ]; then
  mkdir -p "$out_folder"
fi

if [ ! -d "$error_folder" ]; then
  mkdir -p "$error_folder"
fi
for num in $(seq 501 600)
do
    file="${workdir}/preprocess_slurm/shells_parallel/${job_name}_${num}.sh"
    touch $file
    chmod +x $file
    content="#!/bin/bash\n" 
    content+="#SBATCH --job-name=${job_name}_${num}\n"
    content+="#SBATCH --output=${out_folder}/%x_mu.out\n"
    content+="#SBATCH --error=${error_folder}/%x_mu.err\n"
    content+="#SBATCH -p scavenger\n"
    content+="#SBATCH --account=vossenlab\n"
    content+="#SBATCH --cpus-per-task=1\n"
    content+="#SBATCH --mem=2G\n"
    content+="#SBATCH --mail-user=rck32@duke.edu\n"
    content+="echo began job\n"
    content+="source /hpc/group/vossenlab/rck32/ML_venv/bin/activate\n"
    content+="python3 /hpc/group/vossenlab/rck32/eic/work_eic/macros/NF_timing_modeling/NF_preprocess_fast.py --infile ${infiledir} --outfile ${outputdir}Jan_27_Vary_p_theta_z_file_${num}.pt --parallel 1 --file_num ${num}\n"
    echo -e "$content" > $file 
    echo "sbatch shells_parallel/${job_name}_${num}.sh" >> $runJobs
#     bash "./runJobs.sh"
    i=$((i+1))
done
