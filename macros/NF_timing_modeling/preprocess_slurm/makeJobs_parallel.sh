#!/bin/bash
current_date=$(date +"%B_%d")
eicdir="/hpc/group/vossenlab/rck32/eic"
workdir="${eicdir}/work_eic/macros/NF_timing_modeling/"

# hipodir="/lustre19/expphy/cache/clas12/rg-a/production/montecarlo/clasdis/fall2018/torus+1/v1/bkg50nA_10604MeV"
slurm_output="${workdir}/root_files/Slurm"
daydir="${eicdir}/work_eic/macros/NF_timing_modeling/data/${current_date}"
#USER SET VALUES
outputdir="${daydir}/Run_1/"

shell_dir="${eicdir}/work_eic/macros/NF_timing_modeling/preprocess_slurm/shells_parallel/"

out_folder="${eicdir}/work_eic/macros/NF_timing_modeling/preprocess_slurm/output/output${current_date}"
error_folder="${eicdir}/work_eic/macros/NF_timing_modeling/preprocess_slurm/error/error${current_date}"

infiledir="${eicdir}/work_eic/root_files/Jan_30/slurm/run_0_vary_events_one_segment_param/"

rootname="preproccess_600_z_vals"
processdir="${eicdir}/work_eic/macros/NF_timing_modeling/data/"
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

if [ ! -d "$shell_dir" ]; then
  mkdir -p "$shell_dir"
fi
for num in $(seq 0 600)
do
    file="${workdir}/preprocess_slurm/shells_parallel/${rootname}_${num}.sh"
    touch $file
    chmod +x $file
    content="#!/bin/bash\n" 
    content+="#SBATCH --job-name=${rootname}_${num}\n"
    content+="#SBATCH --output=${out_folder}/%x_mu.out\n"
    content+="#SBATCH --error=${error_folder}/%x_mu.err\n"
    content+="#SBATCH -p common\n"
    content+="#SBATCH --account=vossenlab\n"
    content+="#SBATCH --cpus-per-task=1\n"
    content+="#SBATCH --mem=5G\n"
    content+="#SBATCH --mail-user=rck32@duke.edu\n"
    content+="echo began job\n"
    content+="source /hpc/group/vossenlab/rck32/ML_venv/bin/activate\n"
    content+="python3 ${eicdir}/work_eic/macros/NF_timing_modeling/preprocess_fast.py --infile ${infiledir} --outfile ${outputdir}Vary_p_events_file_${num}_600_z_pos_fixed_z_hit_pos.pt --parallel 1 --file_num ${num}\n"
    echo -e "$content" > $file 
    echo "sbatch shells_parallel/${rootname}_${num}.sh" >> $runJobs
#     bash "./runJobs.sh"
    i=$((i+1))
done
