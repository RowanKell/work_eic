#!/bin/bash
current_date=$(date +"%B_%d")
workdir="/cwork/rck32/eic/work_eic/macros/Timing_estimation"

# hipodir="/lustre19/expphy/cache/clas12/rg-a/production/montecarlo/clasdis/fall2018/torus+1/v1/bkg50nA_10604MeV"
slurm_output="${workdir}/root_files/Slurm"
daydir="/cwork/rck32/eic/work_eic/macros/Timing_estimation/data/${current_date}"
#USER SET VALUES
outputdir="${daydir}/Run_0/"

out_folder="/cwork/rck32/eic/work_eic/macros/Timing_estimation/slurm/output/output${current_date}"
error_folder="/cwork/rck32/eic/work_eic/macros/Timing_estimation/slurm/error/error${current_date}"

infiledir="/cwork/rck32/eic/work_eic/root_files/July_1/slurm/mu_vary_z_theta_no_save_all/"

rootname="preproccess_data_parallel_cuts"
processdir="/cwork/rck32/eic/work_eic/macros/Timing_estimation/"
runJobs="${workdir}/slurm/runJobs.sh"
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
for num in $(seq 0 20)
do
    file="${workdir}/slurm/shells_parallel/${rootname}_${num}.sh"
    touch $file
    chmod +x $file
    content="#!/bin/bash\n" 
    content+="#SBATCH --job-name=${rootname}_${num}\n"
    content+="#SBATCH --output=${out_folder}/%x_mu.out\n"
    content+="#SBATCH --error=${error_folder}/%x_mu.err\n"
    content+="#SBATCH -p common\n"
    content+="#SBATCH --account=vossenlab\n"
    content+="#SBATCH --cpus-per-task=1\n"
    content+="#SBATCH --mem=8G\n"
    content+="#SBATCH --mail-user=rck32@duke.edu\n"
    content+="echo began job\n"
    content+="source /cwork/rck32/ML_venv/bin/activate\n"
    content+="python3 /cwork/rck32/eic/work_eic/macros/Timing_estimation/preprocess.py --outfile ${outputdir}Full_4000events_file_${num}_w_cuts.pt --parallel 1 --file_num ${num}\n"
    echo -e "$content" > $file 
    echo "sbatch shells_parallel/${rootname}_${num}.sh" >> $runJobs
#     bash "./runJobs.sh"
    i=$((i+1))
done
