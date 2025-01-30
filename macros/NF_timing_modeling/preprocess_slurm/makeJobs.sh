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

rootname="preproccess_data"
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

file="${workdir}/slurm/shells/${rootname}.sh"
touch $file
chmod +x $file
content="#!/bin/bash\n" 
content+="#SBATCH --job-name=${rootname}\n"
content+="#SBATCH --output=${out_folder}/%x_mu.out\n"
content+="#SBATCH --error=${error_folder}/%x_mu.err\n"
content+="#SBATCH -p common\n"
content+="#SBATCH --account=vossenlab\n"
content+="#SBATCH --cpus-per-task=1\n"
content+="#SBATCH --mem=8G\n"
content+="#SBATCH --mail-user=rck32@duke.edu\n"
content+="echo began job\n"
content+="source /cwork/rck32/ML_venv/bin/activate\n"
content+="python3 /cwork/rck32/eic/work_eic/macros/Timing_estimation/preprocess.py --outfile ${outputdir}Full_21file_200events.pt\n"
echo -e "$content" > $file 
echo "sbatch shells/${rootname}.sh" >> $runJobs
bash "./runJobs.sh"
i=$((i+1))
