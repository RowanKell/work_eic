#!/bin/bash
current_date=$(date +"%B_%d")
workdir="/hpc/group/vossenlab/rck32/eic/work_eic"

# hipodir="/lustre19/expphy/cache/clas12/rg-a/production/montecarlo/clasdis/fall2018/torus+1/v1/bkg50nA_10604MeV"
slurm_output="${workdir}/root_files/Slurm"
daydir="${slurm_output}/${current_date}"
#USER SET VALUES
outputdir="${daydir}/Run_0/"

out_folder="/hpc/group/vossenlab/rck32/eic/work_eic/slurm/output/output${current_date}"
error_folder="/hpc/group/vossenlab/rck32/eic/work_eic/slurm/error/error${current_date}"

rootname="file_"
processdir="/hpc/group/vossenlab/rck32/eic/epic_klm/"
runJobs="${workdir}/slurm/runJobs.sh"
touch $runJobs
chmod +x $runJobs
echo " " > $runJobs
echo $daydir
i=0

if [ ! -d "$daydir" ]; then
  mkdir "$daydir"
fi

if [ ! -d "$outputdir" ]; then
  mkdir "$outputdir"
fi

if [ ! -d "$out_folder" ]; then
  mkdir "$out_folder"
fi

if [ ! -d "$error_folder" ]; then
  mkdir "$error_folder"
fi

for num in $(seq 1 40)
do
    file="${workdir}/slurm/shells/${rootname}${i}.sh"
    touch $file
    content="#!/bin/bash\n"
    content+="#SBATCH --account=vossenlab\n"
    content+="#SBATCH -p common\n"
    content+="#SBATCH --mem=8G\n"
    content+="#SBATCH --job-name=${rootname}${i}\n"
    content+="#SBATCH --cpus-per-task=1\n"
    content+="#SBATCH --time=24:00:00\n"
    content+="#SBATCH --chdir=/hpc/group/vossenlab/rck32/eic/epic_klm\n"
    content+="#SBATCH --output=${out_folder}/%x.out\n"
    content+="#SBATCH --error=${error_folder}/%x.err\n"
    content+="cat << EOF | /hpc/group/vossenlab/rck32/eic/eic-shell\n"
    content+="source install/setup.sh\n"
    content+="/usr/local/bin/ddsim --steeringFile simulations/steering/npsim_local3.py --compactFile /hpc/group/vossenlab/rck32/eic/epic_klm/epic_klmws_only.xml -G -N 30 --gun.particle \"pi-\" --outputFile ${outputdir}pi_5GeV_30events_run_0_file_${i}.edm4hep.root --part.userParticleHandler=\"\"\n"
    content+="EOF\n"
    echo -e "$content" > $file 
    echo "sbatch shells/${rootname}${i}.sh" >> $runJobs
    i=$((i+1))
done