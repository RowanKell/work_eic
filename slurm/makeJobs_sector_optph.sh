#!/bin/bash
current_date=$(date +"%B_%d")
workdir="/cwork/rck32/eic/work_eic"

# hipodir="/lustre19/expphy/cache/clas12/rg-a/production/montecarlo/clasdis/fall2018/torus+1/v1/bkg50nA_10604MeV"
slurm_output="${workdir}/root_files/Slurm"
daydir="/cwork/rck32/eic_output/pi_sims/${current_date}"
#USER SET VALUES
outputdir="${daydir}/Run_0_no_keep_all/"

out_folder="/cwork/rck32/eic/work_eic/slurm/output/output${current_date}"
error_folder="/cwork/rck32/eic/work_eic/slurm/error/error${current_date}"

rootname="vary_th_file_"
processdir="/cwork/rck32/eic/epic_klm/"
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
    file="${workdir}/slurm/shells_vary_theta/${rootname}${i}.sh"
    touch $file
    content="#!/bin/bash\n" 
    content+="#SBATCH --chdir=/cwork/rck32/eic/epic_klm\n"
    content+="#SBATCH --job-name=${rootname}${i}\n"
    content+="#SBATCH --output=${out_folder}/%x_mu.out\n"
    content+="#SBATCH --error=${error_folder}/%x_mu.err\n"
    content+="#SBATCH -p common\n"
    content+="#SBATCH --account=vossenlab\n"
    content+="#SBATCH --cpus-per-task=1\n"
    content+="#SBATCH --mem=8G\n"
    content+="#SBATCH --mail-user=rck32@duke.edu\n"
    content+="echo began job\n"
    content+="cat << EOF | /cwork/rck32/eic/eic-shell\n"
    content+="source install/setup.sh\n"
    content+="/usr/local/bin/ddsim --steeringFile ../work_eic/steering/variation.py --compactFile /cwork/rck32/eic/epic_klm/epic_klmws_only.xml -G -N 200 --gun.particle \"mu-\" --outputFile ../work_eic/root_files/July_21/full_sector_optph/vary_p_theta_optph_200events_file_${i}.edm4hep.root --part.userParticleHandler=\"\"\n"
    content+="EOF\n"
    echo -e "$content" > $file 
    echo "sbatch shells_vary_theta/${rootname}${i}.sh" >> $runJobs
    i=$((i+1))
done
