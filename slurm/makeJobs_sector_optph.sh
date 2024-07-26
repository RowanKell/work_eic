#!/bin/bash
current_date=$(date +"%B_%d")
workdir="/cwork/rck32/eic/work_eic"

# hipodir="/lustre19/expphy/cache/clas12/rg-a/production/montecarlo/clasdis/fall2018/torus+1/v1/bkg50nA_10604MeV"
slurm_output="${workdir}/root_files/Slurm"
#USER SET VALUES
output_dir="${workdir}/root_files/${current_date}/sector_sensor/run_1/"

out_folder="/cwork/rck32/eic/work_eic/slurm/output/output${current_date}"
error_folder="/cwork/rck32/eic/work_eic/slurm/error/error${current_date}"

rootname="sector_sensor_pi"
processdir="/cwork/rck32/eic/epic_klm/"
runJobs="${workdir}/slurm/runJobs.sh"
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

if [ ! -d "$output_dir" ]; then
  mkdir -p "$output_dir"
fi

for num in $(seq 0 20)
do
    file="${workdir}/slurm/shells_sector_sensor/${rootname}${i}.sh"
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
    content+="/usr/local/bin/ddsim --steeringFile ../work_eic/steering/variation.py --compactFile /cwork/rck32/eic/epic_klm/epic_klmws_only.xml -G -N 10 --gun.particle \"pi-\" --outputFile ${output_dir}pi_5GeV_theta_vary_10events${i}.edm4hep.root --part.userParticleHandler=\"\"\n"
    content+="EOF\n"
    echo -e "$content" > $file 
    echo "sbatch shells_sector_sensor/${rootname}${i}.sh" >> $runJobs
    i=$((i+1))
done
