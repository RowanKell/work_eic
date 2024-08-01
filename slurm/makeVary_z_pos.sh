#!/bin/bash

calc_inc() {
    echo "$1 $2 $3" | awk '{ printf "%5f\n", ($2 - $1) / $3 }'
}

inc_z() {
    echo "$1 $2" | awk '{ printf "%5f\n", $1 + $2 }'
}

current_date=$(date +"%B_%d")
workdir="/cwork/rck32/eic/work_eic"

# hipodir="/lustre19/expphy/cache/clas12/rg-a/production/montecarlo/clasdis/fall2018/torus+1/v1/bkg50nA_10604MeV"
slurm_output="${workdir}/root_files/Slurm"

out_folder="/cwork/rck32/eic/work_eic/slurm/output/output${current_date}"
error_folder="/cwork/rck32/eic/work_eic/slurm/error/error${current_date}"

rootname="z_pos_vary_file_"
processdir="/cwork/rck32/eic/epic_klm/"
runJobs="${workdir}/slurm/runJobs.sh"
touch $runJobs
chmod +x $runJobs
echo " " > $runJobs
echo $daydir
i=0


if [ ! -d "$out_folder" ]; then
  mkdir -p "$out_folder"
fi

if [ ! -d "$error_folder" ]; then
  mkdir -p "$error_folder"
fi
z_pos=-732
z_end=767
num_z=20
z_inc=$(calc_inc $z_pos $z_end $num_z)

for num in $(seq 0 20)
do
    file="${workdir}/slurm/shells_vary_z/${rootname}${i}.sh"
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
    content+="/usr/local/bin/ddsim --steeringFile ../work_eic/steering/variation_z_pos.py --compactFile /cwork/rck32/eic/epic_klm/epic_klmws_only.xml -G -N 50000 --gun.particle \"mu-\" --outputFile ../work_eic/root_files/July_31/variation_z_pos/mu_run_1/varied_z_50kevents_${i}.edm4hep.root --part.userParticleHandler=\"\" --gun.position \"(1769.3, 0.0, ${z_pos})\"\n"
    content+="EOF\n"
    echo -e "$content" > $file 
    echo "sbatch shells_vary_z/${rootname}${i}.sh" >> $runJobs
    i=$((i+1))
    z_pos=$(inc_z $z_pos $z_inc)
done
