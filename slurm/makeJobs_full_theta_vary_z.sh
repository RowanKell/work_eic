#!/bin/bash
#Used to generate training data for conditional flow
convert_pos() {
    echo "import math; print(math.tan($1 - 1.57080))" | python | awk '{ printf "%.5f\n", 50 * $1 }'
}

calc_min_theta() {
    echo "import math; print(1.57080 - math.atan2(767 - $1,10))" | python | awk '{ printf "%.5f\n", $1 }'
}

calc_max_theta() {
    echo "import math; print(math.atan2($1 + 732,10) + 1.57080)" | python | awk '{ printf "%.5f\n", $1 }'
}
calc_events() {
    echo "import math;import numpy as np; print(int(np.floor(3500 *(1 -  (732 + $1) / 1500) + 500)))" | python | awk '{ printf "%d\n", $1 }'
}

calc_inc() {
    echo "$1 $2 $3" | awk '{ printf "%5f\n", ($2 - $1) / $3 }'
}

inc_z() {
    echo "$1 $2" | awk '{ printf "%5f\n", $1 + $2 }'
}

sign() {
    echo "import math; print(math.copysign(1, $1))" | python | awk '{ printf "%d\n", $1 }'
}

current_date=$(date +"%B_%d")
workdir="/cwork/rck32/eic/work_eic"

# hipodir="/lustre19/expphy/cache/clas12/rg-a/production/montecarlo/clasdis/fall2018/torus+1/v1/bkg50nA_10604MeV"
slurm_output="${workdir}/root_files/Slurm"
daydir="/cwork/rck32/eic_output/pi_sims/${current_date}"
#USER SET VALUES
outputdir="${daydir}/Run_0_vary_events/"

out_folder="/cwork/rck32/eic/work_eic/slurm/output/output${current_date}"
error_folder="/cwork/rck32/eic/work_eic/slurm/error/error${current_date}"

rootname="200_z_vals_file_"
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
z_pos=-732
z_end=767
num_z=200
z_inc=$(calc_inc $z_pos $z_end $num_z)
x_pos=1769.3

for num in $(seq 0 $num_z)
do
    if [ $num -eq 0 ]; then
        theta=$theta_start
    fi
    num_events=$(calc_events $z_pos)
    file="${workdir}/slurm/shells_full_theta_vary/${rootname}${i}.sh"
    touch $file
    theta_min=$(calc_min_theta $z_pos)
    theta_max=$(calc_max_theta $z_pos)
    content="#!/bin/bash\n" 
    content+="#SBATCH --chdir=/cwork/rck32/eic/epic_klm\n"
    content+="#SBATCH --job-name=${rootname}${i}\n"
    content+="#SBATCH --output=${out_folder}/%x_mu.out\n"
    content+="#SBATCH --error=${error_folder}/%x_mu.err\n"
    content+="#SBATCH -p common\n"
    content+="#SBATCH --account=vossenlab\n"
    content+="#SBATCH --cpus-per-task=1\n"
    content+="#SBATCH --mem=4G\n"
    content+="#SBATCH --mail-user=rck32@duke.edu\n"
    content+="echo began job\n"
    content+="cat << EOF | /cwork/rck32/eic/eic-shell\n"
    content+="source /cwork/rck32/eic/epic_klm/install/setup.sh\n"
    content+="/usr/local/bin/ddsim --steeringFile /cwork/rck32/eic/work_eic/steering/variation_pos.py --compactFile /cwork/rck32/eic/epic_klm/epic_klmws_only.xml -G -N ${num_events} --gun.particle \"mu-\" --outputFile /cwork/rck32/eic/work_eic/root_files/July_19/slurm/run_0_vary_events_one_segment_param/vary_p_z_th_events_${i}_200_z_vals.edm4hep.root --part.userParticleHandler=\"\" --gun.position \"(${x_pos}, 0.0, ${z_pos})\" --gun.thetaMin \"${theta_min}\" --gun.thetaMax \"${theta_max}\"\n"
    content+="EOF\n"
    echo -e "$content" > $file 
    echo "sbatch shells_full_theta_vary/${rootname}${i}.sh" >> $runJobs
    i=$((i+1))
    z_pos=$(inc_z $z_pos $z_inc)
done
