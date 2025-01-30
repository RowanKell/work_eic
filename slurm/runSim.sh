#!/bin/bash

run_num=1

particle="mu"
n_events=100

current_date=$(date +"%B_%d")
workdir="/cwork/rck32/eic/work_eic"

# hipodir="/lustre19/expphy/cache/clas12/rg-a/production/montecarlo/clasdis/fall2018/torus+1/v1/bkg50nA_10604MeV"
daydir="${workdir}/OutputFiles/${current_date}"
rootdir="${workdir}/root_files/${current_date}/sector_scint/"
#USER SET VALUES
outputdir="${daydir}/run_${run_num}/"

out_folder="/cwork/rck32/eic/work_eic/reco_slurm/output/output${current_date}"
error_folder="/cwork/rck32/eic/work_eic/reco_slurm/error/error${current_date}"

shellname="sim"
processdir="/cwork/rck32/eic/epic_klm/"
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

if [ ! -d "$rootdir" ]; then
  mkdir -p "$rootdir"
fi

rootname="${rootdir}run_${run_num}_${particle}_5GeV_full_theta_${n_events}events.edm4hep.root"
recoOutput="${outputdir}${particle}_5GeV_full_theta_reco_${n_events}events.pt"
outputlayer_path="${outputdir}${particle}_5GeV_full_theta_reco_${n_events}events_layers.pt"

file="${workdir}/reco_slurm/shells/${shellname}.sh"
touch $file
content="#!/bin/bash\n" 
content+="#SBATCH --chdir=/cwork/rck32/eic/epic_klm\n"
content+="#SBATCH --job-name=${shellname}${i}\n"
content+="#SBATCH --output=${out_folder}/%x_mu.out\n"
content+="#SBATCH --error=${error_folder}/%x_mu.err\n"
content+="#SBATCH -p vossenlab-gpu\n"
content+="#SBATCH --account=vossenlab\n"
content+="#SBATCH --cpus-per-task=1\n"
content+="#SBATCH --gpus=1\n"
content+="#SBATCH --mem=80G\n"
content+="#SBATCH --mail-user=rck32@duke.edu\n"
content+="echo began job\n"
content+="cat << EOF | /cwork/rck32/eic/eic-shell\n"
content+="source install/setup.sh\n"
content+="/usr/local/bin/ddsim --steeringFile ../work_eic/steering/variation_scint.py --compactFile /cwork/rck32/eic/epic_klm/epic_klmws_only.xml -G -N ${n_events} --gun.particle \"${particle}-\" --outputFile ${rootname} --part.userParticleHandler=\"\"\n"
content+="EOF\n"
content+="source /cwork/rck32/ML_venv/bin/activate\n"
content+="python3 ${workdir}/macros/Timing_estimation/sample.py --useArgs --rootFile ${rootname} --outputFile ${recoOutput} --outputlayeridxs ${outputlayer_path}\n"
contetn+="deactivate\n"
contetn+="source /cwork/reg_venv/bin/activate\n"
content+="python3 ${workdir}/macros/Timing_estimation/plot_timing_res.py"
echo -e "$content" > $file
sbatch $file

