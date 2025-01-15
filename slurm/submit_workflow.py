import subprocess
import os
from datetime import datetime

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
particle_name_dict = {
    "pi-" : "pim",
    "neutron" : "n",
    "kaon0L" : "K_L"
}

def submit_simulation_and_processing_jobs(num_simulations,simulation_start_num, num_events,particle,hepmc_bool = 1):
    particle_name = particle_name_dict[particle]
    current_date = datetime.now().strftime("%B_%d")
    workdir = "/hpc/group/vossenlab/rck32/eic/work_eic"
    slurm_output = f"{workdir}/root_files/Slurm"
    out_folder = f"{workdir}/slurm/output/output{current_date}"
    error_folder = f"{workdir}/slurm/error/error{current_date}"
    root_file_dir = f"{workdir}/root_files/Clustering/{current_date}"
    tensor_path_name = f"{workdir}/macros/Timing_estimation/data/momentum_prediction_pulse/{current_date}_{particle_name}"
    run_name = f"newer_analyze_{num_events}events"
    
    sts_pref="/hpc/group/vossenlab/rck32/eic/work_eic/slurm/status_codes/"
    hepmc_file = "/hpc/group/vossenlab/rck32/eic/EVGEN/K_L_only.hepmc3"
    
    steeringFile = workdir + "/steering/scint_sensitive/sector.py"

    create_directory(out_folder)
    create_directory(error_folder)
    create_directory(tensor_path_name + "/input")
    create_directory(tensor_path_name + "/output")
    create_directory(tensor_path_name)
    create_directory(root_file_dir)

    job_ids = []

    for i in range(simulation_start_num, simulation_start_num + num_simulations):
        shell_script = f"{workdir}/slurm/shells/prediction_sims_{current_date}_{particle_name}_run_1_{i}.sh"
        
        with open(shell_script, 'w') as f:
            f.write(f"""#!/bin/bash
#SBATCH --chdir=/hpc/group/vossenlab/rck32/eic/epic_klm
#SBATCH --job-name=original_analyze_{current_date}_{particle_name}_{i}
#SBATCH --output={out_folder}/%x.out
#SBATCH --error={error_folder}/%x.err
#SBATCH -p vossenlab-gpu
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --gpus=1
#SBATCH --mail-user=rck32@duke.edu

echo began job
touch {sts_pref}ddsim_status{run_name}.txt
touch {sts_pref}process_status{run_name}.txt
touch {sts_pref}analyze_status{run_name}.txt

cat << EOF | /hpc/group/vossenlab/rck32/eic/eic-shell
source install/setup.sh

#run ddsim, capture status

#########   DDSIM    ##########
if [ {hepmc_bool} -eq 0 ]; then
    echo "Running ddsim file from hepmc"
    /usr/local/bin/ddsim  --compactFile /hpc/group/vossenlab/rck32/eic/epic_klm/epic_klmws_only.xml --numberOfEvents {num_events} --inputFiles {hepmc_file} --outputFile {root_file_dir}/{run_name}_{num_events}.edm4hep.root  --part.userParticleHandler=""
    echo $? > {sts_pref}ddsim_status{run_name}.txt
else
    echo "Running ddsim with steeringFile input"
    /usr/local/bin/ddsim  --compactFile /hpc/group/vossenlab/rck32/eic/epic_klm/epic_klmws_only.xml -G --numberOfEvents {num_events} --steeringFile {steeringFile} --outputFile {root_file_dir}/{run_name}_{num_events}.edm4hep.root  --part.userParticleHandler=""
    echo $? > {sts_pref}ddsim_status{run_name}.txt
fi
echo ddsim_status: $(cat {sts_pref}ddsim_status{run_name}.txt)

if [ $(cat {sts_pref}ddsim_status{run_name}.txt) -eq 0 ]; then
    echo "DDSIM completed successfully"
    echo began process root file
    
    #run process script
    
    #########   PROCESS    ##########
    python3 /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/process_root_file_old.py --filePathName {root_file_dir}/{run_name}_{num_events}.edm4hep.root  --processedDataPath /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/data/processed_data/{run_name}.json
    echo $? > {sts_pref}process_status{run_name}.txt
    
    if [ $(cat {sts_pref}process_status{run_name}.txt) -eq 0 ]; then
        echo "Sucessfully processed ROOT file with process_root_file_old.py"
    else
        echo "Processing failed with status $(cat {sts_pref}process_status{run_name}.txt)"
    fi
else
    echo "DDSIM failed with status $(cat {sts_pref}ddsim_status{run_name}.txt)"
    echo $(cat {sts_pref}ddsim_status{run_name}.txt) > {sts_pref}process_status{run_name}.txt
fi
EOF

if [ $(cat {sts_pref}process_status{run_name}.txt) -eq 0 ]; then
    echo "Beginning Analysis with analyze_data_old.py"
    
    source /hpc/group/vossenlab/rck32/ML_venv/bin/activate
    
    #########   ANALYZE OLD    ##########
    python3 /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/analyze_data_old.py --inputProcessedData /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/data/processed_data/{run_name}.json --outputDataframePathName /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/data/df/{run_name}.csv
    echo $? > {sts_pref}analyze_status{run_name}.txt
    deactivate
    
    if [ $(cat {sts_pref}analyze_status{run_name}.txt) -eq 0 ]; then
        echo "Successfully analyzed data with analyze_data_old.py"
    else
        echo "Analysis failed with status $(cat {sts_pref}analyze_status{run_name}.txt)"
    fi
else
    echo "DDSIM or PROCESS failed with status $(cat {sts_pref}process_status{run_name}.txt) "
fi
echo ENDING JOB
""")
            
        # Submit the job and capture the job ID
        result = subprocess.run(['sbatch', shell_script], capture_output=True, text=True)
        job_id = result.stdout.strip().split()[-1]
        job_ids.append(job_id)

    return job_ids
def submit_simulation_and_processing_jobs_test(num_simulations,simulation_start_num, num_events,particle,hepmc_bool = 1):
    particle_name = particle_name_dict[particle]
    current_date = datetime.now().strftime("%B_%d")
    workdir = "/hpc/group/vossenlab/rck32/eic/work_eic"
    slurm_output = f"{workdir}/root_files/Slurm"
    out_folder = f"{workdir}/slurm/output/output{current_date}"
    error_folder = f"{workdir}/slurm/error/error{current_date}"
    root_file_dir = f"{workdir}/root_files/Clustering/{current_date}"
    tensor_path_name = f"{workdir}/macros/Timing_estimation/data/momentum_prediction_pulse/{current_date}_{particle_name}"
    run_name = f"jan_13_old_analyze_{num_events}events"
    
    sts_pref="/hpc/group/vossenlab/rck32/eic/work_eic/slurm/status_codes/"
    hepmc_file = "/hpc/group/vossenlab/rck32/eic/EVGEN/K_L_only.hepmc3"
    
    steeringFile = workdir + "/steering/scint_sensitive/sector.py"

    create_directory(out_folder)
    create_directory(error_folder)
    create_directory(tensor_path_name + "/input")
    create_directory(tensor_path_name + "/output")
    create_directory(tensor_path_name)
    create_directory(root_file_dir)

    job_ids = []

    for i in range(simulation_start_num, simulation_start_num + num_simulations):
        shell_script = f"{workdir}/slurm/shells/prediction_sims_{current_date}_{particle_name}_run_1_{i}.sh"
        
        with open(shell_script, 'w') as f:
            f.write(f"""#!/bin/bash
#SBATCH --chdir=/hpc/group/vossenlab/rck32/eic/epic_klm
#SBATCH --job-name={run_name}_{current_date}_{i}
#SBATCH --output={out_folder}/%x.out
#SBATCH --error={error_folder}/%x.err
#SBATCH -p vossenlab-gpu
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --gpus=1
#SBATCH --mail-user=rck32@duke.edu

echo began job


cat << EOF | /hpc/group/vossenlab/rck32/eic/eic-shell
source install/setup.sh

#run ddsim, capture status

#########   DDSIM    ##########
echo "Running ddsim with steeringFile input"
/usr/local/bin/ddsim  --compactFile /hpc/group/vossenlab/rck32/eic/epic_klm/epic_klmws_only.xml -G --numberOfEvents {num_events} --steeringFile {steeringFile} --outputFile {root_file_dir}/{run_name}_{num_events}.edm4hep.root  --part.userParticleHandler=""
echo "DDSIM completed successfully"
echo began process root file
python3 /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/process_root_file_old.py --filePathName {root_file_dir}/{run_name}_{num_events}.edm4hep.root  --processedDataPath /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/data/processed_data/{run_name}.json
EOF

echo "Beginning Analysis with analyze_data_old.py"    
source /hpc/group/vossenlab/rck32/ML_venv/bin/activate

#########   ANALYZE OLD    ##########
python3 /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/analyze_data.py --inputProcessedData /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/data/processed_data/{run_name}.json --outputDataframePathName /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/data/df/{run_name}.csv

deactivate
echo ENDING JOB
""")
            
        # Submit the job and capture the job ID
        result = subprocess.run(['sbatch', shell_script], capture_output=True, text=True)
        job_id = result.stdout.strip().split()[-1]
        job_ids.append(job_id)

    return job_ids


def main():
    num_simulations = 1
    simulation_start_num = 0
    num_events =300
    particle = "kaon0L"

    # Submit simulation and processing jobs
    job_ids = submit_simulation_and_processing_jobs_test(num_simulations,simulation_start_num, num_events,particle)
    print(f"Submitted {num_simulations} simulation and processing jobs")


if __name__ == "__main__":
    main()

    