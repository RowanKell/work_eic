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

def submit_simulation_and_processing_jobs_new(num_simulations,simulation_start_num, num_events,particle):
    particle_name = particle_name_dict[particle]
    current_date = datetime.now().strftime("%B_%d")
    workdir = "/hpc/group/vossenlab/rck32/eic/work_eic"
    slurm_output = f"{workdir}/root_files/Slurm"
    out_folder = f"{workdir}/slurm/output/output{current_date}"
    error_folder = f"{workdir}/slurm/error/error{current_date}"
    root_file_dir = f"{workdir}/root_files/momentum_prediction/{current_date}"
    tensor_path_name = f"{workdir}/macros/Timing_estimation/data/momentum_prediction_pulse/{current_date}_{particle_name}"
    
#     hepmc_file = "/hpc/group/vossenlab/rck32/eic/EVGEN/ep_noradcor.10x100_q2_1_10_run001.hepmc"
#     hepmc_file = "/hpc/group/vossenlab/rck32/eic/EVGEN/ep_noradcor.10x100_q2_100_1000_run001.hepmc"
    hepmc_file = "/hpc/group/vossenlab/rck32/eic/EVGEN/ep_run001.hepmc3"

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
#SBATCH --job-name=prediction_sims_{current_date}_{particle_name}_run_1_{i}
#SBATCH --output={out_folder}/%x_mu.out
#SBATCH --error={error_folder}/%x_mu.err
#SBATCH -p vossenlab-gpu
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --gpus=1
#SBATCH --mail-user=rck32@duke.edu


echo began job
echo began simulation
cat << EOF | /hpc/group/vossenlab/rck32/eic/eic-shell
source install/setup.sh
echo began process root file

python3 /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/process_root_file.py --filePathName {root_file_dir}/hepmc_{num_events}events_dev_branch_file_{i}.edm4hep.root  --processedDataPath /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/data/processed_data/dev_branch_{num_events}events.csv
echo finished process_root_file
EOF
source /hpc/group/vossenlab/rck32/ML_venv/bin/activate

echo began analyze data
python3 /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/analyze_data.py --inputProcessedData /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/data/processed_data/dev_branch_{num_events}events.csv --outputDataframePathName /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/data/df/dev_branch_{num_events}events.csv
echo finished analyze data
deactivate

""")

        # Submit the job and capture the job ID
        result = subprocess.run(['sbatch', shell_script], capture_output=True, text=True)
        job_id = result.stdout.strip().split()[-1]
        job_ids.append(job_id)

    return job_ids

def submit_simulation_and_processing_jobs_old(num_simulations,simulation_start_num, num_events,particle):
    particle_name = particle_name_dict[particle]
    current_date = datetime.now().strftime("%B_%d")
    workdir = "/hpc/group/vossenlab/rck32/eic/work_eic"
    slurm_output = f"{workdir}/root_files/Slurm"
    out_folder = f"{workdir}/slurm/output/output{current_date}"
    error_folder = f"{workdir}/slurm/error/error{current_date}"
    root_file_dir = f"{workdir}/root_files/momentum_prediction/{current_date}"
    tensor_path_name = f"{workdir}/macros/Timing_estimation/data/momentum_prediction_pulse/{current_date}_{particle_name}"
    
#     hepmc_file = "/hpc/group/vossenlab/rck32/eic/EVGEN/ep_noradcor.10x100_q2_1_10_run001.hepmc"
#     hepmc_file = "/hpc/group/vossenlab/rck32/eic/EVGEN/ep_noradcor.10x100_q2_100_1000_run001.hepmc"
    hepmc_file = "/hpc/group/vossenlab/rck32/eic/EVGEN/ep_run001.hepmc3"

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
#SBATCH --job-name=prediction_sims_{current_date}_{particle_name}_old_{i}
#SBATCH --output={out_folder}/%x_mu.out
#SBATCH --error={error_folder}/%x_mu.err
#SBATCH -p vossenlab-gpu
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --gpus=1
#SBATCH --mail-user=rck32@duke.edu


echo began job
echo began simulation
cat << EOF | /hpc/group/vossenlab/rck32/eic/eic-shell
source install/setup.sh
echo began process root file

python3 /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/process_root_file_old.py --filePathName {root_file_dir}/hepmc_{num_events}events_dev_branch_file_{i}.edm4hep.root  --processedDataPath /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/data/processed_data/old_{num_events}events.csv
echo finished process_root_file
EOF
source /hpc/group/vossenlab/rck32/ML_venv/bin/activate

echo began analyze data
python3 /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/analyze_data_old.py --inputProcessedData /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/data/processed_data/old_{num_events}events.csv --outputDataframePathName /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/data/df/old_{num_events}events.csv
echo finished analyze data
deactivate

""")

        # Submit the job and capture the job ID
        result = subprocess.run(['sbatch', shell_script], capture_output=True, text=True)
        job_id = result.stdout.strip().split()[-1]
        job_ids.append(job_id)

    return job_ids


def main():
    num_simulations = 1
    simulation_start_num = 0
    num_events =20
    particle = "kaon0L"
    runInfo = "run_1_w_inner"

    # Submit simulation and processing jobs
#     job_ids = submit_simulation_and_processing_jobs_new(num_simulations,simulation_start_num, num_events,particle)
    job_ids = submit_simulation_and_processing_jobs_old(num_simulations,simulation_start_num, num_events,particle)
    print(f"Submitted {num_simulations} simulation and processing jobs")


if __name__ == "__main__":
    main()

    
'''
USE FOR GENERATING ROOT FILE

updated 11/1:
source install/setup.sh
/usr/local/bin/ddsim  --compactFile /hpc/group/vossenlab/rck32/eic/epic_klm/epic_klmws_w_inner.xml --numberOfEvents {num_events} --inputFiles {hepmc_file} --outputFile {root_file_dir}/hepmc_{num_events}events_dev_branch_file_{i}.edm4hep.root  --part.userParticleHandler=""


/usr/local/bin/ddsim  --compactFile /hpc/group/vossenlab/rck32/eic/epic_klm/epic_klmws_only.xml --numberOfEvents {num_events} --inputFiles {hepmc_file} --outputFile {root_file_dir}/hepmc_{num_events}events_test_file_{i}.edm4hep.root  --part.userParticleHandler=""

python3 /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/process_data.py --filePathName {root_file_dir}/hepmc_{num_events}events_test_file_{i}.edm4hep.root  --processedDataPath /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/data/processed_data/test.json
EOF

source /hpc/group/vossenlab/rck32/ML_venv/bin/activate
python3 /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/process_data_for_momentum_NN.py --inputProcessedData /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/data/processed_data/test.json --outputDataframePathName /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/data/df/test.csv
deactivate
'''

'''
/usr/local/bin/ddsim  --compactFile ~/eic/epic_klm/epic_klmws_only.xml --numberOfEvents 5 --inputFiles ~/eic/EVGEN/ep_noradcor.10x100_q2_100_1000_run001.hepmc --outputFile root_files/test/hepmc_test.edm4hep.root  --part.userParticleHandler="" --output.part VERBOSE --part.keepAllParticles True
'''

'''
../DD4hep/build/bin/ddsim  --compactFile ~/eic/epic_klm/epic_klmws_only.xml --numberOfEvents 5 --inputFiles ~/eic/EVGEN/ep_noradcor.10x100_q2_100_1000_run001.hepmc --outputFile root_files/test/hepmc_test.edm4hep.root  --part.userParticleHandler="" --output.part VERBOSE --part.keepAllParticles True
'''