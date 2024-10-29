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
def submit_simulation_and_processing_jobs(num_simulations,simulation_start_num, num_events,particle):
    particle_name = particle_name_dict[particle]
    current_date = datetime.now().strftime("%B_%d")
    workdir = "/hpc/group/vossenlab/rck32/eic/work_eic"
    slurm_output = f"{workdir}/root_files/Slurm"
    out_folder = f"{workdir}/slurm/output/output{current_date}"
    error_folder = f"{workdir}/slurm/error/error{current_date}"
    root_file_dir = f"{workdir}/root_files/momentum_prediction/{current_date}"
    tensor_path_name = f"{workdir}/macros/Timing_estimation/data/momentum_prediction_pulse/{current_date}_{particle_name}"
    
#     hepmc_file = "/hpc/group/vossenlab/rck32/eic/EVGEN/ep_noradcor.10x100_q2_1_10_run001.hepmc"
    hepmc_file = "/hpc/group/vossenlab/rck32/eic/EVGEN/ep_noradcor.10x100_q2_100_1000_run001.hepmc"

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
#SBATCH --mem=100G
#SBATCH --gpus=1
#SBATCH --mail-user=rck32@duke.edu


echo began job
echo began postprocessing
cat << EOF | /hpc/group/vossenlab/rck32/eic/eic-shell
source install/setup.sh
/usr/local/bin/ddsim  --compactFile /hpc/group/vossenlab/rck32/eic/epic_klm/epic_klmws_only.xml --numberOfEvents {num_events} --inputFiles {hepmc_file} --outputFile {root_file_dir}/hepmc_{num_events}events_test_file_{i}_with_particleHandler_keepALL.edm4hep.root  --part.userParticleHandler="" --output.part VERBOSE --part.keepAllParticles True
EOF
""")

        # Submit the job and capture the job ID
        result = subprocess.run(['sbatch', shell_script], capture_output=True, text=True)
        job_id = result.stdout.strip().split()[-1]
        job_ids.append(job_id)

    return job_ids

def submit_training_job(dependency_job_ids,particle,runInfo):
    particle_name = particle_name_dict[particle]
    current_date = datetime.now().strftime("%B_%d")
    workdir = "/hpc/group/vossenlab/rck32/eic/work_eic"
    slurm_output = f"{workdir}/root_files/Slurm"
    out_folder = f"{workdir}/slurm/output/output{current_date}"
    error_folder = f"{workdir}/slurm/error/error{current_date}"
    root_file_dir = f"{workdir}/root_files/momentum_prediction/{current_date}"
    tensor_path_name = f"{workdir}/macros/Timing_estimation/data/momentum_prediction_pulse/{current_date}_{particle_name}"
    
    dependency_string = f"afterok:{':'.join(dependency_job_ids)}"
    train_script = f"{workdir}/slurm/shells/train_predictor_{current_date}_{particle_name}_run_1.sh"

    with open(train_script, 'w') as f:
        f.write(f"""#!/bin/bash
#SBATCH --chdir=/hpc/group/vossenlab/rck32/eic/epic_klm
#SBATCH --job-name=train_predictor_{current_date}_{particle_name}_{runInfo}
#SBATCH --output={out_folder}/%x_mu.out
#SBATCH --error={error_folder}/%x_mu.err
#SBATCH --dependency={dependency_string}
#SBATCH -p scavenger-gpu
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --gpus=1
#SBATCH --mail-user=rck32@duke.edu

echo began job
echo began training NN for prediction
source /hpc/group/vossenlab/rck32/ML_venv/bin/activate
python3 /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/trainMomentumReco.py --inputTensorPath {tensor_path_name}/input/ --outputTensorPath {tensor_path_name}/output/ --plotPath {current_date}_{particle_name}_{runInfo} --modelPath {current_date}_{particle_name}_{runInfo} --particle {particle_name} --runInfo {runInfo}
""")
    sbatch_command = [
        "sbatch",
        train_script
    ]
    subprocess.run(sbatch_command)

def main():
    num_simulations = 1
    simulation_start_num = 0
    num_events = 5
    particle = "pi-"
#     particle = "kaon0L"
    runInfo = "run_1"

    # Submit simulation and processing jobs
    job_ids = submit_simulation_and_processing_jobs(num_simulations,simulation_start_num, num_events,particle)
    print(f"Submitted {num_simulations} simulation and processing jobs")

# #     Submit training job
#     submit_training_job(job_ids,particle,runInfo)
#     print("Submitted training job with dependency on all simulation and processing jobs")

if __name__ == "__main__":
    main()

    
'''
USE FOR GENERATING ROOT FILE

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