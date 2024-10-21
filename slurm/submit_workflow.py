import subprocess
import os
from datetime import datetime

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def submit_simulation_and_processing_jobs(num_simulations,simulation_start_num, num_events):
    current_date = datetime.now().strftime("%B_%d")
    workdir = "/hpc/group/vossenlab/rck32/eic/work_eic"
    slurm_output = f"{workdir}/root_files/Slurm"
    out_folder = f"{workdir}/slurm/output/output{current_date}"
    error_folder = f"{workdir}/slurm/error/error{current_date}"
    root_file_dir = f"{workdir}/root_files/momentum_prediction/{current_date}"
    tensor_path_name = f"{workdir}/macros/Timing_estimation/data/momentum_prediction_pulse/{current_date}"

    create_directory(out_folder)
    create_directory(error_folder)
    create_directory(tensor_path_name + "/input")
    create_directory(tensor_path_name + "/output")
    create_directory(tensor_path_name)
    create_directory(root_file_dir)

    job_ids = []

    for i in range(simulation_start_num, simulation_start_num + num_simulations):
        shell_script = f"{workdir}/slurm/shells/prediction_sims_{current_date}_pim_run_1_{i}.sh"
        
        with open(shell_script, 'w') as f:
            f.write(f"""#!/bin/bash
#SBATCH --chdir=/hpc/group/vossenlab/rck32/eic/epic_klm
#SBATCH --job-name=prediction_sims_{current_date}_pim_run_1_{i}
#SBATCH --output={out_folder}/%x_mu.out
#SBATCH --error={error_folder}/%x_mu.err
#SBATCH -p scavenger-gpu
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --gpus=1
#SBATCH --mail-user=rck32@duke.edu

echo began job

cat << EOF | /hpc/group/vossenlab/rck32/eic/eic-shell
source install/setup.sh
/usr/local/bin/ddsim --steeringFile /hpc/group/vossenlab/rck32/eic/work_eic/steering/scint_sensitive/variation_scint.py --compactFile /hpc/group/vossenlab/rck32/eic/epic_klm/epic_klmws_only.xml -G -N {num_events} --gun.particle "pi-" --outputFile {root_file_dir}/pim_{num_events}events_0_8_to_10GeV_90theta_origin_file_{i}.edm4hep.root --part.userParticleHandler=""
EOF

echo finished ddsim

echo began postprocessing
source /hpc/group/vossenlab/rck32/ML_venv/bin/activate
python3 /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/process_data_for_momentum_NN.py --filePathName {root_file_dir}/pim_{num_events}events_0_8_to_10GeV_90theta_origin_file_{i}.edm4hep.root --inputTensorPathName {tensor_path_name}/input/pim_{num_events}events_{i}.pt --outputTensorPathName {tensor_path_name}/output/pim_{num_events}events_{i}.pt
""")

        # Submit the job and capture the job ID
        result = subprocess.run(['sbatch', shell_script], capture_output=True, text=True)
        job_id = result.stdout.strip().split()[-1]
        job_ids.append(job_id)

    return job_ids

def submit_training_job(dependency_job_ids):
    current_date = datetime.now().strftime("%B_%d")
    workdir = "/hpc/group/vossenlab/rck32/eic/work_eic"
    slurm_output = f"{workdir}/root_files/Slurm"
    out_folder = f"{workdir}/slurm/output/output{current_date}"
    error_folder = f"{workdir}/slurm/error/error{current_date}"
    root_file_dir = f"{workdir}/root_files/momentum_prediction/{current_date}"
    tensor_path_name = f"{workdir}/macros/Timing_estimation/data/momentum_prediction_pulse/{current_date}"
    
    dependency_string = f"afterok:{':'.join(dependency_job_ids)}"
    train_script = f"{workdir}/slurm/shells/train_predictor_{current_date}_pim_run_1.sh"

    with open(train_script, 'w') as f:
        f.write(f"""#!/bin/bash
#SBATCH --chdir=/hpc/group/vossenlab/rck32/eic/epic_klm
#SBATCH --job-name=train_predictor_{current_date}_pim_run_1
#SBATCH --output={out_folder}/%x_mu.out
#SBATCH --error={error_folder}/%x_mu.err
#SBATCH --dependency={dependency_string}
#SBATCH -p scavenger-gpu
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --gpus=1
#SBATCH --mail-user=rck32@duke.edu

echo began job
echo began training NN for prediction
source /hpc/group/vossenlab/rck32/ML_venv/bin/activate
python3 /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/trainMomentumReco.py --inputTensorPath {tensor_path_name}/input/ --outputTensorPath {tensor_path_name}/output/
""")
    sbatch_command = [
        "sbatch",
        train_script
    ]
    subprocess.run(sbatch_command)

def main():
    num_simulations = 5
    simulation_start_num = 6
    num_events = 2000

    # Submit simulation and processing jobs
    job_ids = submit_simulation_and_processing_jobs(num_simulations,simulation_start_num, num_events)
    print(f"Submitted {num_simulations} simulation and processing jobs")

    # Submit training job
#     submit_training_job(job_ids)
#     print("Submitted training job with dependency on all simulation and processing jobs")

if __name__ == "__main__":
    main()
