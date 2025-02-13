import subprocess
import os
from datetime import datetime

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def submit_simulation_and_processing_jobs(num_simulations,simulation_start_num, num_events,run_name,geometry_type,hepmc_bool = 1):
    current_date = datetime.now().strftime("%B_%d")
    workdir = "/hpc/group/vossenlab/rck32/eic/work_eic"
    slurm_output = f"{workdir}/root_files/Slurm"
    out_folder = f"{workdir}/slurm/output/output{current_date}"
    error_folder = f"{workdir}/slurm/error/error{current_date}"
    root_file_dir = f"{workdir}/root_files/Clustering/{current_date}"
    
    sts_pref="/hpc/group/vossenlab/rck32/eic/work_eic/slurm/status_codes/"
    hepmc_file = "/hpc/group/vossenlab/rck32/eic/EVGEN/K_L_only.hepmc3"
    
    steeringFile = workdir + "/steering/scint_sensitive/sector.py"

    create_directory(out_folder)
    create_directory(error_folder)
    create_directory(root_file_dir)

    job_ids = []

    for i in range(simulation_start_num, simulation_start_num + num_simulations):
        shell_script = f"{workdir}/slurm/shells/prediction_sims_{current_date}_{run_name}_{i}.sh"
        
        with open(shell_script, 'w') as f:
            f.write(f"""#!/bin/bash
#SBATCH --chdir=/hpc/group/vossenlab/rck32/eic/epic_klm
#SBATCH --job-name={run_name}_{current_date}_{i}
#SBATCH --output={out_folder}/%x.out
#SBATCH --error={error_folder}/%x.err
#SBATCH -p scavenger-gpu
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --gpus=1
#SBATCH --mail-user=rck32@duke.edu
#SBATCH --mail-type=FAIL

echo began job


cat << EOF | /hpc/group/vossenlab/rck32/eic/eic-shell
source install/setup.sh

#run ddsim, capture status

#########   DDSIM    ##########
echo "Running ddsim with steeringFile input"
/usr/local/bin/ddsim  --compactFile /hpc/group/vossenlab/rck32/eic/epic_klm/epic_klmws_only.xml -G --numberOfEvents {num_events} --steeringFile {steeringFile} --outputFile {root_file_dir}/{run_name}_{num_events}_{i}.edm4hep.root  --part.userParticleHandler=""
echo "DDSIM completed successfully"
echo began process root file
#########   PROCESS  ##########
python3 /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/process_root_file.py --filePathName {root_file_dir}/{run_name}_{num_events}_{i}.edm4hep.root  --processedDataPath /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/data/processed_data/{run_name}_{i}.json --geometryType {geometry_type}
EOF

echo "Beginning Analysis with analyze_data_old.py"    
source /hpc/group/vossenlab/rck32/ML_venv/bin/activate

#########   ANALYZE    ##########
python3 /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/analyze_data.py --inputProcessedData /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/data/processed_data/{run_name}_{i}.json --outputDataframePathName /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/data/df/{run_name}_{i}.csv --useCFD --batchSize 10000

deactivate
echo ENDING JOB
""")
            
        # Submit the job and capture the job ID
        result = subprocess.run(['sbatch', shell_script], capture_output=True, text=True)
        job_id = result.stdout.strip().split()[-1]
        job_ids.append(job_id)

    return job_ids

def submit_training_job(dependency_job_ids,run_name,run_num,num_simulations,use_dependency):
    current_date = datetime.now().strftime("%B_%d")
    workdir = "/hpc/group/vossenlab/rck32/eic/work_eic"
    slurm_output = f"{workdir}/root_files/Slurm"
    out_folder = f"{workdir}/slurm/output/output{current_date}"
    error_folder = f"{workdir}/slurm/error/error{current_date}"
    df_dir = f"{workdir}/root_files/momentum_prediction/{current_date}"
    
    dependency_string = f"afterok:{':'.join(dependency_job_ids)}"
    train_script = f"{workdir}/slurm/shells/train_predictor_{current_date}_{run_name}.sh"
    Timing_path = "/hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/"
    if(use_dependency):
        dependency_directive = f"\n#SBATCH --dependency={dependency_string}"
    else:
        dependency_directive = ""
    with open(train_script, 'w') as f:
        f.write(f"""#!/bin/bash
#SBATCH --chdir=/hpc/group/vossenlab/rck32/eic/epic_klm
#SBATCH --job-name=train_predictor_{current_date}_{run_name}
#SBATCH --output={out_folder}/%x_mu.out
#SBATCH --error={error_folder}/%x_mu.err{dependency_directive}
#SBATCH -p vossenlab-gpu
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --gpus=1
#SBATCH --mail-user=rck32@duke.edu
#SBATCH --mail-type=END

echo began job
echo began training NN for prediction
source /hpc/group/vossenlab/rck32/ML_venv/bin/activate
python3 /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/train_GNN.py --numDfs 200 --runNum {run_num} --inputDataPref "/hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/data/df/{run_name}_" --modelPath "/hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/models/{current_date}/run_{run_num}/" --numDfs {num_simulations} --resultsFilePath "/hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/results/" --framePlotPath "{Timing_path}plots/training_gif_frames/{current_date}_{run_num}/" --gifPlotPath "{Timing_path}plots/gifs/" --lossPlotPath "{Timing_path}plots/GNN_loss/" --testPlotPath "{Timing_path}plots/GNN_test/" --runName "{run_name}" --resultsPlotPath "{Timing_path}plots/GNN_results/"
""")
    sbatch_command = [
        "sbatch",
        train_script
    ]
    subprocess.run(sbatch_command)


def main():
    num_simulations = 200
    simulation_start_num = 400
    num_events = 50
    run_num = 1
    geometry_type = 1
    run_name = f"naive_CFD_Feb_10_{num_events}events_run_{run_num}"

    # Submit simulation and processing jobs
    job_ids = submit_simulation_and_processing_jobs(num_simulations,simulation_start_num, num_events,run_name,geometry_type)
    print(f"Submitted {num_simulations} simulation and processing jobs")
    #Submit training job
    use_dependency = True
#     job_ids = [""]
    submit_training_job(job_ids,run_name,run_num,num_simulations,use_dependency)
    print("Submitted training job with dependency on all simulation and processing jobs")


if __name__ == "__main__":
    main()

    