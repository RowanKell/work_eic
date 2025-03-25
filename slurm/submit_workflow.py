import subprocess
import os
from datetime import datetime
import argparse
import time
from pathlib import Path

def create_directory(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except FileExistsError as e:
            print(f"Caught error while trying to create directory: {e}\n I think this is a concurrency issue where 2 jobs will pass the if statement at the same time")

def submit_simulation_and_processing_jobs(num_simulations,simulation_start_num, num_events,run_name,geometry_type,compactFile,setupPath,loadEpicCommand,chPath,particle,hepmc_bool = 1):
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
    shell_scripts = []
    errors = []
    outputs = []

    for i in range(simulation_start_num, simulation_start_num + num_simulations):
        shell_script = f"{workdir}/slurm/shells/prediction_sims_{current_date}_{run_name}_{i}.sh"
        
        with open(shell_script, 'w') as f:
            f.write(f"""#!/bin/bash
#SBATCH --chdir={chPath}
#SBATCH --job-name={run_name}_{current_date}_{i}
#SBATCH --output={out_folder}/%x.out
#SBATCH --error={error_folder}/%x.err
#SBATCH -p scavenger-gpu
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --gpus=1
#SBATCH --mail-user=rck32@duke.edu
#SBATCH --mail-type=FAIL

echo began job


cat << EOF | /hpc/group/vossenlab/rck32/eic/eic-shell
echo "compactFile: {compactFile}"
source {setupPath}
{loadEpicCommand}

#run ddsim, capture status

#########   DDSIM    ##########
echo "Running ddsim with steeringFile input"
/usr/local/bin/ddsim  --compactFile {compactFile} -G --numberOfEvents {num_events} --steeringFile {steeringFile} --outputFile {root_file_dir}/{run_name}_{num_events}_{i}.edm4hep.root  --part.userParticleHandler="" --gun.particle {particle}
echo "DDSIM completed successfully"
echo began process root file
#########   PROCESS  ##########
python3 /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/process_root_file.py --filePathName {root_file_dir}/{run_name}_{num_events}_{i}.edm4hep.root  --processedDataPath /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/data/processed_data/{run_name}_{i}.json --geometryType {geometry_type} --compactFile {compactFile} --deleteROOTFile
EOF

echo "Beginning Analysis with analyze_data_old.py"    
source /hpc/group/vossenlab/rck32/ML_venv/bin/activate

#########   ANALYZE    ##########
python3 /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/analyze_data.py --inputProcessedData /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/data/processed_data/{run_name}_{i}.json --outputDataframePathName /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/data/df/{run_name}_{i}.csv --useCFD --batchSize 10000 --deleteJSON

deactivate
echo ENDING JOB
""")
            
        # Submit the job and capture the job ID
        result = subprocess.run(['sbatch', shell_script], capture_output=True, text=True)
        job_id = result.stdout.strip().split()[-1]
        job_ids.append(job_id)
        shell_scripts.append(shell_script)
        errors.append(f"{error_folder}/{run_name}_{current_date}_{i}.err")
        outputs.append(f"{out_folder}/{run_name}_{current_date}_{i}.out")

    return job_ids,shell_scripts, errors, outputs

def submit_training_job(dependency_job_ids,run_name,run_num,use_dependency,num_dfs,outFile,deleteDfs,particle,save_gif):
    current_date = datetime.now().strftime("%B_%d")
    workdir = "/hpc/group/vossenlab/rck32/eic/work_eic"
    slurm_output = f"{workdir}/root_files/Slurm"
    out_folder = f"{workdir}/slurm/output/output{current_date}"
    error_folder = f"{workdir}/slurm/error/error{current_date}"
    df_dir = f"{workdir}/root_files/momentum_prediction/{current_date}"
    
    dependency_string = f"afterok:{':'.join(dependency_job_ids)}"
    train_script = f"{workdir}/slurm/shells/train_predictor_{current_date}_{run_name}.sh"
    Timing_path = "/hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/"
    if(save_gif):
        frame_gif_command = "--framePlotPath \"{Timing_path}plots/training_gif_frames/{current_date}_{run_num}/\" --gifPlotPath \"{Timing_path}plots/gifs/\""
    else:
        frame_gif_command = ""
    if(deleteDfs):
        deleteDfsString = "--deleteDfs"
    else:
        deleteDfsString = ""
    if(use_dependency):
        dependency_directive = f"\n#SBATCH --dependency={dependency_string}"
    else:
        dependency_directive = ""
    if(outFile == "NA"):
        outFile = "/hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/results/"
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
#SBATCH --mail-user=slurm_eicklm@outlook.com
#SBATCH --mail-type=END

echo began job
echo began training NN for prediction
source /hpc/group/vossenlab/rck32/ML_venv/bin/activate
python3 /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/train_GNN.py --numDfs {num_dfs} --runNum {run_num} --inputDataPref "/hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/data/df/{run_name}_" --modelPath "/hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/models/{current_date}/run_{run_num}/"  --resultsFilePath {outFile} {frame_gif_command} --lossPlotPath "{Timing_path}plots/GNN_loss/" --testPlotPath "{Timing_path}plots/GNN_test/" --runName "{run_name}" --resultsPlotPath "{Timing_path}plots/GNN_results/"  {deleteDfsString} --particle {particle}
""")
    sbatch_command = [
        "sbatch",
        train_script
    ]
    result = subprocess.run(sbatch_command, capture_output=True, text=True)
    job_id = result.stdout.strip().split()[-1]
    return job_id,train_script

def get_job_status(jobid):

    ### HERE: run bash command to retrieve status, exit code
    shellcommand = ["/hpc/group/vossenlab/rck32/eic/work_eic/slurm/util/checkSlurmStatus.sh", str(jobid)]
    commandout = subprocess.run(shellcommand,stdout=subprocess.PIPE)

    output = commandout.stdout.decode('utf-8')
    line_split = output.split()

    if len(line_split) == 1:
        status = line_split[0]
    else:
        #something wrong, try again
        print("Error in checking slurm status, assuming still running")
        return 0

    if status == "0": #Running
        return 0
    elif status == "1": #Completed
        return 1
    elif status == "-1": #Failed
        return -1

    return 0
    
def main():
    current_date = datetime.now().strftime("%B_%d")
    parser = argparse.ArgumentParser(description = 'Training GNN to predict KLM momentum')

    parser.add_argument('--run_name_pref', type=str, default="NA",
                        help='') 
    parser.add_argument('--outFile', type=str, default="NA")
    parser.add_argument('--compactFile', type=str, default="/hpc/group/vossenlab/rck32/eic/epic_klm/epic_klmws_only.xml")
    parser.add_argument('--runNum', type=int, default=-1)
    parser.add_argument("--waitForFinish",action=argparse.BooleanOptionalAction)
    parser.add_argument("--saveGif",action=argparse.BooleanOptionalAction)
    parser.add_argument("--deleteDfs",type =str, default ="False")
    parser.add_argument("--particle",type =str, default ="NA")
    parser.add_argument("--setupPath",type=str,default = "install/setup.sh")
    parser.add_argument("--loadEpicPath",type=str,default = "NA")
    parser.add_argument("--chPath",type=str,default = "/hpc/group/vossenlab/rck32/eic/epic_klm")
    args = parser.parse_args()
    num_simulations = 1
    simulation_start_num = 0
    num_events = 50
    if(args.runNum == -1):
        run_num = 1
    else:
        run_num = args.runNum
    if(args.particle == "NA"):
        particle = "neutron"
#         particle = "kaon0L"
    else:
        particle = args.particle
    geometry_type = 1
    if(args.run_name_pref == "NA"):
        run_name = f"Consistency_test_status_quo_geometry_{current_date}_{particle}_0_5GeV_to_5GeV{num_events}events_run_{run_num}"
    else:
        run_name = f"{args.run_name_pref}_{num_events}events_run_{run_num}"
#     run_name = f"naive_CFD_Feb_10_{num_events}events_run_{run_num}"
    if(args.deleteDfs == "False"):
        deleteDfs = False
    elif(args.deleteDfs == "True"):
        deleteDfs = True
    else:
        deleteDfs = False
        
        
    # Submit simulation and processing jobs
    if(args.loadEpicPath == "NA"):
        loadEpicCommand = ""
    else:
        loadEpicCommand = f"source {args.loadEpicPath}"
    job_ids,shell_scripts, shell_errors, shell_outputs = submit_simulation_and_processing_jobs(num_simulations,simulation_start_num, num_events,run_name,geometry_type,args.compactFile,args.setupPath,loadEpicCommand,args.chPath,particle)
    print(f"Submitted {num_simulations} simulation and processing jobs")
    #Submit training job
    use_dependency = True
#     job_ids = [""]
    num_dfs_total = num_simulations + simulation_start_num
    train_job_id,train_script = submit_training_job(job_ids,run_name,run_num,use_dependency,num_dfs_total, args.outFile,deleteDfs,particle,args.saveGif)
    print("Submitted training job with dependency on all simulation and processing jobs")
    if(args.waitForFinish):
        train_status = 0
        while(train_status == 0):
            train_status = get_job_status(train_job_id)
            if(train_status == 1):
                print("Train job succeeded")
            elif(train_status == -1):
                print("Train job failed")
                break
            elif(train_status == 0):
                print("Job running... sleeping for 30")
                time.sleep(30)
                continue
        for shell_script in shell_scripts:
            script_file = Path(shell_script)
            if(script_file.is_file()):
                script_file.unlink()
                print(f"deleted shell script file {shell_script}")

        for error_script in shell_errors:
            error_file = Path(error_script)
            if(error_file.is_file()):
                error_file.unlink()
                print(f"deleted error file {error_script}")

        for output_script in shell_outputs:
            output_file = Path(output_script)
            if(output_file.is_file()):
                output_file.unlink()
                print(f"deleted output file {output_script}")
        train_script_file = Path(train_script)
        if(train_script_file.is_file()):
            train_script_file.unlink()
            print(f"deleted shell script file {train_script}")



if __name__ == "__main__":
    main()