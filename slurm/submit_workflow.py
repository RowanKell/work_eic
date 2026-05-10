import subprocess
import os
from datetime import datetime
import argparse
import time
from pathlib import Path
from workflow_util import get_mem_limit, describe_geometry

# Set env variables as python variables for ease of use
try:
    workdir = os.environ['WORK_EIC']
    EIC_SHELL_HOME = os.environ['EIC_SHELL_HOME']
    ML_VENV_HOME = os.environ['ML_VENV_HOME']
    mail_user = os.environ['MAIL_USER']
    EPIC_HOME = os.environ['EPIC_HOME']
except KeyError as e:
    print(f"KeyError: {e}\nProbably need to source the setup script work_eic/setup.sh")
    exit(1)

def create_directory(directory):
    os.makedirs(directory, exist_ok=True)

# XML geometry readers and memory limit logic live in workflow_util.py

def submit_simulation_and_processing_jobs(num_simulations,simulation_start_num, num_events,run_name,geometry_type,compactFile,setupPath,loadEpicCommand,chPath,particle,useGPU,run_num,deleteROOTFile = True,deleteJSON = True,fastScint = False, useCFD = True, pixel_threshold = 3,hepmc_bool = 1, mem_limit = "8G"):
    current_date = datetime.now().strftime("%B_%d")
    slurm_output = f"{workdir}/root_files/Slurm"
    out_folder = f"{workdir}/slurm/output/output{current_date}"
    error_folder = f"{workdir}/slurm/error/error{current_date}"
    root_file_dir = f"{workdir}/root_files/Clustering/{current_date}"
    
    sts_pref=f"{workdir}/slurm/status_codes/"
    hepmc_file = f"{EIC_SHELL_HOME}/EVGEN/K_L_only.hepmc3"
    
    steeringFile = f"{workdir}/steering/scint_sensitive/sector.py"
    

    create_directory(out_folder)
    create_directory(error_folder)
    create_directory(root_file_dir)

    job_ids = []
    shell_scripts = []
    errors = []
    outputs = []
        
    if(useCFD):
        useCFDString = "--useCFD"
    else:
        useCFDString = "--no-useCFD"
        
    
    if(fastScint):
        scintThickness = "--scintThickness 2cm_1point8ns_time_constant_run_6"
    else:
        scintThickness = ""
    if(useGPU):
        useGPUString = "--useGPU"
        partition = "scavenger-gpu"
        request_gpu_string = "#SBATCH --gpus=1"
    else:
        useGPUString = "--no-useGPU"
        partition = "common"
        request_gpu_string = ""
    deleteROOTFileString = ''
    if(deleteROOTFile):
        deleteROOTFileString = '--deleteROOTFile'
    deleteJSONString = ''
    if(deleteJSON):
        deleteJSONString = "--deleteJSON"

    for i in range(simulation_start_num, simulation_start_num + num_simulations):
        shell_script = f"{workdir}/slurm/shells/prediction_sims_{current_date}_{run_name}_{i}.sh"
        
        with open(shell_script, 'w') as f:
            f.write(f"""#!/bin/bash
#SBATCH --chdir={chPath}
#SBATCH --job-name={run_name}_{current_date}_{i}
#SBATCH --output={out_folder}/%x.out
#SBATCH --error={error_folder}/%x.err
#SBATCH -p {partition}
#SBATCH --time=00:60:00
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
{request_gpu_string}
#SBATCH --mem={mem_limit}
#SBATCH --mail-user={mail_user}
#SBATCH --mail-type=FAIL
#SBATCH --exclude=dcc-gehmlab-gpu-ferc-s-z25-18
set -eo pipefail

echo began job
echo "=== GPU DIAGNOSTICS ==="
nvidia-smi
echo "SLURMD_NODENAME: $SLURMD_NODENAME"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "=== END GPU DIAGNOSTICS ==="

cat << EOF | {EIC_SHELL_HOME}/eic-shell
set -e
echo "compactFile: {compactFile}"
source {workdir}/setup.sh
source {setupPath}
{loadEpicCommand}

#run ddsim, capture status

#########   DDSIM    ##########
echo "Running ddsim with steeringFile input"

/usr/local/bin/ddsim  --compactFile {compactFile} -G --numberOfEvents {num_events} --steeringFile {steeringFile} --outputFile {root_file_dir}/{run_name}_{num_events}_{i}.edm4hep.root  --part.userParticleHandler="" --gun.particle {particle}
echo "DDSIM completed successfully"
echo began process root file
#########   PROCESS  ##########
python3 {workdir}/macros/Timing_estimation/process_root_file.py --filePathName {root_file_dir}/{run_name}_{num_events}_{i}.edm4hep.root  --processedDataPath {workdir}/macros/Timing_estimation/data/processed_data/{run_name}_{i}.json --geometryType {geometry_type} --compactFile {compactFile} {deleteROOTFileString}
EOF
echo "Beginning Analysis with analyze_data_old.py"    
source {ML_VENV_HOME}/bin/activate

#########   ANALYZE    ##########
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
python3 {workdir}/macros/Timing_estimation/analyze_data.py --inputProcessedData {workdir}/macros/Timing_estimation/data/processed_data/{run_name}_{i}.json --outputDataframePathName {workdir}/macros/Timing_estimation/data/df/{run_name}_{i}.csv {useCFDString} --batchSize 50000 {deleteJSONString} {useGPUString} {scintThickness} --pixelThreshold {pixel_threshold}

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

def submit_training_job(run_name,run_num,num_dfs,outFile,deleteDfs,particle,save_gif,lowEnergyObjectiveFlag, highEnergyObjectiveFlag):
    current_date = datetime.now().strftime("%B_%d")
    slurm_output = f"{workdir}/root_files/Slurm"
    out_folder = f"{workdir}/slurm/output/output{current_date}"
    error_folder = f"{workdir}/slurm/error/error{current_date}"
    df_dir = f"{workdir}/root_files/momentum_prediction/{current_date}"
    
    train_script = f"{workdir}/slurm/shells/train_predictor_{current_date}_{run_name}.sh"
    Timing_path = f"{workdir}/macros/Timing_estimation/"
    if(save_gif):
        frame_gif_command = "--framePlotPath \"{Timing_path}plots/training_gif_frames/{current_date}_{run_num}/\" --gifPlotPath \"{Timing_path}plots/gifs/\""
    else:
        frame_gif_command = ""
    if(deleteDfs):
        deleteDfsString = "--deleteDfs"
    else:
        deleteDfsString = ""
    if(highEnergyObjectiveFlag == 1):
        highEnergyObjective_string = "--writeHighEnergyObjective"
    elif(highEnergyObjectiveFlag == -1):
        highEnergyObjective_string = "--no-writeHighEnergyObjective"
    
    if(lowEnergyObjectiveFlag == 1):
        lowEnergyObjective_string = "--writeLowEnergyObjective"
    elif(lowEnergyObjectiveFlag == -1):
        lowEnergyObjective_string = "--no-writeLowEnergyObjective"
    with open(train_script, 'w') as f:
        f.write(f"""#!/bin/bash
#SBATCH --chdir={EPIC_HOME}
#SBATCH --job-name=train_predictor_{current_date}_{run_name}
#SBATCH --output={out_folder}/%x_mu.out
#SBATCH --error={error_folder}/%x_mu.err
#SBATCH -p scavenger-gpu
#SBATCH --time=00:45:00
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --mem=60G
#SBATCH --gpus=1
#SBATCH --mail-user={mail_user}
#SBATCH --mail-type=FAIL
#SBATCH --exclude=dcc-gehmlab-gpu-ferc-s-z25-18
set -e

echo began job
echo began training NN for prediction
source {ML_VENV_HOME}/bin/activate
python3 {workdir}/macros/Timing_estimation/train_GNN.py --numDfs {num_dfs} --runNum {run_num} --inputDataPref "{workdir}/macros/Timing_estimation/data/df/{run_name}_" --modelPath "{workdir}/macros/Timing_estimation/models/{current_date}/run_{run_num}/"  --resultsFilePath {outFile} {frame_gif_command} --lossPlotPath "{Timing_path}plots/GNN_loss/" --testPlotPath "{Timing_path}plots/GNN_test/" --runName "{run_name}" --resultsPlotPath "{Timing_path}plots/GNN_results/"  {deleteDfsString} --particle {particle} {highEnergyObjective_string} {lowEnergyObjective_string} 
""")
    sbatch_command = [
        "sbatch",
        train_script
    ]
    result = subprocess.run(sbatch_command, capture_output=True, text=True)
    job_id = result.stdout.strip().split()[-1]
    return job_id,train_script

def submit_classification_training_job(run_name_mu, run_name_pi, run_num, num_dfs, outFile, deleteDfs, testPlotPath=""):
    current_date = datetime.now().strftime("%B_%d")
    out_folder = f"{workdir}/slurm/output/output{current_date}"
    error_folder = f"{workdir}/slurm/error/error{current_date}"
    Timing_path = f"{workdir}/macros/Timing_estimation/"
    model_dir = f"{Timing_path}models/{current_date}/classifier_run_{run_num}/"

    inputDataPrefMu = f"{Timing_path}data/df/{run_name_mu}_"
    inputDataPrefPi = f"{Timing_path}data/df/{run_name_pi}_"

    deleteDfsString = "--deleteDfs" if deleteDfs else ""
    testPlotString = f"--testPlotPath \"{testPlotPath}\"" if testPlotPath else ""

    train_script = f"{workdir}/slurm/shells/train_classifier_{current_date}_{run_name_mu}.sh"
    with open(train_script, 'w') as f:
        f.write(f"""#!/bin/bash
#SBATCH --chdir={EPIC_HOME}
#SBATCH --job-name=train_classifier_{current_date}_{run_num}
#SBATCH --output={out_folder}/%x.out
#SBATCH --error={error_folder}/%x.err
#SBATCH -p scavenger-gpu
#SBATCH --time=00:45:00
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --gpus=1
#SBATCH --mail-user={mail_user}
#SBATCH --mail-type=FAIL
#SBATCH --exclude=dcc-gehmlab-gpu-ferc-s-z25-18
set -e

echo began classifier training job
source {ML_VENV_HOME}/bin/activate
python3 {Timing_path}train_GNN_classifier.py --inputDataPrefMu "{inputDataPrefMu}" --inputDataPrefPi "{inputDataPrefPi}" --numDfs {num_dfs} --resultsFilePath {outFile} --modelPath "{model_dir}" --runName "classifier_{run_num}" {deleteDfsString} {testPlotString}
""")
    result = subprocess.run(["sbatch", train_script], capture_output=True, text=True)
    job_id = result.stdout.strip().split()[-1]
    return job_id, train_script

def get_job_status(jobid):

    ### HERE: run bash command to retrieve status, exit code
    shellcommand = [f"{workdir}/slurm/util/checkSlurmStatus.sh", str(jobid)]
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
    
def write_failed_result(run_name, results_file_path):
    if(os.path.isdir(results_file_path)):
        results_write_path = f"{results_file_path}{run_name}.txt"
    else:
        results_write_path = f"{results_file_path}"
    with open(results_write_path, "w") as f:
        f.write(f"{-1}\n{-1}")
        print(f"writing RMSE as -1 to indicate failure: more than two data production jobs failed.")
    return
    
def main():
    current_date = datetime.now().strftime("%B_%d")
    parser = argparse.ArgumentParser(description = 'Training GNN to predict KLM momentum')

    parser.add_argument('--run_name_pref', type=str, default="NA",
                        help='') 
    parser.add_argument('--outFile', type=str, default="NA")
    parser.add_argument('--compactFile', type=str, default=f"{EPIC_HOME}/epic_klmws_only.xml")
    parser.add_argument('--runNum', type=int, default=-1)
    parser.add_argument("--saveGif",action=argparse.BooleanOptionalAction)
    parser.add_argument("--lowEnergyObjective",type=int, default = 1)
    parser.add_argument("--highEnergyObjective",type=int, default = 1)
    parser.add_argument("--deleteDfs",type =str, default ="False")
    parser.add_argument("--particle",type =str, default ="NA")
    parser.add_argument("--setupPath",type=str,default = "install/setup.sh")
    parser.add_argument("--loadEpicPath",type=str,default = "NA")
    parser.add_argument("--chPath",type=str,default = f"{EPIC_HOME}")
    parser.add_argument("--skipTraining",action=argparse.BooleanOptionalAction, default=False,
                        help='Skip training job submission, only run sim+process+analyze')
    parser.add_argument("--classification",action=argparse.BooleanOptionalAction, default=False,
                        help='Run mu-/pi+ classification workflow: produce data for both particles, then train GNN classifier')
    parser.add_argument("--num_simulations", type=int, default=None,
                        help='Override number of simulation jobs (default: 5 in debug mode, 50 otherwise)')
    parser.add_argument("--geo_config", type=str, default="basic", choices=["basic", "preshower", "linear_ratio"],
                        help='MOBO parameter space configuration: "basic" (num_layers × steel_ratio) '
                             '"preshower" (preshower_steel_value × division_layer_number), '
                             'or "linear_ratio" (steel_slope × scint_slope × and steel_ratio)'
                             'Selects the memory limit model used for sim/process SLURM jobs.')
    args = parser.parse_args()
    
    """
    USER DEFINED SETTINGS
    """
    
    debug_mode = True
    if(debug_mode):
        num_simulations = 50
        num_events = 500
        deleteROOTFile = False
        deleteJSON = False
        deleteShellsErrorsOutputs = False
    else:
        num_simulations = 20
        num_events = 500
        deleteROOTFile = True
        deleteJSON = True
        deleteShellsErrorsOutputs = True
    if args.num_simulations is not None:
        num_simulations = args.num_simulations
    simulation_start_num = 0
    useGPU = True
    fastScint = False
    pixel_threshold = 3
    useCFD = True
    
    """
    END SETTINGS
    """
    if(args.runNum == -1):
        run_num = 40
    else:
        run_num = args.runNum
    if(args.particle == "NA"):
#          particle = "proton"
         particle = "neutron"
#         particle = "kaon0L"
#         particle = "pi+"
        # particle = "mu-"
    else:
        particle = args.particle

    print(f"geo_config={args.geo_config}, {describe_geometry(args.compactFile, args.geo_config)}, "
          f"mem_limit for {particle}: {get_mem_limit(args.compactFile, particle, args.geo_config)}")

    if (useCFD):
        useCFD_filename = "CFD"
    else:
        useCFD_filename = "noCFD"
    geometry_type = 1
    if(args.run_name_pref == "NA"):
        run_name = f"baseline_{useCFD_filename}_pixel_threshold_{pixel_threshold}_{current_date}_{particle}_0_5GeV_to_5GeV{num_events}events_run_{run_num}"
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

    if(args.outFile == "NA"):
        outFile = f"{workdir}/macros/Timing_estimation/results/"
    else:
        outFile = args.outFile

    if(args.classification):
        # Classification workflow: produce data for mu- and pi+, then train classifier
        run_name_base = run_name.replace(f"_{particle}_", "_").replace(particle, "")
        if(args.run_name_pref == "NA"):
            run_name_mu = f"baseline_{useCFD_filename}_pixel_threshold_{pixel_threshold}_{current_date}_mu-_0_5GeV_to_5GeV{num_events}events_run_{run_num}"
            run_name_pi = f"baseline_{useCFD_filename}_pixel_threshold_{pixel_threshold}_{current_date}_pi+_0_5GeV_to_5GeV{num_events}events_run_{run_num}"
        else:
            run_name_mu = f"{args.run_name_pref}_mum_{num_events}events_run_{run_num}"
            run_name_pi = f"{args.run_name_pref}_pip_{num_events}events_run_{run_num}"

        # Submit mu- and pi+ sim jobs in parallel
        job_ids_mu, scripts_mu, errors_mu, outputs_mu = submit_simulation_and_processing_jobs(num_simulations, simulation_start_num, num_events, run_name_mu, geometry_type, args.compactFile, args.setupPath, loadEpicCommand, args.chPath, "mu-", useGPU, run_num, deleteROOTFile, deleteJSON, fastScint, useCFD, pixel_threshold, mem_limit=get_mem_limit(args.compactFile, "mu-", args.geo_config))
        print(f"Submitted {num_simulations} mu- simulation jobs")
        job_ids_pi, scripts_pi, errors_pi, outputs_pi = submit_simulation_and_processing_jobs(num_simulations, simulation_start_num, num_events, run_name_pi, geometry_type, args.compactFile, args.setupPath, loadEpicCommand, args.chPath, "pi+", useGPU, run_num, deleteROOTFile, deleteJSON, fastScint, useCFD, pixel_threshold, mem_limit=get_mem_limit(args.compactFile, "pi+", args.geo_config))
        print(f"Submitted {num_simulations} pi+ simulation jobs")

        all_job_ids = job_ids_mu + job_ids_pi
        all_shell_scripts = scripts_mu + scripts_pi
        all_shell_errors = errors_mu + errors_pi
        all_shell_outputs = outputs_mu + outputs_pi

        # Wait for all data jobs
        all_data_jobs_done = False
        while(all_data_jobs_done == False):
            all_jobs_succeeded = 1
            num_jobs_failed = 0
            for job_id in all_job_ids:
                job_status = get_job_status(job_id)
                if(job_status == -1):
                    num_jobs_failed += 1
                elif(job_status == 0):
                    all_jobs_succeeded = 0
            if(all_jobs_succeeded == 1):
                if(num_jobs_failed > 0):
                    write_failed_result(run_name_mu, outFile)
                    print("writing failed results...")
                    return
                all_data_jobs_done = True
            else:
                print("Data jobs running... sleeping for 30")
                time.sleep(30)

        # Submit classifier training
        num_dfs_total = num_simulations + simulation_start_num
        Timing_path = f"{workdir}/macros/Timing_estimation/"
        testPlotPath = f"{Timing_path}plots/classifier_roc/"
        train_job_id, train_script = submit_classification_training_job(run_name_mu, run_name_pi, run_num, num_dfs_total, outFile, deleteDfs, testPlotPath)
        print(f"Submitted classifier training job")

        train_status = 0
        while(train_status == 0):
            train_status = get_job_status(train_job_id)
            if(train_status == 1):
                print("Classifier training job succeeded")
            elif(train_status == -1):
                print("Classifier training job failed")
                break
            elif(train_status == 0):
                print("Classifier training job running... sleeping for 30")
                time.sleep(30)
                continue

        if(deleteShellsErrorsOutputs):
            for f in all_shell_scripts + all_shell_errors + all_shell_outputs + [train_script]:
                p = Path(f)
                if(p.is_file()):
                    p.unlink()
                    print(f"deleted {f}")

    else:
        # Standard single-particle workflow
        job_ids, shell_scripts, shell_errors, shell_outputs = submit_simulation_and_processing_jobs(num_simulations, simulation_start_num, num_events, run_name, geometry_type, args.compactFile, args.setupPath, loadEpicCommand, args.chPath, particle, useGPU, run_num, deleteROOTFile, deleteJSON, fastScint, useCFD, pixel_threshold, mem_limit=get_mem_limit(args.compactFile, particle, args.geo_config))
        print(f"Submitted {num_simulations} simulation and processing jobs")
        print("Submitted training job with dependency on all simulation and processing jobs")

        # Check for running jobs
        all_data_jobs_done = False
        while(all_data_jobs_done == False):
            all_jobs_succeeded = 1
            num_jobs_failed = 0
            for job_id in job_ids:
                job_status = get_job_status(job_id)
                if(job_status == -1):
                    num_jobs_failed += 1
                elif(job_status == 0):
                    all_jobs_succeeded = 0
            if(all_jobs_succeeded == 1):
                if(num_jobs_failed > 0):
                    write_failed_result(run_name, outFile)
                    print("writing failed results...")
                    return
                all_data_jobs_done = True
            else:
                print("Data jobs running... sleeping for 30")
                time.sleep(30)
        #Submit training job now that data jobs done
        if(not args.skipTraining):
            num_dfs_total = num_simulations + simulation_start_num
            train_job_id, train_script = submit_training_job(run_name, run_num, num_dfs_total, outFile, deleteDfs, particle, args.saveGif, args.lowEnergyObjective, args.highEnergyObjective)

            train_status = 0
            while(train_status == 0):
                train_status = get_job_status(train_job_id)
                if(train_status == 1):
                    print("Train job succeeded")
                elif(train_status == -1):
                    print("Train job failed")
                    break
                elif(train_status == 0):
                    print("Train job running... sleeping for 30")
                    time.sleep(30)
                    continue
        elif(args.skipTraining):
            print("Skipping training job (--skipTraining flag set)")
        if(deleteShellsErrorsOutputs):
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
            if(not args.skipTraining):
                train_script_file = Path(train_script)
                if(train_script_file.is_file()):
                    train_script_file.unlink()
                    print(f"deleted shell script file {train_script}")



if __name__ == "__main__":
    main()
