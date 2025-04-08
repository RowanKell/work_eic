import numpy as np
import uproot as up
import os
from util import time_func,p_func
import normflows as nf
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#time_func(p,m,dx)

def checkdir(path):
    if not os.path.exists(path): 
        os.makedirs(path)
def avg_time(num,event_times):
    ten_sorted_times = sorted(event_times)[:num]
    return sum(ten_sorted_times) / len(ten_sorted_times)

def process_times(uproot_path,threshold = 10, multipleFiles = False):
    if(multipleFiles):
        times_arrays_list = []
        cells_arrays_list = []
        x_pos_arrays_list = []
        y_pos_arrays_list = []
        z_pos_arrays_list = []
        px_arrays_list = []
        py_arrays_list = []
        pz_arrays_list = []

        # Loop through all files in the directory
        for file_name in os.listdir(uproot_path):
            if file_name.endswith(".root"):  # Ensure we're only processing ROOT files
                file_path = os.path.join(uproot_path, file_name)

                # Open the ROOT file
                with up.open(file_path) as file:
                    # Open tree
                    tree = file["events"]

                    times_arrays_list.append(tree["HcalBarrelHits/HcalBarrelHits.time"].array(library="np"))
                    cells_arrays_list.append(tree["HcalBarrelHits/HcalBarrelHits.cellID"].array(library="np"))
                    x_pos_arrays_list.append(tree["HcalBarrelHits/HcalBarrelHits.position.x"].array(library="np"))
                    y_pos_arrays_list.append(tree["HcalBarrelHits/HcalBarrelHits.position.y"].array(library="np"))
                    z_pos_arrays_list.append(tree["HcalBarrelHits/HcalBarrelHits.position.z"].array(library="np"))
                    
                    px_arrays_list.append(tree["HcalBarrelHits/HcalBarrelHits.momentum.x"].array(library="np"))
                    py_arrays_list.append(tree["HcalBarrelHits/HcalBarrelHits.momentum.y"].array(library="np"))
                    pz_arrays_list.append(tree["HcalBarrelHits/HcalBarrelHits.momentum.z"].array(library="np"))

        # Combine arrays for each branch
        times = np.concatenate(times_arrays_list)
        cells = np.concatenate(cells_arrays_list)
        x_pos_branch = np.concatenate(x_pos_arrays_list)
        y_pos_branch = np.concatenate(y_pos_arrays_list)
        z_pos_branch = np.concatenate(z_pos_arrays_list)

        # Now combined_arrays contains the concatenated arrays for each branch across all files
    else:
        events = up.open(uproot_path)

        times = events["HcalBarrelHits/HcalBarrelHits.time"].array(library='np')
        cells = events["HcalBarrelHits/HcalBarrelHits.cellID"].array(library='np')
        x_pos_branch = events["HcalBarrelHits/HcalBarrelHits.position.x"].array(library='np')
        y_pos_branch = events["HcalBarrelHits/HcalBarrelHits.position.y"].array(library='np')
        z_pos_branch = events["HcalBarrelHits/HcalBarrelHits.position.z"].array(library='np')
        
    accepted_times = []
    second_lowest_list = []
    avg_accepted_times = []
    rel_accepted_times = []

    duplicates = 0
    total = 0
    total_cells = []

    skipped = 0
    num_list = []
    #First loop over events
    for event_num in range(len(cells)):

        #skip events with less than 2 photons
    #     if(times[event_num].shape[0] < threshold): continue

        #Keep track of which cell IDs are hit
        curr_list = []
        for photon_num in range(len(cells[event_num])):
            if(cells[event_num][photon_num] in curr_list):
                duplicates += 1
            else:
                curr_list.append(cells[event_num][photon_num])
            if(cells[event_num][photon_num] not in total_cells):
                total_cells.append(cells[event_num][photon_num])
            total += 1
        num_list.append(len(curr_list))
        #check if 2 unique pixels are hit
        if(len(curr_list) < threshold): 
            skipped += 1
            continue
        curr_min = min(times[event_num])
        accepted_times.append(curr_min)
        second_lowest_list.append(min([x for x in times[event_num] if x != curr_min]))
        avg_accepted_times.append(avg_time(threshold,times[event_num]))
    #     if(len(curr_list) == 15):
    #         print(f"event #{event_num}")
    print(f"total: {total} | duplicates: {duplicates} | ratio: {duplicates / total} | num unique cells hit: {len(total_cells)} | skipped: {skipped}")
    return accepted_times, second_lowest_list, avg_accepted_times

def get_all_times(uproot_path,threshold = 10, multipleFiles = False):
    if(multipleFiles):
        times_list = []

        # Loop through all files in the directory
        for file_name in os.listdir(uproot_path):
            if file_name.endswith(".root"):  # Ensure we're only processing ROOT files
                file_path = os.path.join(uproot_path, file_name)

                # Open the ROOT file
                with up.open(file_path) as file:
                    # Open tree
                    tree = file["events"]

                    times_list.append(tree["HcalBarrelHits/HcalBarrelHits.time"].array(library="np"))

        # Combine arrays for each branch
        times = np.concatenate(times_list)

        # Now combined_arrays contains the concatenated arrays for each branch across all files
    else:
        events = up.open(uproot_path)

        times = events["HcalBarrelHits/HcalBarrelHits.time"].array(library='np')
        
    #First loop over events
    flattened_times = np.concatenate(times)
#         for i in range(len(times[event_num])):
    return flattened_times
def get_compiled_NF_model(thickness = "2cm", useGPU = True):
    if(thickness == "1cm"):
        run_num = 7
        run_num_str = str(run_num)

        #NF Stuff

        K = 8 #num flows

        latent_size = 1 #dimension of PDF
        hidden_units = 256 #nodes in hidden layers
        hidden_layers = 26
        context_size = 3 #conditional variables for PDF
        num_context = 3

        K_str = str(K)
        batch_size= 2000
        hidden_units_str = str(hidden_units)
        hidden_layers_str = str(hidden_layers)
        batch_size_str = str(batch_size)
        flows = []
        for i in range(K):
            flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units, 
                                                                     num_context_channels=context_size)]
            flows += [nf.flows.LULinearPermute(latent_size)]

        # Set base distribution
        q0 = nf.distributions.DiagGaussian(1, trainable=False)

        # Construct flow model
        model = nf.ConditionalNormalizingFlow(q0, flows)

        model_path = "/hpc/group/vossenlab/rck32/NF_time_res_models/"
        if(useGPU):
            model.load(model_path + "run_" + run_num_str + "_" + str(num_context)+ "context_" +K_str +  "flows_" + hidden_layers_str+"hl_" + hidden_units_str+"hu_" + batch_size_str+"bs.pth")
        else:
            state_dict = torch.load(model_path + "run_" + run_num_str + "_" + str(num_context)+ "context_" +K_str +  "flows_" + hidden_layers_str+"hl_" + hidden_units_str+"hu_" + batch_size_str+"bs.pth",  map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            model.to(torch.device('cpu'))
        model_compiled = torch.compile(model,mode = "reduce-overhead").to(device)
    elif(thickness == "2cm"):
        run_num = 1
        run_num_str = str(run_num)

        #NF Stuff

        K = 8 #num flows

        latent_size = 1 #dimension of PDF
        hidden_layers = 26
        hidden_units = 256 #nodes in hidden layers
        context_size = 3 #conditional variables for PDF
        num_context = 3
        batch_size= 20000
        K_str = str(K)
        hidden_units_str = str(hidden_units)
        hidden_layers_str = str(hidden_layers)
        batch_size_str = str(batch_size)
        flows = []
        for i in range(K):
            flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units, 
                                                                     num_context_channels=context_size)]
            flows += [nf.flows.LULinearPermute(latent_size)]

        # Set base distribution
        q0 = nf.distributions.DiagGaussian(1, trainable=False)

        # Construct flow model
        model = nf.ConditionalNormalizingFlow(q0, flows)

        model_path = "/hpc/group/vossenlab/rck32/NF_time_res_models/thicker_2cm/"
        if(useGPU):
            model.load(model_path + "run_" + run_num_str + "_" + str(num_context)+ "context_" +K_str +  "flows_" + hidden_layers_str+"hl_" + hidden_units_str+"hu_" + batch_size_str+"bs_checkpoint_e13.pth")
        else:
            state_dict = torch.load(model_path + "run_" + run_num_str + "_" + str(num_context)+ "context_" +K_str +  "flows_" + hidden_layers_str+"hl_" + hidden_units_str+"hu_" + batch_size_str+"bs_checkpoint_e13.pth", map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            model.to(torch.device('cpu'))
        model_compiled = torch.compile(model,mode = "reduce-overhead").to(device)
    elif(thickness == "5.55cm"):
        run_num = 1
        run_num_str = str(run_num)

        #NF Stuff

        K = 8 #num flows

        latent_size = 1 #dimension of PDF
        hidden_layers = 26
        hidden_units = 256 #nodes in hidden layers
        context_size = 3 #conditional variables for PDF
        num_context = 3
        batch_size= 20000
        K_str = str(K)
        hidden_units_str = str(hidden_units)
        hidden_layers_str = str(hidden_layers)
        batch_size_str = str(batch_size)
        flows = []
        for i in range(K):
            flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units, 
                                                                     num_context_channels=context_size)]
            flows += [nf.flows.LULinearPermute(latent_size)]

        # Set base distribution
        q0 = nf.distributions.DiagGaussian(1, trainable=False)

        # Construct flow model
        model = nf.ConditionalNormalizingFlow(q0, flows)

        model_path = "/hpc/group/vossenlab/rck32/NF_time_res_models/thicker_5.55cm/"
        if(useGPU):
            model.load(model_path + "run_" + run_num_str + "_" + str(num_context)+ "context_" +K_str +  "flows_" + hidden_layers_str+"hl_" + hidden_units_str+"hu_" + batch_size_str+"bs.pth")
        else:
            state_dict = torch.load(model_path + "run_" + run_num_str + "_" + str(num_context)+ "context_" +K_str +  "flows_" + hidden_layers_str+"hl_" + hidden_units_str+"hu_" + batch_size_str+"bs.pth", map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            model.to(torch.device('cpu'))
        model_compiled = torch.compile(model,mode = "reduce-overhead").to(device)
    else:
        print("model not found")
    return model_compiled