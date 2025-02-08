import uproot
import numpy as np
import torch
from collections import defaultdict
from util import get_layer, theta_func,create_layer_map
from reco import calculate_num_pixels_z_dependence
import matplotlib.pyplot as plot
import time
from collections import defaultdict
# Get device to be used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os
def checkdir(path):
    if not os.path.exists(path): 
        os.makedirs(path)
from IPython.display import clear_output
from tqdm import tqdm
import normflows as nf
import datetime
import pathlib

from momentum_prediction_util import process_root_file,prepare_prediction_input,Predictor,train,prepare_prediction_input_pulse

from momentum_prediction_util import prepare_nn_input as old_prepare_nn_input

# import argparse
# parser = argparse.ArgumentParser(description = 'Preparing data for momentum prediction training')

# parser.add_argument('--filePathName', type=str, default="NA",
#                         help='directory of root file') 
# parser.add_argument('--inputTensorPathName', type=str, default="NA",
#                         help='directory of input tensors') 
# parser.add_argument('--outputTensorPathName', type=str, default="NA",
#                         help='directory of output tensors') 
# args = parser.parse_args()
# filePathName = args.filePathName
# inputTensorPathName = args.inputTensorPathName
# outputTensorPathName = args.outputTensorPathName

filePathName = "/hpc/group/vossenlab/rck32/eic/work_eic/root_files/momentum_prediction/October_17/pim_50events_0_8_to_10GeV_90theta_origin_file_0.edm4hep.root"

# Tensor_parent = str(pathlib.Path(outputTensorPathName).parent)

# file_name = f"n_5kevents_0_8_to_10GeV_90theta_origin_file_{file_num}.edm4hep.root"

layer_map, super_layer_map = create_layer_map()

x = datetime.datetime.now()
today = x.strftime("%B_%d")

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

# Move model on GPU if available
model = model.to(device)
# model_date = "August_03"
# today = "August_03"
# model_path = "models/" + model_date + "/"
# checkdir(model_path)

model_path = "/hpc/group/vossenlab/rck32/NF_time_res_models/"

# checkdir(Tensor_parent)

model.load(model_path + "run_" + run_num_str + "_" + str(num_context)+ "context_" +K_str +  "flows_" + hidden_layers_str+"hl_" + hidden_units_str+"hu_" + batch_size_str+"bs.pth")
model = model.to(device)
model_compile = torch.compile(model,mode = "reduce-overhead")
model_compile = model_compile.to(device)
print("Starting process_root_file")
processed_data = process_root_file(filePathName)
print("Finished running process_root_file")

def new_prepare_nn_input(processed_data, normalizing_flow, batch_size=1024, device='cuda'):
    nn_input = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    nn_output = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    all_context = []
    all_time_pixels = []
    all_metadata = []
    
    print("Processing data...")
    for event_idx, event_data in tqdm(processed_data.items()):
        for layer, layer_data in event_data.items():
            for particle_id, particle_data in layer_data.items():
                primary_momentum = particle_data["primary_momentum"].item()
                base_context = torch.tensor([particle_data['z_pos'], particle_data['theta'], particle_data['momentum']], 
                                            dtype=torch.float32)
                base_time_pixels = torch.tensor([particle_data['time'], particle_data['num_pixels']], 
                                                dtype=torch.float32)
                
                for SiPM_idx in range(2):
                    z_pos = 1500 - particle_data['z_pos'] if SiPM_idx == 1 else particle_data['z_pos']
                    context = base_context.clone()
                    context[0] = z_pos
                    
                    all_context.append(context.repeat(particle_data['num_pixels'], 1))
                    all_time_pixels.append(base_time_pixels.repeat(particle_data['num_pixels'], 1))
                    all_metadata.extend([(event_idx, layer, SiPM_idx, particle_id, primary_momentum)] * particle_data['num_pixels'])

    all_context = torch.cat(all_context)
    all_time_pixels = torch.cat(all_time_pixels)
    
    print("Sampling data...")
    sampled_data = []
    for i in tqdm(range(0, len(all_context), batch_size)):
        batch_context = all_context[i:i+batch_size].to(device)
        batch_time_pixels = all_time_pixels[i:i+batch_size]
        
        with torch.no_grad():
            samples = abs(normalizing_flow.sample(num_samples=len(batch_context), context=batch_context)[0]).squeeze(1)
        
        adjusted_times = samples.cpu() + batch_time_pixels[:, 0]
        sampled_data.extend(adjusted_times)

    print("Reorganizing data...")
    for (event, layer, SiPM, particle, momentum), sample in zip(all_metadata, sampled_data):
        nn_input[event][layer][SiPM].append(sample)
        nn_output[event][layer][SiPM].append(torch.tensor([momentum]))

    return nn_input, nn_output


from memory_profiler import memory_usage
import psutil

def measure_peak_memory(func, *args, **kwargs):
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # Initial memory in MB
    mem_usage = memory_usage((func, args, kwargs), interval=0.1, timeout=None, max_iterations=1)
    peak_memory = max(mem_usage) - initial_memory
    return peak_memory

print("Starting prepare_nn_input")
# Then use it like this:
old_peak = measure_peak_memory(old_prepare_nn_input, processed_data, model)
new_peak = measure_peak_memory(new_prepare_nn_input, processed_data, model)

print(f"Old function peak memory usage: {old_peak:.2f} MB")
print(f"New function peak memory usage: {new_peak:.2f} MB")

# nn_input, nn_output = prepare_nn_input(processed_data, model_compile,batch_size = 50000)

# print("Starting prepare_prediction_input")
# prediction_input, prediction_output= prepare_prediction_input_pulse(nn_input,nn_output)
# torch.save(prediction_input,inputTensorPathName)
# torch.save(prediction_output,outputTensorPathName)
print("finished study")
