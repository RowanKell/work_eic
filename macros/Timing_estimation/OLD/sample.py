# Import packages
import torch
import numpy as np
import normflows as nf

import uproot as up

from matplotlib import pyplot as plot
import math
from tqdm import tqdm
from util import PVect, theta_func, r_func,get_layer, create_layer_map
from IPython.display import clear_output
import time
from concurrent.futures import ThreadPoolExecutor
from reco import process_data, create_dataloader, prepare_data_for_nn, create_dataloader

# Get device to be used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import os
def checkdir(path):
    if not os.path.exists(path): 
        os.makedirs(path)
slurm = True

import argparse
parser = argparse.ArgumentParser(description = 'NF training')
parser.add_argument('--rootFile', type=str, default="NA",
                        help='input simulation file')
parser.add_argument('--outputFile', type=str, default="NA",
                        help='pt file to output samples')
parser.add_argument('--outputlayeridxs', type=str, default="NA",
                        help='pt file to output samples')
parser.add_argument('--useArgs', action=argparse.BooleanOptionalAction,
                        help='If True, uses argparse arguments')

args = parser.parse_args()


useArgs = args.useArgs

if(useArgs):
    rootFile = args.rootFile
    outputFile = args.outputFile
    outputlayeridxs = args.outputlayeridxs
else:
    rootFile = "/cwork/rck32/eic/work_eic/root_files/July_23/sector_scint/run_2_mu_5GeV_theta_vary_10kevents.edm4hep.root"
    outputFile = "/cwork/rck32/eic/work_eic/macros/Timing_estimation/data/samples/July_25/run_2_mu_5GeV_theta_vary_10kevents.pt"
    outputlayeridxs = "/cwork/rck32/eic/work_eic/macros/Timing_estimation/data/samples/July_25/run_2_mu_5GeV_theta_vary_10kevents_layers.pt"

# Define flows
run_num = 4
run_num_str = str(run_num)

K = 1

latent_size = 1
hidden_units = 100
hidden_layers = 6
context_size = 3
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

import datetime
if(slurm):
    pref = "/cwork/rck32/eic/work_eic/macros/Timing_estimation/slurm/"
else:
    pref = "/cwork/rck32/eic/work_eic/macros/Timing_estimation/"
x = datetime.datetime.now()
# today = x.strftime("%B_%d")
today = "July_25"
# model_path = "models/context_3/" + today + "/"
# checkdir(model_path)
model_path = pref + "models/" + "July_22" + "/"
checkdir(model_path)

samples_path = pref + "data/samples/" + today + "/"
checkdir(samples_path)

test_data_path = pref + "data/test/" + today + "/"
checkdir(test_data_path)

test_dist_path = pref + "plots/test_distributions/" + today + "/"
checkdir(test_dist_path)

model.load(model_path + "run_" + run_num_str + "_" + str(num_context)+ "context_" +K_str +  "flows_" + hidden_layers_str+"hl_" + hidden_units_str+"hu_" + batch_size_str+"bs.pth")

model = model.to(device)
up_path = rootFile + ":events"
data = process_data(up_path)

features, metadata = prepare_data_for_nn(data)
print("Features shape:", features.shape)
print("Metadata shape:", metadata.shape)

# Create DataLoader
batch_size = 1000000
dataloader = create_dataloader(features, metadata, batch_size)

min_time = 0
samples = torch.empty(features.shape[0],device = 'cpu')
layer_idxs = torch.empty(features.shape[0],device = 'cpu')
it = 0
for batch_features, batch_metadata in dataloader:
    print(f"beginning iteration #{it}")
    clear_output(wait=True)
    begin = batch_size * it
    end = batch_size * (it + 1)
    # Initialize a mask for valid samples
    context_features = torch.empty(batch_features.shape[0],3)
    context_features[:,0] = batch_features[:,0]
    context_features[:,1] = batch_features[:,1] * 180 / 3.14159
    context_features[:,2] = batch_features[:,2]
    context_features = context_features.to(device)
    samples[begin:end] = (model.sample(num_samples=context_features.shape[0], context=context_features)[0].cpu().detach() + batch_features[:,3].unsqueeze(1)).squeeze(1)
    layer_idxs[begin:end] = batch_metadata[:,2]
    it += 1
samples_cpu = samples.cpu()

torch.save(samples_cpu,outputFile)
torch.save(layer_idxs,outputlayeridxs)