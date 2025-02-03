# Import packages
import torch
import numpy as np
import normflows as nf

import uproot as up

from matplotlib import pyplot as plot
import math
from tqdm import tqdm
from NF_util import PVect, theta_func, r_func,process_data, create_dataloader, prepare_data_for_nn_one_segment, create_dataloader
from IPython.display import clear_output
import time
from concurrent.futures import ThreadPoolExecutor

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
parser.add_argument('--useArgs', action=argparse.BooleanOptionalAction,
                        help='If True, uses argparse arguments')

args = parser.parse_args()


useArgs = args.useArgs

checkpoint_string = "_checkpoint_e15"

if(useArgs):
    rootFile = args.rootFile
    outputFile = args.outputFile
    outputlayeridxs = args.outputlayeridxs
else:
    rootFile = "/cwork/rck32/eic/work_eic/root_files/July_23/sector_scint/run_2_mu_5GeV_theta_vary_10kevents.edm4hep.root"
    outputFile = "/cwork/rck32/eic/work_eic/macros/Timing_estimation/data/samples/July_25/run_2_mu_5GeV_theta_vary_10kevents.pt"

# Define flows
run_num = 1
run_num_str = str(run_num)

K = 8

latent_size = 1
hidden_units = 256
hidden_layers = 26
context_size = 3
K_str = str(K)
batch_size= 20000
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
pref = "/hpc/group/vossenlab/rck32/eic/work_eic/macros/NF_timing_modeling/"
x = datetime.datetime.now()
# today = x.strftime("%B_%d")
today = "July_25"
# model_path = "models/context_3/" + today + "/"
# checkdir(model_path)
model_path = pref + "models/" + "Jan_28" + "/"
checkdir(model_path)

samples_path = pref + "data/samples/" + today + "/"
checkdir(samples_path)

test_data_path = pref + "data/test/" + today + "/"
checkdir(test_data_path)

test_dist_path = pref + "plots/test_distributions/" + today + "/"
checkdir(test_dist_path)

model.load(model_path + "run_" + run_num_str + "_" + str(context_size)+ "context_" +K_str +  "flows_" + hidden_layers_str+"hl_" + hidden_units_str+"hu_" + batch_size_str+f"bs{checkpoint_string}.pth")

model = model.to(device)
up_path = rootFile + ":events"
data = process_data(up_path)

features, metadata = prepare_data_for_nn_one_segment(data)

# Create DataLoader
eval_batch_size = 20000
eval_max_iter = features.shape[0] // eval_batch_size

model = model.to("cuda")
features = features.to("cuda")
model.eval()
sampled_data = torch.empty(len(features))
for i in tqdm(range(eval_max_iter)):
    begin = eval_batch_size * i
    end = eval_batch_size * (i + 1)
    with torch.no_grad():
        samples = abs(model.sample(num_samples = eval_batch_size, context = features[begin:end,:num_context])[0]).squeeze(1)
    sampled_data[begin:end] = samples.cpu() + features[begin:end,3].cpu()

torch.save(sampled_data,outputFile)