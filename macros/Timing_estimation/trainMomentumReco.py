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
from torch import nn

from momentum_prediction_util import Predictor,split_data,train,calculate_metrics
import argparse
parser = argparse.ArgumentParser(description = 'Preparing data for momentum prediction training')

parser.add_argument('--inputTensorPath', type=str, default="NA",
                        help='directory of output tensors') 
parser.add_argument('--outputTensorPath', type=str, default="NA",
                        help='directory of output tensors') 
args = parser.parse_args()
inputTensorPath = args.inputTensorPath
outputTensorPath = args.outputTensorPath
pref_TE = "/hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/"



### TODO: load tensors from inputTensorPath and output, then almost done
inputs = torch.load("data/momentum_prediction_pulse/Oct_6_cut_bad_events/input_5k_n_slurm_0.pt",weights_only = True)
outputs = torch.load("data/momentum_prediction_pulse/Oct_6_cut_bad_events/output_5k_n_slurm_0.pt",weights_only = True)
for i in range(1,51):
    new_input = torch.load(f"data/momentum_prediction_pulse/Oct_6_cut_bad_events/input_5k_n_slurm_{i}.pt",weights_only = True)
    inputs = torch.cat((inputs,new_input))
    new_output = torch.load(f"data/momentum_prediction_pulse/Oct_6_cut_bad_events/output_5k_n_slurm_{i}.pt",weights_only = True)
    outputs = torch.cat((outputs,new_output))

train_data, val_data, test_data, split_info = split_data(
    scaled_inputs, outputs
)

num_layers = 28
num_input_features_per_layer = 2
model = Predictor(input_size=num_layers * num_input_features_per_layer, num_classes=1, hidden_dim = num_layers * num_input_features_per_layer * 2, num_layers = 16)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)

loss_hist,val_hist = train(model,train_data['inputs'],train_data['outputs'],val_data['inputs'],val_data['outputs'],optimizer,device,num_epochs = 200, batch_size = 256)

pref_loss = pref_TE + "plots/momentum_prediction/loss/"

fig,axs = plot.subplots(1,1)
axs.plot(loss_hist,label = "Train")
axs.plot(val_hist,label = "Validation")
axs.set_ylabel("Loss")
fig.suptitle("Training Loss")
axs.set_xlable("epoch")
axs.legend()
fig.savefig(pref_loss + "test_loss.pdf")
pref_model = pref_TE + "models/"
model.save(pref_model + "Momentum_prediction/test.pth")

dataset = test_data
model_out = np.zeros(len(dataset['outputs']))
for i in range(len(model_out)):
    model_out[i] = model(dataset['inputs'][i].flatten().to(device)).detach().cpu()
    
num_bins = 20
bin_edges = np.linspace(0,10,num_bins + 1)

binned_model_out = [[] for _ in range(num_bins)]
binned_real_out = [[] for _ in range(num_bins)]
dataset = test_data
real_out = dataset['outputs']
model_out = np.zeros(len(real_out))
for i in range(len(model_out)):
    model_out[i] = model(dataset['inputs'][i].flatten().to(device)).detach().cpu()
    for j in range(1,len(bin_edges)):
        if(real_outs[i] < bin_edges[j]):
            binned_model_out[j - 1].append(model_out[i])
            binned_real_out[j - 1].append(real_out[i])
            break
binned_real_out = np.array(binned_real_out)
binned_model_out = np.array(binned_model_out)

bin_centers = np.array(bin_edges[1:]) - (bin_edges[1] - bin_edges[0]) / 2

RMSE_arr = np.zeros(len(bin_centers))
for i in range(len(bin_centers)):
    mse = np.mean((np.array(binned_real_out[i]) - np.array(binned_model_out[i])) ** 2)  # Mean Squared Error
    RMSE_arr[i] = np.sqrt(mse)  # Root Mean Squared Error

    

fig_RMSE,axs_RMSE = plot.subplots(1,1)

axs_RMSE.scatter(bin_centers[1:],RMSE_arr[1:])
axs_RMSE.xlabel("primary momentum")
axs_RMSE.ylabel("RMSE")
axs_RMSE.show();
pref_RMSE = pref_TE + "momentum_prediction/RMSE/"
fig_RMSE.savefig(pref_RMSE + "test.pdf")