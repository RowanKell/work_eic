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

from momentum_prediction_util import Predictor,split_data,train,calculate_metrics,load_and_concatenate_tensors,filter_tensors_by_values
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

inputs = load_and_concatenate_tensors(inputTensorPath)
outputs = load_and_concatenate_tensors(outputTensorPath)
filtered_inputs, filtered_outputs = filter_tensors_by_values(inputs,outputs)
train_data, val_data, test_data, split_info = split_data(
    filtered_inputs, filtered_outputs,train_ratio=0.1, val_ratio=0.45, test_ratio=0.45
)

num_layers = 28
num_input_features_per_layer = 2 * 2 #two sipms, 2 features (charge, time)
model = Predictor(input_size=num_layers * num_input_features_per_layer, num_classes=1, hidden_dim = num_layers * num_input_features_per_layer * 2, num_layers = 4)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-6, weight_decay=1e-5)

loss_hist,val_hist = train(model,train_data['inputs'],train_data['outputs'],val_data['inputs'],val_data['outputs'],optimizer,device,num_epochs = 200, batch_size = 128)

pref_loss = pref_TE + "plots/momentum_prediction/loss/"
print("plotting loss")
fig,axs = plot.subplots(1,1)
axs.plot(loss_hist,label = "Train")
axs.plot(val_hist,label = "Validation")
axs.set_ylabel("Loss")
fig.suptitle("Training Loss")
axs.set_xlabel("epoch")
axs.legend()
fig.savefig(pref_loss + "test_loss.pdf")
pref_model = pref_TE + "models/"
model.save(pref_model + "Momentum_prediction/test.pth")
print("running test data")
dataset = test_data
model_out = np.zeros(len(dataset['outputs']))
for i in range(len(model_out)):
    model_out[i] = model(dataset['inputs'][i].flatten().to(device)).detach().cpu()
    
real_out = dataset['outputs']

m_pi = np.ones(len(model_out)) * 0.139570 #pion mass in GeV

model_e = np.sqrt(np.square(model_out) + np.square(m_pi))
real_e = np.sqrt(np.square(real_out) + np.square(m_pi))

num_bins = 20
bin_edges = np.linspace(0.8,10,num_bins + 1)
print("binning test data")
binned_model_e = [[] for _ in range(num_bins)]
binned_real_e = [[] for _ in range(num_bins)]
for i in range(len(model_e)):
    for j in range(1,len(bin_edges)):
        if(real_e[i] < bin_edges[j]):
            binned_model_e[j - 1].append(model_e[i])
            binned_real_e[j - 1].append(real_e[i])
            break

bin_centers = np.array(bin_edges[1:]) - (bin_edges[1] - bin_edges[0]) / 2
print("calculating RMSE")
RMSE_arr = np.zeros(len(bin_centers))
rel_RMSE_arr = np.zeros(len(bin_centers))
for i in range(len(bin_centers)):
    mse = np.mean((np.array(binned_real_e[i]) - np.array(binned_model_e[i])) ** 2)  # Mean Squared Error
    RMSE_arr[i] = np.sqrt(mse)  # Root Mean Squared Error
    rel_RMSE_arr[i] = np.sqrt(mse) / bin_centers[i]  # Root Mean Squared Error

    
print("Plotting results")
fig_RMSE,axs_RMSE = plot.subplots(1,3,figsize=(15,6))
print(bin_centers)
print(RMSE_arr)
# print(real_e)
print(f"model_e: {model_e}")
print(f"model_out: {model_out}")
print(f"real_e: {real_e}")
axs_RMSE[0].scatter(bin_centers,RMSE_arr)
axs_RMSE[0].set_xlabel("Primary Energy")
axs_RMSE[0].set_ylabel("RMSE")

axs_RMSE[1].scatter(bin_centers,rel_RMSE_arr)
axs_RMSE[1].set_xlabel("Primary Energy")
axs_RMSE[1].set_ylabel("relative RMSE")

axs_RMSE[2].scatter(real_e,model_e, alpha = 0.1,color = "blue")
axs_RMSE[2].plot(range(10),color = "red");
axs_RMSE[2].set_ylabel("Learned Energy")
axs_RMSE[2].set_xlabel("Real Energy")
axs_RMSE[2].set_ylim(0,10)
axs_RMSE[2].set_xlim(0,10)

pref_RMSE = pref_TE + "plots/momentum_prediction/RMSE/"
fig_RMSE.tight_layout()
fig_RMSE.savefig(pref_RMSE + "test_0_8_dataset_new_params.pdf")

fig, axs = plot.subplots(1,1)
axs.scatter(bin_centers,rel_RMSE_arr)
axs.set_xlabel("Primary Energy")
axs.set_ylabel("Relative RMSE")
fig.suptitle(r"Relative RMSE vs Energy for $\pi^-$")
fig.tight_layout()
fig.savefig(pref_RMSE + "relRMSE_0_8_dataset_new_params.pdf")
