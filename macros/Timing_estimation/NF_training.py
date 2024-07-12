from datetime import date
date = date.today().strftime("%b_%d")
# date = "July_11"

# Import packages
import torch
import numpy as np
import normflows as nf

import uproot as up

from matplotlib import pyplot as plot
import math
from tqdm import tqdm
from util import PVect, theta_func, r_func, check_and_create_directory
from IPython.display import clear_output
import time
# Get device to be used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Timing_path = "/cwork/rck32/eic/work_eic/macros/Timing_estimation/"

inputs = torch.load(Timing_path + "data/July_05/Run_0/Vary_p_2000events_file_0_July_5_50_z_pos.pt")
for i in range(1,51):
    inputs = torch.cat((inputs, torch.load(Timing_path + f"data/July_05/Run_0/Vary_p_2000events_file_{i}_July_5_50_z_pos.pt")),0)

import argparse
parser = argparse.ArgumentParser(description = 'NF training')
parser.add_argument('--K', type=int, default=12,
                        help='# of flows')
parser.add_argument('--hu', type=int, default=64,
                        help='# hidden units')
parser.add_argument('--hl', type=int, default=8,
                        help='# hidden layers')
parser.add_argument('--bs', type=int, default=500,
                        help='# datapoints per sample')
parser.add_argument('--useArgs', action=argparse.BooleanOptionalAction,
                        help='If True, uses argparse arguments')
    
args = parser.parse_args()
useArgs = args.useArgs
# Define flows
if(useArgs):
    K = args.K
    hidden_units = args.hu
    hidden_layers = args.hl
    batch_size = args.bs
else:
    K = 12
    hidden_units = 64
    hidden_layers = 8
    batch_size = 2000
    
latent_size = 1
context_size = 4
num_context = 4
#strings for file paths
K_str = str(K)
hidden_units_str = str(hidden_units)
hidden_layers_str = str(hidden_layers)
batch_size_str = str(batch_size)
'''
fig, axs = plot.subplots(4,2,figsize = (12,20))
axs[0,0].hist(inputs[:,0],bins = 100);
axs[0,0].set_title("hit z pos")
axs[0,0].set_xlabel("mm")
axs[0,1].hist(inputs[:,1],bins = 100)
axs[0,1].set_title("mu hit incident time")
axs[0,1].set_xlabel("(ns)")

axs[1,0].hist(inputs[:,2],bins = 100);
axs[1,0].set_title("gun theta")
axs[1,0].set_xlabel("theta (degrees)")
axs[1,1].hist(inputs[:,4],bins = 100);
axs[1,1].set_title("photon hit time on sensor")
axs[1,1].set_xlabel("(ns)")

axs[2,0].hist2d(inputs[:,2],inputs[:,1],bins = 500);
axs[2,0].set_title("theta vs mu hit incident time")
axs[2,0].set_xlabel("theta")
axs[2,0].set_ylabel("mu hit incident time")
axs[2,1].hist2d(inputs[:,0], inputs[:,4],bins = 500);
axs[2,1].set_title("photon hit time on sensor vs z hit position")
axs[2,1].set_xlabel("z hit pos (mm)")
axs[2,1].set_ylabel("photon hit time on sensor (ns)")

axs[3,0].hist(inputs[:,3],bins = 500);
axs[3,0].set_title("mu momentum")
axs[3,0].set_xlabel("momentum (GeV/c)")

axs[3,1].hist2d(inputs[:,3], inputs[:,4],bins = 500);
axs[3,1].set_title("photon hit time on sensor vs mu momentum")
axs[3,1].set_xlabel("mu P (GeV/c)")
axs[3,1].set_ylabel("photon hit time on sensor (ns)")



fig.savefig(Timing_path + "plots/inputs/" + date + "/inputs_vary_p_uniform_" + K_str + "_flows_" + hidden_layers_str + "hl_" + hidden_units_str + "hu_" + batch_size_str + "bs.pdf")
'''
indexes = torch.randperm(inputs.shape[0])
dataset = inputs[indexes]
train_frac = 0.8
test_frac = 0.2
# val_frac = 0.1
train_lim = int(np.floor(dataset.shape[0] * train_frac))
test_lim = int(np.floor(dataset.shape[0] * test_frac)) + train_lim
train_data = dataset[:train_lim]
test_data = dataset[train_lim:test_lim]
# val_data = dataset[test_lim:]



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

# Train model

num_context = 4


max_iter = int(np.floor(train_data.shape[0] / batch_size))

loss_hist = np.array([])

#val stuff
# val_batch_size = int(np.floor(val_data.shape[0] / max_iter))
# val_hist = np.array([])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
start_time = time.time()
for it in range(max_iter):
#     if(it > 100):
#         break
    optimizer.zero_grad()
    
    # Get training samples
    begin = it * batch_size
    end = (it + 1) * batch_size
    it_data = train_data[begin:end]
    context = it_data[:,:num_context].to(device)
    samples = it_data[:,num_context].unsqueeze(1).to(device)
    
    # Compute loss
    loss = model.forward_kld(samples, context)
    
    # Do backprop and optimizer step
    if ~(torch.isnan(loss) | torch.isinf(loss)):
        loss.backward()
        optimizer.step()
    
    # Log loss
    loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
    if(it % int(max_iter // 100)):
        curr_time = time.time()
        elapsed_time = curr_time - start_time
        elapsed_minutes = elapsed_time // 60
        print(f"Finished iteration #{it} | {it / max_iter * 100}% complete, {elapsed_minutes} minutes running")
#     #val step
#     val_begin = it * val_batch_size
#     val_end = (it + 1) * val_batch_size
#     it_val_data = val_data[val_begin:val_end]
#     val_context = it_val_data[:,:num_context].to(device)
#     val_samples = it_data[:,num_context].unsqueeze(1).to(device)
#     model.eval()
#     val_loss = model.forward_kld(samples,context)
#     val_hist = np.append(val_hist, val_loss.to('cpu').data.numpy())
# Plot loss
loss_fig, loss_axs = plot.subplots(1,1, figsize=(5, 5))
loss_axs.plot(loss_hist, label='train loss')
# loss_axs.plot(val_hist, label = 'val loss', color = 'orange')
loss_axs.legend()
loss_dir = Timing_path + "plots/loss/" + date + "/"
check_and_create_directory(loss_dir)
loss_fig.savefig(loss_dir + "loss_vary_p_" + K_str + "_flows_" + hidden_layers_str + "hl_" + hidden_units_str + "hu_" + batch_size_str + "bs.pdf")

model_dir = Timing_path + "models/" + date + "/"
check_and_create_directory(model_dir)
model.save(model_dir + "vary_p_uniform_" + K_str + "_flows_" + hidden_layers_str + "hl_" + hidden_units_str + "hu_" + batch_size_str + "bs.pth")

test_data_times = test_data[:,num_context]

eval_batch_size = 1000
eval_max_iter = test_data.shape[0] // eval_batch_size
eval_test_data = test_data[:eval_max_iter * eval_batch_size]

eval_test_data.to('cpu')
model.to('cpu')
model.eval()
samples = torch.empty(eval_test_data.shape[0])
for i in range(eval_max_iter):
    begin = eval_batch_size * i
    end = eval_batch_size * (i + 1)
    samples[begin:end] = model.sample(num_samples = eval_batch_size, context = test_data[begin:end,:num_context])[0].cpu().detach().squeeze(1)
model.train();

samples_dir = Timing_path + "data/samples/ " + date +  "/"
check_and_create_directory(samples_dir)
test_dir = Timing_path + "data/test/ " + date + "/"
check_and_create_directory(test_dir)
torch.save(samples,samples_dir + "vary_p_uniform_" + K_str + "_flows_" + hidden_layers_str + "hl_" + hidden_units_str + "hu_" + batch_size_str + "bs.pt")
torch.save(eval_test_data,test_dir + "vary_p_uniform_" + K_str + "_flows_" + hidden_layers_str + "hl_" + hidden_units_str + "hu_" + batch_size_str + "bs.pt")

sample_fig, sample_axs = plot.subplots(1,1,figsize=(6,6))
sample_fig.suptitle("photon hit timing (p varied) " + hidden_layers_str + " hidden layers, " + K_str + " flows, " + hidden_units_str + " units, bs = " + batch_size_str)
sample_axs.hist(samples,bins = 300, alpha = 0.5,color = 'b', label = 'learned')
sample_axs.set_title("learned and true distributions")
sample_axs.set_xlabel("time (ns)")
sample_axs.set_ylabel("counts")
sample_axs.hist(eval_test_data[:,num_context],bins = 300, color = 'r', alpha = 0.5, label = 'true')
sample_axs.legend(loc='upper right')
sample_fig.show()
sample_fig_dir = Timing_path + "plots/test_distributions/" + date + "/"
check_and_create_directory(sample_fig_dir)
sample_fig.savefig(sample_fig_dir + "vary_p_uniform_" + K_str + "_flows_" + hidden_layers_str + "hl_" + hidden_units_str + "hu_" + batch_size_str + "bs.pdf")