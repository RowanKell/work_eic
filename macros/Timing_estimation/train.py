from datetime import date
date = date.today().strftime("%b_%d")

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
import os
def checkdir(path):
    if not os.path.exists(path): 
        os.makedirs(path)
import argparse
parser = argparse.ArgumentParser(description = 'NF training')
parser.add_argument('--K', type=int, default=1,
                        help='# of flows')
parser.add_argument('--hu', type=int, default=100,
                        help='# hidden units')
parser.add_argument('--hl', type=int, default=6,
                        help='# hidden layers')
parser.add_argument('--bs', type=int, default=2000,
                        help='# datapoints per sample')
parser.add_argument('--run_num', type=int, default=0,
                        help='run number (of the day)')
parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate for optimizer')
parser.add_argument('--num_epochs', type=int, default=6,
                        help='# epochs to train')
parser.add_argument('--useArgs', action=argparse.BooleanOptionalAction,
                        help='If True, uses argparse arguments')
    

# Get device to be used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Timing_path = "/cwork/rck32/eic/work_eic/macros/Timing_estimation/"

train_data = torch.load(Timing_path + "data/combined/July_23/tenth_600_z_pos_train.pt")
test_data = torch.load(Timing_path + "data/combined/July_23/tenth_600_z_pos_test.pt")
val_data = torch.load(Timing_path + "data/combined/July_23/tenth_600_z_pos_val.pt")

args = parser.parse_args()
useArgs = args.useArgs
# Define flows
if(useArgs):
    run_num = args.run_num
    K = args.K
    hidden_units = args.hu
    hidden_layers = args.hl
    batch_size = args.bs
    lr = args.lr
    num_epochs = args.num_epochs
else:
    run_num = 1
    K = 2
    hidden_units = 64
    hidden_layers = 8
    batch_size = 2000
    lr = 1e-5
    num_epochs = 6



latent_size = 1
context_size = 3
num_context = 3
#strings for file paths
lr_str = str(lr)
K_str = str(K)
hidden_units_str = str(hidden_units)
hidden_layers_str = str(hidden_layers)
batch_size_str = str(batch_size)
num_context_str = str(num_context)
run_num_str = str(run_num)
# input_fig, input_axs = plot.subplots(1,3,figsize=(18,8))
# input_fig.suptitle("train.py training inputs")
# input_axs[0].hist(train_data[:100000,0],bins = 100)
# input_axs[0].set_title("hit z")
# input_axs[1].hist(train_data[:100000,1],bins = 100)
# input_axs[1].set_title("theta (deg)")
# input_axs[2].hist(train_data[:100000,2],bins = 100)
# input_axs[2].set_title("momentum")
# input_fig.savefig("/cwork/rck32/eic/work_eic/macros/Timing_estimation/plots/inputs/August_5/inputs train_" + run_num_str+  ".jpeg")
print("finished making inputs plot")
flows = []
for i in range(K):
    flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units, 
                                                             num_context_channels=3)]
    flows += [nf.flows.LULinearPermute(latent_size)]

# Set base distribution
q0 = nf.distributions.DiagGaussian(1, trainable=False)
    
# Construct flow model
model = nf.ConditionalNormalizingFlow(q0, flows)

# Move model on GPU if available
model = model.to(device)

import datetime

x = datetime.datetime.now()
today = x.strftime("%B_%d")
# model_path = "models/context_3/" + today + "/"
# checkdir(model_path)

model_path = "../models/" + today + "/"
checkdir(model_path)

loss_path = "../plots/loss/" + today + "/"
checkdir(loss_path)

test_data_path = "../data/test/" + today + "/"
checkdir(test_data_path)

run_info = "run_" + run_num_str+"_"+num_context_str+ "context_"+ K_str + "flows_" + hidden_layers_str + "hl_" + hidden_units_str + "hu_" + batch_size_str + "bs"

# Train model
num_context = 4
max_iter = int(np.floor(train_data.shape[0] / batch_size))
train_loss_hist = np.array([])
val_loss_hist = np.array([])
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

validation_frequency = 100  # Perform validation every 100 training steps
print("beginning training loop")
global_step = 0
for epoch in range(num_epochs):
    print(f"Beginning epoch #{epoch}")
    model.train()  # Set model to training mode
    for it in range(max_iter):
        optimizer.zero_grad()
        # Get training samples
        begin = it * batch_size
        end = (it + 1) * batch_size
        it_data = train_data[begin:end]
        context = torch.empty(it_data.size()[0], 3)
        context[:,0] = it_data[:,0]
        context[:,1] = it_data[:,1]
        context[:,2] = it_data[:,2]
        context = context.to(device)
        samples = (it_data[:,4] - it_data[:,3]).unsqueeze(1).to(device)
        # Compute loss
        loss = model.forward_kld(samples, context)
        # Do backprop and optimizer step
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer.step()
        # Log loss
        train_loss_hist = np.append(train_loss_hist, loss.to('cpu').data.numpy())

        global_step += 1

        # Validation step every 100 training steps
        if global_step % validation_frequency == 0:
            model.eval()  # Set model to evaluation mode
            val_loss = 0
            val_iter = min(100, int(np.floor(val_data.shape[0] / batch_size)))  # Limit validation to 100 batches
            with torch.no_grad():
                for val_it in range(val_iter):
                    begin = val_it * batch_size
                    end = (val_it + 1) * batch_size
                    it_data = val_data[begin:end]
                    context = torch.empty(it_data.size()[0], 3)
                    context[:,0] = it_data[:,0]
                    context[:,1] = it_data[:,1]
                    context[:,2] = it_data[:,2]
                    context = context.to(device)
                    samples = (it_data[:,4] - it_data[:,3]).unsqueeze(1).to(device)
                    loss = model.forward_kld(samples, context)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / val_iter
            val_loss_hist = np.append(val_loss_hist, avg_val_loss)
            
#             print(f"Step {global_step} - Train Loss: {train_loss_hist[-1]:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            model.train()  # Set model back to training mode
    model.save(model_path + run_info + f"_checkpoint_e{epoch}.pth")
    print(f"Epoch {epoch} completed.")
    
model.save(model_path + run_info + ".pth")
# Plot loss
plot.figure(figsize=(5, 5))
plot.scatter(range(len(train_loss_hist)),train_loss_hist, np.ones(len(train_loss_hist)) * 0.01, label='loss', alpha = 1)
plot.scatter(np.linspace(0,len(train_loss_hist),len(val_loss_hist)),val_loss_hist, np.ones(len(val_loss_hist)) * 0.1, label='loss', alpha = 1)
plot.legend()
plot.savefig( loss_path + run_info + ".jpeg")

torch.save(test_data, test_data_path + "full_test_data_run_" + run_num_str+ "_"+ K_str + "flows_" + hidden_layers_str + "hl_" + hidden_units_str + "hu_" + batch_size_str + "bs.pt")