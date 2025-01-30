from datetime import date
today = date.today().strftime("%b_%d")

# Import packages
import torch
import numpy as np
import normflows as nf

import uproot as up

from matplotlib import pyplot as plot
import math
from tqdm import tqdm
from NF_util import PVect, theta_func, r_func, checkdir,train_NF_timing
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
parser.add_argument('--load_data', action=argparse.BooleanOptionalAction,
                        help='If True, loads data from parallel .pt files')
    

args = parser.parse_args()
useArgs = args.useArgs
load_data = args.load_data
# Get device to be used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Timing_path = "/hpc/group/vossenlab/rck32/eic/work_eic/macros/NF_timing_modeling/"

if(load_data == True):
    raw_inputs = torch.load(f"{Timing_path}data/January_28/Run_1/Jan_27_Vary_p_theta_z_file_0.pt")
    for i in range(601):
        clear_output(wait=True)
        print(f"loaded file #{i+1} of 601")
        new_raw_input = torch.load(Timing_path + f"data/January_28/Run_1/Jan_27_Vary_p_theta_z_file_{i}.pt")
        new_input = new_raw_input[new_raw_input[:,4] < 100]
        raw_inputs = torch.cat((raw_inputs, new_input),0)
    inputs = raw_inputs
    indexes = torch.randperm(inputs.shape[0])
    dataset = inputs[indexes]
    train_frac = 0.1
    test_frac = 0.02
    val_frac = 0.02
    train_lim = int(np.floor(dataset.shape[0] * train_frac))
    test_lim = train_lim + int(np.floor(dataset.shape[0] * test_frac))
    val_lim = test_lim + int(np.floor(dataset.shape[0] * val_frac))
    train_data = dataset[:train_lim]
    test_data = dataset[train_lim:test_lim]
    val_data = dataset[test_lim:val_lim]

    checkdir(f"{Timing_path}data/combined/{today}/")

    torch.save(train_data,Timing_path + f"data/combined/{today}/ten_percent_train_600_z_pos_train.pt")
    torch.save(test_data,Timing_path + f"data/combined/{today}/ten_percent_train_600_z_pos_test.pt")
    torch.save(val_data,Timing_path + f"data/combined/{today}/ten_percent_train_600_z_pos_val.pt")
else:
    train_data = torch.load(Timing_path + f"data/combined/{today}/ten_percent_train_600_z_pos_train.pt")
    test_data = torch.load(Timing_path + f"data/combined/{today}/ten_percent_train_600_z_pos_train.pt")
    val_data = torch.load(Timing_path + f"data/combined/{today}/ten_percent_train_600_z_pos_train.pt")

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
    K = 8
    hidden_units = 256
    hidden_layers = 26
    batch_size = 20000
    lr = 1e-5
    num_epochs = 25



latent_size = 1
num_context = 3
#strings for file paths
lr_str = str(lr)
K_str = str(K)
hidden_units_str = str(hidden_units)
hidden_layers_str = str(hidden_layers)
batch_size_str = str(batch_size)
num_context_str = str(num_context)
run_num_str = str(run_num)

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

# model_path = "models/context_3/" + today + "/"
# checkdir(model_path)

model_path = "../models/" + today + "/"
checkdir(model_path)

loss_path = "../plots/loss/" + today + "/"
checkdir(loss_path)

test_data_path = "../data/test/" + today + "/"
checkdir(test_data_path)

run_info = "run_" + run_num_str+"_"+num_context_str+ "context_"+ K_str + "flows_" + hidden_layers_str + "hl_" + hidden_units_str + "hu_" + batch_size_str + "bs"

model, train_loss_hist, val_loss_hist = train_NF_timing(model,device, train_data, val_data, batch_size,num_context = num_context, num_epochs = num_epochs, max_upticks = 5, validation_frequency = 2000,lr = lr,model_path = model_path, run_info = run_info)

# Plot loss
plot.figure(figsize=(5, 5))
plot.scatter(range(len(train_loss_hist)),train_loss_hist, np.ones(len(train_loss_hist)) * 0.01, label='train loss', alpha = 1)
plot.scatter(np.linspace(0,len(train_loss_hist),len(val_loss_hist)),val_loss_hist, np.ones(len(val_loss_hist)) * 0.1, label='val loss', alpha = 1)
plot.legend()
plot.savefig( loss_path + run_info + ".jpeg")

torch.save(test_data, test_data_path + "full_test_data_run_" + run_num_str+ "_"+ K_str + "flows_" + hidden_layers_str + "hl_" + hidden_units_str + "hu_" + batch_size_str + "bs.pt")