# Import packages
import torch
import numpy as np

import uproot as up

from matplotlib import pyplot as plot
import math
from tqdm import tqdm
from util import PVect, theta_func, r_func,get_layer, create_layer_map
import time
from concurrent.futures import ThreadPoolExecutor

from reco import process_data, create_dataloader, prepare_data_for_nn, create_dataloader,load_truth

# Get device to be used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import os
def checkdir(path):
    if not os.path.exists(path): 
        os.makedirs(path)

        
particle = "mu"   
        
     
'''
Load samples
'''
workdir = "/cwork/rck32/eic/work_eic/"
samples_path = workdir + f"OutputFiles/July_25/run_1/{particle}_full_theta_reco_100events.pt"
sample_layers_path = workdir+f"OutputFiles/July_25/run_1/{particle}_full_theta_reco_100events_layers.pt"
samples_cpu = torch.load(samples_path)
layer_idxs = torch.load(sample_layers_path)
    
'''
Load truth data
'''
        
truth_times = load_truth("/cwork/rck32/eic/work_eic/root_files/July_23/sector_sensor/")

print("loaded samples and truth")

# plot integrated distributions
comp_fig, comp_axs = plot.subplots(1,1,figsize = (8,6))
comp_fig.suptitle(f"truth vs learned optph times {particle}")
comp_axs.hist(samples_cpu[:500000],bins = 300,density = True,color = "blue", alpha = 0.4,label = "learned")
# comp_axs.set_title("learned photon hit times")
comp_axs.set_xlabel("sensor hit time (ns)")
comp_axs.set_ylabel("counts (normalized)")
comp_axs.hist(truth_times[:500000][:,0],bins = 300,density = True, color = "red", alpha = 0.4,label = "truth");
comp_axs.legend()

comp_fig.savefig(workdir + f"macros/Timing_estimation/plots/truth_comp/{particle}_test_file_july_25_run_1_full_theta.jpeg")

samples_short = samples_cpu[:500000]
layer_idxs_short = layer_idxs[:500000]

import matplotlib.ticker as plticker

layer_fig, layer_axs = plot.subplots(7,4,figsize=(16,20),sharex=True)
for i in range(7):
    for j in range(4):
        curr_layer = i * 4 + j
        curr_samples = samples_short[layer_idxs_short == curr_layer]#select hits in ith layer
        curr_truth = truth_times[truth_times[:,1] == curr_layer][:,0]
        layer_fig.suptitle(f"photon time of arrival on sensor binned by layer # {particle}",y = 0.94)
        layer_axs[i][j].hist(curr_samples,density = True, bins = 100,color = "blue",alpha = 0.5,label = "learned")
        layer_axs[i][j].hist(curr_truth,density = True, bins = 100, color = "red",alpha = 0.4,label = "truth")
        layer_axs[i][j].set_title(f"layer #{curr_layer}")
        if(i == 0 and j == 0):
            layer_axs[i][j].legend()
        if(i == 6):
            layer_axs[i][j].set_xlabel(f"time (ns)")
        if(j == 0):
            layer_axs[i][j].set_ylabel(f"counts (normalized)")
            
layer_fig.savefig(workdir + f"macros/Timing_estimation/plots/truth_comp/{particle}_test_file_july_25_run_1_layer_binning.pdf")