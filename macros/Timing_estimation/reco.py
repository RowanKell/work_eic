import numpy as np
import uproot as up
import numba as nb
from multiprocessing import Pool
import pandas as pd
import matplotlib.pyplot as plot
from scipy.stats import norm
from scipy.optimize import curve_fit
import sympy
from IPython.display import clear_output
import math
import time
import util
#My imports
from util import PVect,get_layer,create_layer_map,theta_func,phi_func,findBin,bin_percent_theta_phi, train, test, create_data, create_data_depth,p_func, calculate_num_pixels,Classifier,plot_roc_curve
import torch
layer_map, super_layer_map = create_layer_map()

# @nb.njit
def inverse(x, a, b, c):
    return a / (x + b) + c

# @nb.njit
def calculate_num_pixels_z_dependence(energy_dep, z_hit):
    efficiency = inverse(770 - z_hit, 494.98, 9.9733, -0.16796)
    return 10 * energy_dep * (1000 * 1000) * efficiency / 100
def calculate_efficiency(z_hit):
    return inverse(770 - z_hit, 494.98, 9.9733, -0.16796)

num_layers = 28

def get_label(PDG):
    return (PDG + 211) // 224

def create_unique_mapping(arr):
    # Get unique values and their inverse mapping
    unique_values, inverse_indices = np.unique(arr, return_inverse=True)
    
    # Create a dictionary mapping unique values to their indices
    value_to_index = {val: idx for idx, val in enumerate(unique_values)}
    
    # Create an array of indices
    index_array = inverse_indices
    
    return len(unique_values), value_to_index

def process_data(uproot_path, file_num=0, particle="pion"):
    data = []
    events = up.open(uproot_path)
    
    x_pos_branch = events["HcalBarrelHits/HcalBarrelHits.position.x"].array(library='np')
    z_pos_branch = events["HcalBarrelHits/HcalBarrelHits.position.z"].array(library='np')
    EDep_branch = events["HcalBarrelHits.EDep"].array(library='np')
    PDG_branch = events["MCParticles.PDG"].array(library='np')
    x_momentum_branch = events["HcalBarrelHits/HcalBarrelHits.momentum.x"].array(library='np')
    y_momentum_branch = events["HcalBarrelHits/HcalBarrelHits.momentum.y"].array(library='np')
    z_momentum_branch = events["HcalBarrelHits/HcalBarrelHits.momentum.z"].array(library='np')
    Hits_MC_idx_branch = events["_HcalBarrelHits_MCParticle.index"].array(library='np')
    time_branch = events["HcalBarrelHits.time"].array(library='np')   
    num_events = len(x_pos_branch)
    for event_idx in range(num_events):
        Hits_MC_idx_event = Hits_MC_idx_branch[event_idx]
        n_unique_parts, idx_dict = create_unique_mapping(Hits_MC_idx_event)
        
        p_layer_list = np.ones((n_unique_parts,num_layers)) * -1
        z_hit_layer_list = np.ones((n_unique_parts,num_layers)) * -1
        theta_layer_list = np.ones((n_unique_parts,num_layers)) * -1
        hit_time_layer_list = np.ones((n_unique_parts,num_layers)) * -1
        edep_event = np.ones((n_unique_parts,num_layers)) * -1
        
        x_pos_event = x_pos_branch[event_idx]
        px_event = x_momentum_branch[event_idx]
        py_event = y_momentum_branch[event_idx]
        pz_event = z_momentum_branch[event_idx]
        z_event = z_pos_branch[event_idx]
        time_event = time_branch[event_idx]
        EDep_event = EDep_branch[event_idx]
        for hit_idx in range(len(x_pos_event)):
            idx = Hits_MC_idx_branch[event_idx][hit_idx]
            part_idx = idx_dict[idx]
            layer_idx = get_layer(x_pos_event[hit_idx], super_layer_map)
            if layer_idx == -1: #error handling for get_layer
                continue
            elif p_layer_list[part_idx,layer_idx] == -1:
                p_layer_list[part_idx,layer_idx] = np.sqrt(px_event[hit_idx]**2 + py_event[hit_idx]**2 + pz_event[hit_idx]**2)
                z_hit_layer_list[part_idx,layer_idx] = z_event[hit_idx]
                theta_layer_list[part_idx,layer_idx] = np.arctan2(np.sqrt(px_event[hit_idx]**2 + py_event[hit_idx]**2), pz_event[hit_idx])
                hit_time_layer_list[part_idx,layer_idx] = time_event[hit_idx]
                edep_event[part_idx,layer_idx] = EDep_event[hit_idx]
            else:
                edep_event[part_idx,layer_idx] += EDep_event[hit_idx]
        data.append(np.stack([z_hit_layer_list,hit_time_layer_list,theta_layer_list,p_layer_list,(np.floor(calculate_num_pixels_z_dependence(edep_event,z_hit_layer_list)).astype(int))],axis = -1))


    
    return data #returns list: each entry is a diff event array; each event array has shape: (#unique particles, #layers, #features)
                #features: z hit, hit time, theta, p, energy dep
    

from torch.utils.data import TensorDataset, DataLoader

def prepare_data_for_nn(processed_data):
    all_features = []
    all_metadata = []
    
    for event_idx, event_data in enumerate(processed_data):
        for particle_idx in range(event_data.shape[0]):
            for layer_idx in range(event_data.shape[1]):
                features = event_data[particle_idx, layer_idx, :4]  # Get first 4 features
                repeat_count = int(event_data[particle_idx, layer_idx, 4])  # Get 5th feature as repeat count
                
                if not np.any(features == -1) and repeat_count > 0:  # Check if all features are -1 and repeat_count is valid
                    # Repeat the features and metadata by repeat_count
                    all_features.extend([features] * repeat_count)
                    all_metadata.extend([(event_idx, particle_idx, layer_idx)] * repeat_count)
    
    # Convert to numpy arrays
    features_array = np.array(all_features)
    metadata_array = np.array(all_metadata)
    
    return features_array, metadata_array

def create_dataloader(features, metadata, batch_size=32):
    # Convert to PyTorch tensors
    features_tensor = torch.tensor(features, dtype=torch.float32)
    metadata_tensor = torch.tensor(metadata, dtype=torch.long)
    
    # Create TensorDataset
    dataset = TensorDataset(features_tensor, metadata_tensor)
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader
