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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tqdm import tqdm
#My imports
from util import PVect,get_layer,create_layer_map,theta_func,phi_func,findBin,bin_percent_theta_phi, create_data, create_data_depth,p_func, calculate_num_pixels,plot_roc_curve
import torch
import torch.nn as nn
layer_map, super_layer_map = create_layer_map()
import os
# @nb.njit
def inverse(x, a, b, c):
    return a / (x + b) + c

# @nb.njit
def calculate_num_pixels_z_dependence(energy_dep, z_hit):
    efficiency = inverse(770 - z_hit, 494.98, 9.9733, -0.16796)
    return 10 * energy_dep * (1000 * 1000) * efficiency / 100 * 50 #50% for QE
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
    num_layers = 28
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
        PDG_event = PDG_branch[event_idx]
        n_unique_parts, idx_dict = create_unique_mapping(Hits_MC_idx_event)
        
        p_layer_list = np.ones((num_layers,n_unique_parts)) * -999
        z_hit_layer_list = np.ones((num_layers,n_unique_parts)) * -999
        theta_layer_list = np.ones((num_layers,n_unique_parts)) * -999
        hit_time_layer_list = np.ones((num_layers,n_unique_parts)) * -999
        edep_event = np.ones((num_layers,n_unique_parts)) * -999
        PDG_list = np.ones((num_layers,n_unique_parts)) * -999
        
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
            elif p_layer_list[layer_idx,part_idx] == -999:
                p_layer_list[layer_idx,part_idx] = np.sqrt(px_event[hit_idx]**2 + py_event[hit_idx]**2 + pz_event[hit_idx]**2)
                z_hit_layer_list[layer_idx,part_idx] = z_event[hit_idx]
                theta_layer_list[layer_idx,part_idx] = np.arctan2(np.sqrt(px_event[hit_idx]**2 + py_event[hit_idx]**2), pz_event[hit_idx])
                hit_time_layer_list[layer_idx,part_idx] = time_event[hit_idx]
                edep_event[layer_idx,part_idx] = EDep_event[hit_idx]
                PDG_list[layer_idx,part_idx] = PDG_event[part_idx]
            else:
                edep_event[layer_idx,part_idx] += EDep_event[hit_idx]
        data.append(np.stack([z_hit_layer_list,theta_layer_list,p_layer_list,hit_time_layer_list,(np.floor(calculate_num_pixels_z_dependence(edep_event,z_hit_layer_list)).astype(int))],axis = -1))


    
    return data #returns list: each entry is a diff event array; each event array has shape: (#unique particles, #layers, #features)
                #features: z hit, hit time, theta, p, energy dep
def process_data_one_segment(uproot_path, file_num=0, particle="pion"):
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
    num_features = 5
    
    data = np.ones((num_events,num_features),dtype=float)
    
    for event_idx in range(num_events):
        Hits_MC_idx_event = Hits_MC_idx_branch[event_idx]
        PDG_event = PDG_branch[event_idx]
        
        p = -999
        z_hit = -999
        theta = -999
        hit_time = -999
        edep_event = -999
        PDG_list = -999
        
        x_pos_event = x_pos_branch[event_idx]
        px_event = x_momentum_branch[event_idx]
        py_event = y_momentum_branch[event_idx]
        pz_event = z_momentum_branch[event_idx]
        z_event = z_pos_branch[event_idx]
        time_event = time_branch[event_idx]
        EDep_event = EDep_branch[event_idx]
        for hit_idx in range(len(x_pos_event)):
            idx = Hits_MC_idx_branch[event_idx][hit_idx]
            if(PDG_event[idx] != 13):
                continue
            if (p == -999):
                p = np.sqrt(px_event[hit_idx]**2 + py_event[hit_idx]**2 + pz_event[hit_idx]**2)
                z_hit = z_event[hit_idx]
                theta = np.arctan2(np.sqrt(px_event[hit_idx]**2 + py_event[hit_idx]**2), pz_event[hit_idx]) * 180 / 3.14159
                hit_time = time_event[hit_idx]
                edep_event = EDep_event[hit_idx]
                PDG_list = PDG_event[idx]
            else:
                edep_event += EDep_event[hit_idx]
        data[event_idx] = np.stack([z_hit,theta,p,hit_time,(np.floor(calculate_num_pixels_z_dependence(edep_event,z_hit)).astype(int))],axis = -1)


    
    return data #returns list: each entry is a diff event array; each event array has shape: (#unique particles, #layers, #features)
                #features: z hit, hit time, theta, p, energy dep

from torch.utils.data import TensorDataset, DataLoader

def prepare_data_for_nn(processed_data):
    all_features = []
    all_metadata = []
    print(f"len of events: {len(processed_data)}")
    for event_idx, event_data in enumerate(processed_data):
        for particle_idx in range(event_data.shape[0]):
            for layer_idx in range(event_data.shape[1]):
                features = event_data[particle_idx, layer_idx, :4]  # Get first 4 features
                repeat_count = int(event_data[particle_idx, layer_idx, 4])  # Get 5th feature as repeat count
                
                #cuts
                if(features[3] > 50):
                    continue
                
                
                if not np.any(features == -1) and repeat_count > 0:  # Check if all features are -1 and repeat_count is valid
                    # Repeat the features and metadata by repeat_count
                    all_features.extend([features] * repeat_count)
                    all_metadata.extend([(event_idx, particle_idx, layer_idx)] * repeat_count)
    
    # Convert to numpy arrays
    features_array = np.array(all_features)
    metadata_array = np.array(all_metadata)
    
    return features_array, metadata_array
def prepare_data_for_nn_one_segment(processed_data):
    all_features = []
    all_metadata = []
#     print(f"len of events: {len(processed_data)}")
    for event_idx, event_data in enumerate(processed_data):
        features = event_data[:4]  # Get first 4 features
        repeat_count = int(event_data[4])  # Get 5th feature as repeat count

        #cuts
        if(features[3] > 50):
            continue


        if not np.any(features == -1) and repeat_count > 0:  # Check if all features are -1 and repeat_count is valid
            # Repeat the features and metadata by repeat_count
            all_features.extend([features] * repeat_count)
            all_metadata.extend([(event_idx)] * repeat_count)
    
    # Convert to numpy arrays
    features_array = np.array(all_features)
    metadata_array = np.array(all_metadata)
    
    return features_array, metadata_array
def create_dataloader(features, metadata, batch_size=32,shuffle_bool=True):
    # Convert to PyTorch tensors
    features_tensor = torch.tensor(features, dtype=torch.float32)
    metadata_tensor = torch.tensor(metadata, dtype=torch.long)
    print(features.shape)
    # Create TensorDataset
    dataset = TensorDataset(features_tensor, metadata_tensor)
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_bool)
    
    return dataloader

'''
Utility functions for loading in truth data quickly
'''

from concurrent.futures import ThreadPoolExecutor
from typing import Union, List



#Load real data



def process_file(file_name):
    time_branch_name = "HcalBarrelHits.time"
    hit_x_branch_name = "HcalBarrelHits.position.x"
    tree_ext = ":events"
    with up.open(file_name + tree_ext) as file:
        times = file[time_branch_name].array(library="np")
        x_hits = file[hit_x_branch_name].array(library="np")
    return times, x_hits

def vectorized_get_layer(x_pos_array):
    return np.array([get_layer(x) for x in x_pos_array])

def load_truth(file_dir):

    file_names = [file_dir + name for name in os.listdir(file_dir) if not os.path.isdir(os.path.join(file_dir, name))]
    tree_ext = ":events"

    layer_map, super_layer_map = create_layer_map()

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_file, file_names))

    # Combine results
    event_times = np.concatenate([r[0] for r in results])
    event_x_hit = np.concatenate([r[1] for r in results])

    # Process events
    truth_times = []
    for times, x_hits in zip(event_times, event_x_hit):
        mask = times < 50
        if np.any(mask):  # Only process if there are hits passing the time condition
            try:
                layer_idx = vectorized_get_layer(x_hits[mask])
                truth_times.extend(np.column_stack((times[mask], layer_idx)))
            except Exception as e:
                print(f"Error processing event: {e}")
                print(f"x_hits[mask]: {x_hits[mask]}")
                continue

    truth_times = np.array(truth_times)

    print(f"Processed {len(truth_times)} hits")
    print(f"Shape of truth_times: {truth_times.shape}")
    return truth_times

'''
Methods for Energy_reco.ipynb energy reconstruction
'''

def get_p(px,py,pz):
    sq = pow(px,2) + pow(py,2) + pow(pz,2)
    return pow(sq,0.5)
#preprocess data/put data into training format
def process_data_energy_reco(uproot_path, file_num=0, particle="pion"):
    num_layers = 28
    events = up.open(uproot_path)
    
    x_pos_branch = events["HcalBarrelHits/HcalBarrelHits.position.x"].array(library='np')
    z_pos_branch = events["HcalBarrelHits/HcalBarrelHits.position.z"].array(library='np')
    EDep_branch = events["HcalBarrelHits.EDep"].array(library='np')
    PDG_branch = events["MCParticles.PDG"].array(library='np')
    mc_px = events["MCParticles.momentum.x"].array(library='np')
    mc_py = events["MCParticles.momentum.y"].array(library='np')
    mc_pz = events["MCParticles.momentum.z"].array(library='np')
    x_momentum_branch = events["HcalBarrelHits/HcalBarrelHits.momentum.x"].array(library='np')
    y_momentum_branch = events["HcalBarrelHits/HcalBarrelHits.momentum.y"].array(library='np')
    z_momentum_branch = events["HcalBarrelHits/HcalBarrelHits.momentum.z"].array(library='np')
    Hits_MC_idx_branch = events["_HcalBarrelHits_MCParticle.index"].array(library='np')
    time_branch = events["HcalBarrelHits.time"].array(library='np')   
    num_events = len(x_pos_branch)
    data = torch.empty(num_events,2 * num_layers + 1)
    for event_idx in range(num_events):
        clear_output(wait=True)
        print(f"Event #{event_idx}")
        primary_momentum = get_p(mc_px[event_idx][0],mc_py[event_idx][0],mc_pz[event_idx][0])
        
        Hits_MC_idx_event = Hits_MC_idx_branch[event_idx]
        PDG_event = PDG_branch[event_idx]
        n_unique_parts, idx_dict = create_unique_mapping(Hits_MC_idx_event)
        
        z_hit_layer_list = np.ones((num_layers)) * -999
        edep_event = np.ones((num_layers)) * -999
        
        x_pos_event = x_pos_branch[event_idx]
        z_event = z_pos_branch[event_idx]
        EDep_event = EDep_branch[event_idx]
        
        time_event = time_branch[event_idx]
        event_layer_first_times = np.ones(28) * 9999
        event_layer_all_times = [[] for i in range(28)]
        for hit_idx in range(len(x_pos_event)):
            idx = Hits_MC_idx_branch[event_idx][hit_idx]
            layer_idx = get_layer(x_pos_event[hit_idx], super_layer_map)
            if layer_idx == -1: #error handling for get_layer
                continue
            event_layer_all_times[layer_idx].append(time_event[hit_idx])
            if edep_event[layer_idx] == -999:
                z_hit_layer_list[layer_idx] = z_event[hit_idx]
                edep_event[layer_idx] = EDep_event[hit_idx]
            else:
                edep_event[layer_idx] += EDep_event[hit_idx]
        for i in range(28):
            if(len(event_layer_all_times[i])):
                event_layer_first_times[i] = min(event_layer_all_times[i])
        data[event_idx][:28] = torch.tensor(np.floor(calculate_num_pixels_z_dependence(edep_event,z_hit_layer_list)).astype(int))
        data[event_idx][28:56] = torch.tensor(event_layer_first_times)
        data[event_idx][56] = primary_momentum
    data[data < 0] = 0
    return data #returns list: each entry is a diff event array; each event array has shape: (57) - pixels per layer then first photon time per layer than label (primary momentum)

# shuffle and split data into train test val
def shuffle_segment_data(inputs, train_frac = 0.8, test_frac = 0.1, val_frac = 0.1):
    indexes = torch.randperm(inputs.shape[0])
    dataset = inputs[indexes]
    train_lim = int(np.floor(dataset.shape[0] * train_frac))
    test_lim = train_lim + int(np.floor(dataset.shape[0] * test_frac))
    val_lim = test_lim + int(np.floor(dataset.shape[0] * val_frac))
    train_data = dataset[:train_lim]
    test_data = dataset[train_lim:test_lim]
    val_data = dataset[test_lim:val_lim]
    return (train_data, test_data, val_data)
class Predictor(nn.Module):
    """
    Prediction network
    """
    def __init__(self, input_size=28, num_classes=2, hidden_dim = 512, num_layers = 10):
        super(Predictor, self).__init__()
        self.layer = nn.Sequential()
        for i in range(num_layers):
            if(i == 0):
                self.layer.append(
                nn.Linear(input_size, hidden_dim)
                )
                self.layer.append(
                    nn.LeakyReLU(inplace=True)
                )
            elif(i == num_layers - 1):
                self.layer.append(
                nn.Linear(hidden_dim, num_classes)
                )
            else:
                self.layer.append(
                    nn.Linear(hidden_dim, hidden_dim)
                )
                self.layer.append(
                    nn.LeakyReLU(inplace=True)
                )
        self.name = "Predictor"
#         self.double()
        
    def forward(self, h):
        c = self.layer(h)
        return c
    
    # @property
    def name(self):
        """
        Name of model.
        """
        return self.name
#training loop for prediction model
def train(predictor, train_data,optimizer,device, num_epochs = 18, batch_size = 100, show_progress = True):
    max_index = train_data.shape[1] - 1
    
    criterion = nn.MSELoss()
    predictor.train()
    num_it = train_data.shape[0] // batch_size

    show_progress = True
    loss_hist = []
    curr_losses = []
    for i in range(num_epochs):
        clear_output(wait=True)
        print(f"Training epoch #{i}")
        epoch_hist = np.array([])
        val_epoch_hist = np.array([])
        with tqdm(total=num_it, position=0, leave=True) as pbar:
            for it in range(num_it):
                optimizer.zero_grad()
                begin = it * batch_size
                end = (it + 1) * batch_size
                it_data = train_data[begin:end]
                samples = it_data[:,:max_index]
                labels = it_data[:,max_index].unsqueeze(-1)
                samples = samples.to(device)
                labels = labels.to(device)
                outputs = predictor(samples)
                loss = criterion(outputs, labels)
                # Do backprop and optimizer step
                if ~(torch.isnan(loss) | torch.isinf(loss)):
                    loss.backward()
                    optimizer.step()
                # Log loss
                if~(torch.isnan(loss)):
                    curr_losses.append(loss.to('cpu').data.numpy())
                    if(not (it % 5)):
                        loss_hist.append(sum(curr_losses) / len(curr_losses))
                        curr_losses = []
                if(show_progress):
                    pbar.update(1)
        

    print('Finished Training')
    return loss_hist

def plot_loss_predictions(loss_list,save_loc = ''):
    fig,axs = plot.subplots(1,1)
    fig.suptitle("Prediction model loss over iterations")
    axs.plot(loss_list, label = 'train loss')
    axs.legend()
    fig.show()
    if(save_loc != ''):
        fig.savefig(save_loc)

#plot predictions vs correct
def plot_predictions(true_values, predictions):
    plot.figure(figsize=(10, 6))
    alpha = 0.55 - 0.5*(len(true_values) / 100000)
    plot.scatter(true_values, predictions, alpha=alpha)
    plot.plot([true_values.min(), true_values.max()], [true_values.min(), true_values.max()], 'r--', lw=2)
    plot.xlabel('True Values')
    plot.ylabel('Predictions')
    plot.title('True Values vs. Predictions')
    plot.tight_layout()
    plot.show()
def evaluate_model(model, test_data, device):
    model.eval()  # Set the model to evaluation mode
    max_index = test_data.shape[1] - 1
    
    with torch.no_grad():
        samples = test_data[:, :max_index].float().to(device)
        true_values = test_data[:, max_index].float().cpu().numpy()
        
        predictions = model(samples).cpu().numpy().flatten()
    
    # Calculate metrics
    mae = mean_absolute_error(true_values, predictions)
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_values, predictions)
    
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R-squared Score: {r2:.4f}")
    plot_predictions(true_values, predictions)
    return mae, mse, rmse, r2