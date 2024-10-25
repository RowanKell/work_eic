import uproot
import numpy as np
import torch
from collections import defaultdict
from util import get_layer, theta_func,create_layer_map,calculate_num_pixels_z_dependence
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
import normflows as nf
import datetime
from torch import nn
from scipy import signal
import optuna
from typing import Optional, Union, Literal, Dict, Any, List,Tuple
import json
from datetime import datetime as dt

layer_map, super_layer_map = create_layer_map()



def load_and_concatenate_tensors(directory):
    tensor_files = [f for f in os.listdir(directory) if f.endswith('.pt')]
    tensors = []

    for file in tensor_files:
        file_path = os.path.join(directory, file)
        tensor = torch.load(file_path)
        tensors.append(tensor)

    # Concatenate all tensors along the first dimension (assuming this is what you want)
    concatenated_tensor = torch.cat(tensors, dim=0)

    return concatenated_tensor
def filter_tensors_by_values(tensor1, tensor2, threshold1=200, threshold2=10000,outputthreshold=100):
    """
    Filter two tensors based on value thresholds, keeping event indices aligned.
    
    Args:
        tensor1: First tensor with shape [event, layer, SiPM, value]
        tensor2: Second tensor with same shape
        threshold1: Maximum allowed value for first value
        threshold2: Maximum allowed value for second value
        
    Returns:
        Tuple of filtered tensors (filtered_tensor1, filtered_tensor2)
    """
    # Get first and second values for all events
    first_values = tensor1[..., 0]   # [...] keeps all dimensions except last
    second_values = tensor1[..., 1]
    
    # Create masks for each condition
    mask_first = (first_values < threshold1).all(dim=(1, 2))  # Check across layer and SiPM dims
    mask_second = (second_values < threshold2).all(dim=(1, 2))
    mask_outputs = (tensor2 < outputthreshold)
    
    # Combine masks
    valid_events = mask_first & mask_second & mask_outputs
    
    # Apply masks to both tensors
    filtered_tensor1 = tensor1[valid_events]
    filtered_tensor2 = tensor2[valid_events]
    print("fraction %.2f of events survived filters"%(len(filtered_tensor2) / len(tensor2)))
    return filtered_tensor1, filtered_tensor2

def process_root_file(file_path,max_events = -1):
    print("began processing")
    with uproot.open(file_path) as file:
        tree_HcalBarrelHits = file["events/HcalBarrelHits"]
        tree_MCParticles = file["events/MCParticles"]
        
        
        momentum_x_MC = tree_MCParticles["MCParticles.momentum.x"].array(library="np")
        momentum_y_MC = tree_MCParticles["MCParticles.momentum.y"].array(library="np")
        momentum_z_MC = tree_MCParticles["MCParticles.momentum.z"].array(library="np")
        
        z_pos = tree_HcalBarrelHits["HcalBarrelHits.position.z"].array(library="np")
        x_pos = tree_HcalBarrelHits["HcalBarrelHits.position.x"].array(library="np")
        energy = tree_HcalBarrelHits["HcalBarrelHits.EDep"].array(library="np")
        momentum_x = tree_HcalBarrelHits["HcalBarrelHits.momentum.x"].array(library="np")
        momentum_y = tree_HcalBarrelHits["HcalBarrelHits.momentum.y"].array(library="np")
        momentum_z = tree_HcalBarrelHits["HcalBarrelHits.momentum.z"].array(library="np")
        hit_time = tree_HcalBarrelHits["HcalBarrelHits.time"].array(library="np")
        mc_hit_idx = file["events/_HcalBarrelHits_MCParticle/_HcalBarrelHits_MCParticle.index"].array(library="np")  # Add PDG code for particle identification
        print("finished loading branches")
        
        processed_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        
        for event_idx in tqdm(range(len(z_pos))):
            if(len(z_pos[event_idx]) == 0):
                continue
            primary_momentum = (momentum_x_MC[event_idx][0],
                            momentum_y_MC[event_idx][0],
                            momentum_z_MC[event_idx][0])
            primary_momentum_mag = np.linalg.norm(primary_momentum)
            if(primary_momentum_mag <= 0):
                continue
            if(primary_momentum_mag > 100):
                continue
            energy_per_layer_particle = defaultdict(lambda: defaultdict(float))
            first_hit_per_layer_particle = defaultdict(dict)
            # First pass: collect first hit data and calculate energy per layer per particle
            for hit_idx in range(len(z_pos[event_idx])):
                z = z_pos[event_idx][hit_idx]
                x = x_pos[event_idx][hit_idx]
                e = energy[event_idx][hit_idx]
                momentum = (momentum_x[event_idx][hit_idx],
                            momentum_y[event_idx][hit_idx],
                            momentum_z[event_idx][hit_idx])
                momentum_mag = np.linalg.norm(momentum)
                theta = theta_func(momentum_x[event_idx][hit_idx], momentum_y[event_idx][hit_idx], momentum_z[event_idx][hit_idx])
                layer = get_layer(x)
                particle_id = mc_hit_idx[event_idx][hit_idx]
                
                energy_per_layer_particle[layer][particle_id] += e
                
                if layer not in first_hit_per_layer_particle or particle_id not in first_hit_per_layer_particle[layer]:
                    first_hit_per_layer_particle[layer][particle_id] = {
                        "z_pos": z,
                        "x_pos": x,
                        "momentum": momentum_mag,
                        "primary_momentum": primary_momentum_mag,
                        "theta": theta,
                        "time": hit_time[event_idx][hit_idx],
                        "mc_hit_idx": particle_id
                    }
            
            
            # Second pass: process first hit with total layer energy per particle
            for layer, particle_data in first_hit_per_layer_particle.items():
                for particle_id, hit_data in particle_data.items():
                    layer_particle_energy = energy_per_layer_particle[layer][particle_id]
                    num_pixels_high_z = calculate_num_pixels_z_dependence(layer_particle_energy, hit_data["z_pos"])
                    num_pixels_low_z = calculate_num_pixels_z_dependence(layer_particle_energy, -1 * hit_data["z_pos"])
                    hit_data["num_pixels_high_z"] = int(np.floor(num_pixels_high_z))
                    hit_data["num_pixels_low_z"] = int(np.floor(num_pixels_low_z))
                    hit_data["layer_energy"] = layer_particle_energy  # Store total layer energy for this particle
                    processed_data[event_idx][layer][particle_id.item()] = hit_data
            if(max_events > 0 and event_idx > max_events):
                break
    print("finished processing")
    return processed_data
def new_prepare_nn_input(processed_data, normalizing_flow, batch_size=1024, device='cuda'):
    nn_input = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    nn_output = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    all_context = []
    all_time_pixels = []
    all_metadata = []
    num_pixel_list = ["num_pixels_high_z","num_pixels_low_z"]
    print("Processing data in new_prepare_nn_input...")
    for event_idx, event_data in tqdm(processed_data.items()):
        for layer, layer_data in event_data.items():
            for particle_id, particle_data in layer_data.items():
                primary_momentum = particle_data["primary_momentum"].item()
                base_context = torch.tensor([particle_data['z_pos'], particle_data['theta'], particle_data['momentum']], 
                                            dtype=torch.float32)
                base_time_pixels_low = torch.tensor([particle_data['time'], particle_data['num_pixels_low_z']], 
                                                dtype=torch.float32)
                base_time_pixels_high = torch.tensor([particle_data['time'], particle_data['num_pixels_high_z']], 
                                                dtype=torch.float32)
                
                for SiPM_idx in range(2):
                    z_pos = particle_data['z_pos']
                    context = base_context.clone()
                    context[0] = z_pos
                    num_pixel_tag = num_pixel_list[SiPM_idx]
                    all_context.append(context.repeat(particle_data[num_pixel_tag], 1))
                    if(SiPM_idx == 0):
                        all_time_pixels.append(base_time_pixels_high.repeat(particle_data[num_pixel_tag], 1))
                    else:
                        all_time_pixels.append(base_time_pixels_low.repeat(particle_data[num_pixel_tag], 1))
                    all_metadata.extend([(event_idx, layer, SiPM_idx, particle_id, primary_momentum)] * particle_data[num_pixel_tag])

    all_context = torch.cat(all_context)
    all_time_pixels = torch.cat(all_time_pixels)
    
    print("Sampling data...")
    sampled_data = []
    begin = time.time()
    for i in tqdm(range(0, len(all_context), batch_size)):
        batch_end = min(i + batch_size, len(all_context))
        batch_context = all_context[i:batch_end].to(device)
        batch_time_pixels = all_time_pixels[i:batch_end]
        
        with torch.no_grad():
            samples = abs(normalizing_flow.sample(num_samples=len(batch_context), context=batch_context)[0]).squeeze(1)
        
        sampled_data.extend(samples.cpu() + batch_time_pixels[:, 0])
    end = time.time()
    print(f"sampling took {end - begin} seconds")
    print("Reorganizing data...")
    begin = time.time()
    for (event, layer, SiPM, particle, momentum), sample in zip(all_metadata, sampled_data):
        nn_input[event][layer][SiPM].append(sample)
        nn_output[event][layer][SiPM].append(torch.tensor([momentum]))
    end = time.time()
    print(f"reorganizing took {end - begin} seconds")
    return nn_input, nn_output



def prepare_prediction_input(nn_input, nn_output):
    #note - some events do not have dictionaries in nn_input due to being empty
    #need to skip over these and condense tensor
    prediction_input = torch.ones(len(nn_input),28,10) * 999
    prediction_output = torch.ones(len(nn_input)) * 999
    
    input_dict = defaultdict(lambda: defaultdict(list))
    output_dict = {}
    curr_event_num = 0
    for event_idx in tqdm(list(nn_input)):
        event_input = []
        set_output = False
        layer_keys = nn_input[event_idx].keys()
        for layer in range(28):
            if(layer in layer_keys):
                layer_times = torch.tensor(sorted(nn_input[event_idx][layer]))
                # Pad or truncate to exactly 10 times per layer
                if len(layer_times) < 10:
                    padding = torch.full((10 - len(layer_times),), float(999))
                    layer_times = torch.cat([layer_times, padding])
                if(not set_output):
                    prediction_output[curr_event_num] = nn_output[event_idx][layer][0]
                    set_output = True
            else:
                layer_times = torch.full([28],999)

            prediction_input[curr_event_num][layer] = layer_times[:10]
        curr_event_num += 1
    return prediction_input, prediction_output
def prepare_prediction_input_pulse(nn_input, nn_output,pixel_threshold = 3):
    processor = SiPMSignalProcessor()
    
    #note - some events do not have dictionaries in nn_input due to being empty
    #need to skip over these and condense tensor
    prediction_input = torch.ones(len(nn_input),28,2,2) * 9999
    prediction_output = torch.ones(len(nn_input)) * 9999
    
    input_dict = defaultdict(lambda: defaultdict(list))
    output_dict = {}
    curr_event_num = 0
    for event_idx in tqdm(list(nn_input)):
        event_input = []
        set_output = False
        layer_keys = nn_input[event_idx].keys()
        for layer in range(28):
            if(layer in layer_keys):
                for SiPM_idx in range(2):
                    if (len(nn_input[event_idx][layer][SiPM_idx])>0):
                        photon_times = torch.tensor(sorted(nn_input[event_idx][layer][SiPM_idx])) * 10 **(-9)

                        #calculate time and charge
                        time,waveform = processor.generate_waveform(photon_times)
                        charge = processor.integrate_charge(waveform)
                        timing = processor.get_pulse_timing(waveform,threshold = pixel_threshold)
                        if(not set_output):
                            prediction_output[curr_event_num] = nn_output[event_idx][layer][SiPM_idx][0]
                            set_output = True
                        if(timing is not None):
                            prediction_input[curr_event_num][layer][SiPM_idx][0] = charge * 1e6
                            prediction_input[curr_event_num][layer][SiPM_idx][1] = timing * 1e8
                        else: #pad with 0 and 9999 if SiPM pulse doesn't meet threshold
                            prediction_input[curr_event_num][layer][SiPM_idx][0] = 0
                            prediction_input[curr_event_num][layer][SiPM_idx][1] = 9999
                            
                    else:
                        prediction_input[curr_event_num][layer][SiPM_idx][0] = 0
                        prediction_input[curr_event_num][layer][SiPM_idx][1] = 9999
            else:
                charge = 0
                timing = 9999
                prediction_input[curr_event_num][layer][0][0] = charge
                prediction_input[curr_event_num][layer][0][1] = timing
                prediction_input[curr_event_num][layer][1][0] = charge
                prediction_input[curr_event_num][layer][1][1] = timing

        curr_event_num += 1
    return prediction_input, prediction_output
def process_root_file_for_greg(file_path):
    print("began processing")
    with uproot.open(file_path) as file:
        tree_HcalBarrelHits = file["events/HcalBarrelHits"]
        tree_MCParticles = file["events/MCParticles"]
        
        
        momentum_x_MC = tree_MCParticles["MCParticles.momentum.x"].array(library="np")
        momentum_y_MC = tree_MCParticles["MCParticles.momentum.y"].array(library="np")
        momentum_z_MC = tree_MCParticles["MCParticles.momentum.z"].array(library="np")
        
        z_pos = tree_HcalBarrelHits["HcalBarrelHits.position.z"].array(library="np")
        x_pos = tree_HcalBarrelHits["HcalBarrelHits.position.x"].array(library="np")
        energy = tree_HcalBarrelHits["HcalBarrelHits.EDep"].array(library="np")
        momentum_x = tree_HcalBarrelHits["HcalBarrelHits.momentum.x"].array(library="np")
        momentum_y = tree_HcalBarrelHits["HcalBarrelHits.momentum.y"].array(library="np")
        momentum_z = tree_HcalBarrelHits["HcalBarrelHits.momentum.z"].array(library="np")
        hit_time = tree_HcalBarrelHits["HcalBarrelHits.time"].array(library="np")
        mc_hit_idx = file["events/_HcalBarrelHits_MCParticle/_HcalBarrelHits_MCParticle.index"].array(library="np")  # Add PDG code for particle identification
        print("finished loading branches")
        
        processed_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        
        for event_idx in tqdm(range(len(z_pos))):
            if(len(z_pos[event_idx]) == 0):
                continue
            primary_momentum = (momentum_x_MC[event_idx][0],
                            momentum_y_MC[event_idx][0],
                            momentum_z_MC[event_idx][0])
            primary_momentum_mag = np.linalg.norm(primary_momentum)
            if(primary_momentum_mag <= 0):
                continue
            if(primary_momentum_mag > 100):
                continue
            energy_per_layer_particle = defaultdict(lambda: defaultdict(float))
            first_hit_per_layer_particle = defaultdict(dict)
            # First pass: collect first hit data and calculate energy per layer per particle
            for hit_idx in range(len(z_pos[event_idx])):
                z = z_pos[event_idx][hit_idx]
                x = x_pos[event_idx][hit_idx]
                e = energy[event_idx][hit_idx]
                momentum = (momentum_x[event_idx][hit_idx],
                            momentum_y[event_idx][hit_idx],
                            momentum_z[event_idx][hit_idx])
                momentum_mag = np.linalg.norm(momentum)
                theta = theta_func(momentum_x[event_idx][hit_idx], momentum_y[event_idx][hit_idx], momentum_z[event_idx][hit_idx])
                layer = get_layer(x)
                particle_id = mc_hit_idx[event_idx][hit_idx]
                
                energy_per_layer_particle[layer][particle_id] += e
                
                if layer not in first_hit_per_layer_particle or particle_id not in first_hit_per_layer_particle[layer]:
                    first_hit_per_layer_particle[layer][particle_id] = {
                        "z_pos": z,
                        "x_pos": x,
                        "momentum": momentum_mag,
                        "primary_momentum": primary_momentum_mag,
                        "theta": theta,
                        "time": hit_time[event_idx][hit_idx],
                        "mc_hit_idx": particle_id,
                        "pid": pid,
                        "phi": phi
                    }
            
            
            # Second pass: process first hit with total layer energy per particle
            for layer, particle_data in first_hit_per_layer_particle.items():
                for particle_id, hit_data in particle_data.items():
                    layer_particle_energy = energy_per_layer_particle[layer][particle_id]
                    num_pixels_high_z = calculate_num_pixels_z_dependence(layer_particle_energy, hit_data["z_pos"])
                    num_pixels_low_z = calculate_num_pixels_z_dependence(layer_particle_energy, -1 * hit_data["z_pos"])
                    hit_data["num_pixels_high_z"] = int(np.floor(num_pixels_high_z))
                    hit_data["num_pixels_low_z"] = int(np.floor(num_pixels_low_z))
                    hit_data["layer_energy"] = layer_particle_energy  # Store total layer energy for this particle
                    processed_data[event_idx][layer][particle_id.item()] = hit_data
    
    print("finished processing")
    return processed_data
def new_prepare_nn_input_for_greg(processed_data, normalizing_flow, batch_size=1024, device='cuda'):
    nn_input = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    nn_output = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    all_context = []
    all_time_pixels = []
    all_metadata = []
    num_pixel_list = ["num_pixels_high_z","num_pixels_low_z"]
    
    print("Processing data...")
    for event_idx, event_data in tqdm(processed_data.items()):
        for layer, layer_data in event_data.items():
            for particle_id, particle_data in layer_data.items():
                primary_momentum = particle_data["primary_momentum"].item()
                base_context = torch.tensor([particle_data['z_pos'], particle_data['theta'], particle_data['momentum']], 
                                            dtype=torch.float32)
                base_time_pixels_low = torch.tensor([particle_data['time'], particle_data['num_pixels_low_z']], 
                                                dtype=torch.float32)
                base_time_pixels_high = torch.tensor([particle_data['time'], particle_data['num_pixels_high_z']], 
                                                dtype=torch.float32)
                
                for SiPM_idx in range(2):
                    z_pos = particle_data['z_pos']
                    context = base_context.clone()
                    context[0] = z_pos
                    num_pixel_tag = num_pixel_list[SiPM_idx]
                    all_context.append(context.repeat(particle_data[num_pixel_tag], 1))
                    if(SiPM_idx == 0):
                        all_time_pixels.append(base_time_pixels_high.repeat(particle_data[num_pixel_tag], 1))
                    else:
                        all_time_pixels.append(base_time_pixels_low.repeat(particle_data[num_pixel_tag], 1))
                    all_metadata.extend([(event_idx, layer, SiPM_idx, particle_id, primary_momentum,particle_data['mc_hit_idx'],particle_data['pid'],particle_data['theta'],particle_data['phi'])] * particle_data[num_pixel_tag])

    all_context = torch.cat(all_context)
    all_time_pixels = torch.cat(all_time_pixels)
    
    print("Sampling data...")
    sampled_data = []
    for i in tqdm(range(0, len(all_context), batch_size)):
        batch_context = all_context[i:i+batch_size].to(device)
        batch_time_pixels = all_time_pixels[i:i+batch_size]
        
        with torch.no_grad():
            samples = abs(normalizing_flow.sample(num_samples=len(batch_context), context=batch_context[:,:3])[0]).squeeze(1)
        
        adjusted_times = samples.cpu() + batch_time_pixels[:, 0]
        sampled_data.extend(adjusted_times)

    print("Reorganizing data...")
    for (event, layer, SiPM, particle, momentum,mc_hit_idx,pid,theta,phi), sample in zip(all_metadata, sampled_data):
        nn_input[event][layer][SiPM].append(sample)
        nn_output[event][layer][SiPM].append(torch.tensor([momentum,particle,mc_hit_idx,pid,theta,phi]))

    return nn_input, nn_output

#TODO Need to get particle idx, theta, phi, p, PID from prepare_prediction_input - don't use new indices, just add as extra values bc we don't need these for my nn training
def prepare_prediction_input_pulse_for_greg(nn_input, nn_output,pixel_threshold = 3):
    processor = SiPMSignalProcessor()
    
    #note - some events do not have dictionaries in nn_input due to being empty
    #need to skip over these and condense tensor
    out_columns = ['event_idx','layer_idx','trueid','truePID','P','Theta','Phi','Charge1','Time1','Charge2','Time2']
    running_index = 0
    out_df = pd.DataFrame(columns = out_columns)
    
    input_dict = defaultdict(lambda: defaultdict(list))
    output_dict = {}
    curr_event_num = 0
    for event_idx in tqdm(list(nn_input)):
        event_input = []
        set_output = False
        layer_keys = nn_input[event_idx].keys()
        for layer in range(28):
            charge_times = np.empty(2,2)
            if(layer in layer_keys): #ignore layers with no hits
                for SiPM_idx in range(2):
                    photon_times = torch.tensor(sorted(nn_input[event_idx][layer][SiPM_idx])) * 10 **(-9)
                    #get relative times
                    if(len(photon_times) > 0):

                        #calculate time and charge
                        time,waveform = processor.generate_waveform(photon_times)
                        charge_times[SiPM_idx][0] = processor.integrate_charge(waveform) * 1e6
                        charge_times[SiPM_idx][1] = processor.get_pulse_timing(waveform,threshold = pixel_threshold) * 1e8
                    else:
                        charge_times[SiPM_idx][0] = 0
                        charge_times[SiPM_idx][1] = 9999
                    
                P = nn_output[event_idx][layer][0][0]
                particle_idx = nn_output[event_idx][layer][0][2]
                pid = nn_output[event_idx][layer][0][3]
                theta = nn_output[event_idx][layer][0][4]
                phi = nn_output[event_idx][layer][0][5]
                new_row = [event_idx,layer_idx,particle_idx,pid,P,theta,phi,charge_times[0,0],charge_times[0,1],charge_times[1,0],charge_times[1,1]]
                out_df.append(new_row,columns=out_columns,index = running_index)
                running_index += 1
        curr_event_num += 1
    return out_df



class Predictor(nn.Module):
    def __init__(
        self,
        input_size: int = 28 * 2 * 2,
        num_classes: int = 1,
        hidden_dim: int = 512,
        num_layers: int = 10,
        dropout_rate: float = 0.1,
        activation: Literal['relu', 'leaky_relu', 'elu'] = 'leaky_relu'
    ):
        super().__init__()
        
        # Store configuration
        self.model_name = "Predictor"
        self.input_size = input_size
        self.config = {
            'input_size': input_size,
            'num_classes': num_classes,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'dropout_rate': dropout_rate,
            'activation': activation
        }
        
        # Create activation function
        self.activation_map = {
            'relu': nn.ReLU,
            'leaky_relu': nn.LeakyReLU,
            'elu': nn.ELU
        }
        activation_fn = self.activation_map[activation]
        
        # Build network layers
        layers = []
        for i in range(num_layers):
            # Input layer
            if i == 0:
                layers.extend([
                    nn.Linear(input_size, hidden_dim),
                    activation_fn(inplace=True),
                    nn.Dropout(dropout_rate)
                ])
            # Output layer
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, num_classes))
            # Hidden layers
            else:
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    activation_fn(inplace=True),
                    nn.Dropout(dropout_rate)
                ])
        
        self.layers = nn.Sequential(*layers)
        
        # Store expected layer count for verification
        self._expected_layer_count = self._calculate_expected_layers(num_layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _calculate_expected_layers(self, num_layers: int) -> int:
        """Calculate expected number of layers based on architecture"""
        # First n-1 layers have Linear + Activation + Dropout (3 components each)
        # Last layer has only Linear (1 component)
        return (num_layers - 1) * 3 + 1
    
    def _get_layer_info(self) -> List[Dict[str, Any]]:
        """Get detailed information about each layer"""
        layer_info = []
        for idx, layer in enumerate(self.layers):
            layer_info.append({
                'index': idx,
                'type': layer.__class__.__name__,
                'params': sum(p.numel() for p in layer.parameters()) if hasattr(layer, 'parameters') else 0
            })
        return layer_info

    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.layers(x)

    def verify_model_integrity(self) -> Dict[str, Any]:
        """
        Verify model structure and parameters.
        Returns dict with verification results.
        """
        verification = {
            'timestamp': dt.utcnow().isoformat(),
            'architecture_valid': True,
            'parameter_check': True,
            'issues': [],
            'layer_info': self._get_layer_info()
        }
        
        try:
            # Get actual layer count
            actual_layers = len(list(self.layers))
            expected_layers = self._expected_layer_count
            
            # Detailed layer verification
            verification['layer_counts'] = {
                'actual': actual_layers,
                'expected': expected_layers,
                'linear_layers': sum(1 for layer in self.layers if isinstance(layer, nn.Linear)),
                'activation_layers': sum(1 for layer in self.layers if isinstance(layer, (nn.ReLU, nn.LeakyReLU, nn.ELU))),
                'dropout_layers': sum(1 for layer in self.layers if isinstance(layer, nn.Dropout))
            }
            
            # Check layer count
            if expected_layers != actual_layers:
                verification['architecture_valid'] = False
                verification['issues'].append(
                    f'Layer count mismatch: expected {expected_layers}, got {actual_layers}\n'
                    f'Layer breakdown: {verification["layer_counts"]}'
                )
            
            # Verify parameter shapes and values
            for name, param in self.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    verification['parameter_check'] = False
                    verification['issues'].append(f'Invalid values in parameter: {name}')
                    
        except Exception as e:
            verification['architecture_valid'] = False
            verification['issues'].append(f'Verification error: {str(e)}')
            
        return verification

    def save(self, save_path: str, include_verification: bool = True):
        """
        Save model state and configuration.
        
        Args:
            save_path (str): Path to save the model
            include_verification (bool): Whether to include model verification info
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        save_dict = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'model_name': self.model_name,
            'pytorch_version': torch.__version__,
            'expected_layer_count': self._expected_layer_count
        }
        
        if include_verification:
            save_dict['verification'] = self.verify_model_integrity()
            
        # Save main model file
        torch.save(save_dict, save_path)
        
        # Save human-readable config alongside
        config_path = save_path.replace('.pth', '_config.json')
        with open(config_path, 'w') as f:
            json.dump({k: v for k, v in save_dict.items() if k != 'model_state_dict'}, 
                     f, indent=4)
    
    @classmethod
    def load(cls, 
             load_path: str, 
             map_location: Optional[Union[str, torch.device]] = None,
             verify: bool = True,
             strict: bool = False) -> 'Predictor':
        """
        Load model from saved state with verification.
        
        Args:
            load_path (str): Path to saved model
            map_location: Optional device mapping for loaded model
            verify (bool): Whether to verify model integrity after loading
            strict (bool): If True, raises error on verification failure
            
        Returns:
            Predictor: Loaded model instance
        """
        try:
            save_dict = torch.load(load_path, map_location=map_location)
            
            # Verify saved config contains all required keys
            required_keys = {'model_state_dict', 'config'}
            if not all(k in save_dict for k in required_keys):
                raise ValueError(f"Saved model missing required keys: {required_keys - set(save_dict.keys())}")
            
            # Create new model instance with saved configuration
            model = cls(**save_dict['config'])
            
            # Load state dictionary
            model.load_state_dict(save_dict['model_state_dict'])
            
            if verify:
                verification = model.verify_model_integrity()
                if not verification['architecture_valid'] or not verification['parameter_check']:
                    issues = '\n'.join(verification['issues'])
                    message = f"Loaded model verification failed:\n{issues}\n"
                    message += "\nDetailed layer information:\n"
                    for layer in verification['layer_info']:
                        message += f"Layer {layer['index']}: {layer['type']} (params: {layer['params']})\n"
                    
                    if strict:
                        raise ValueError(message)
                    else:
                        print(f"Warning: {message}")
            
            return model
            
        except Exception as e:
            raise RuntimeError(f"Error loading model from {load_path}: {str(e)}")

#OLD w/broken load/save:
'''
class Predictor(nn.Module):
    """
    Neural network for prediction tasks with configurable architecture.
    
    Args:
        input_size (int): Number of input features
        num_classes (int): Number of output classes/values
        hidden_dim (int): Dimension of hidden layers
        num_layers (int): Number of hidden layers
        dropout_rate (float): Dropout probability between layers
        activation (str): Activation function to use ('relu', 'leaky_relu', or 'elu')
    """
    def __init__(
        self,
        input_size: int = 28 * 2 * 2,
        num_classes: int = 1,
        hidden_dim: int = 512,
        num_layers: int = 10,
        dropout_rate: float = 0.1,
        activation: Literal['relu', 'leaky_relu', 'elu'] = 'leaky_relu'
    ):
        super().__init__()
        
        # Store configuration
        self.model_name = "Predictor"
        self.input_size = input_size
        self.config = {
            'input_size': input_size,
            'num_classes': num_classes,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'dropout_rate': dropout_rate,
            'activation': activation
        }
        
        # Create activation function
        activation_map = {
            'relu': nn.ReLU,
            'leaky_relu': nn.LeakyReLU,
            'elu': nn.ELU
        }
        activation_fn = activation_map[activation]
        
        # Build network layers
        layers = []
        for i in range(num_layers):
            # Input layer
            if i == 0:
                layers.extend([
                    nn.Linear(input_size, hidden_dim),
                    activation_fn(inplace=True),
                    nn.Dropout(dropout_rate)
                ])
            # Output layer
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, num_classes))
            # Hidden layers
            else:
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    activation_fn(inplace=True),
                    nn.Dropout(dropout_rate)
                ])
        
        self.layers = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights using Xavier/Glorot initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def _check_input_shape(self, x: torch.Tensor) -> torch.Tensor:
        """
        Check and reshape input tensor if necessary.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Properly shaped input tensor
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension
            
        if x.dim() > 2:
            # Flatten all dimensions except batch dimension
            x = x.view(x.size(0), -1)
            
        if x.size(-1) != self.input_size:
            raise ValueError(
                f"Expected input size {self.input_size}, but got {x.size(-1)}. "
                f"Input shape: {tuple(x.shape)}"
            )
            
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size) or (input_size,)
            
        Returns:
            torch.Tensor: Output predictions
        """
        x = self._check_input_shape(x)
        return self.layers(x)

    @property
    def name(self) -> str:
        """Get model name."""
        return self.model_name

    def get_config(self) -> dict:
        """Get model configuration."""
        return self.config.copy()

    def save(self, save_path: str):
        """
        Save model state and configuration.
        
        Args:
            save_path (str): Path to save the model
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        save_dict = {
            'model_state_dict': self.state_dict(),
            'config': self.config
        }
        torch.save(save_dict, save_path)
    
    @classmethod
    def load(cls, load_path: str, map_location: Optional[Union[str, torch.device]] = None):
        """
        Load model from saved state.
        
        Args:
            load_path (str): Path to saved model
            map_location: Optional device mapping for loaded model
            
        Returns:
            Predictor: Loaded model instance
        """
        save_dict = torch.load(load_path, map_location=map_location)
        
        # Create new model instance with saved configuration
        model = cls(**save_dict['config'])
        
        # Load state dictionary
        model.load_state_dict(save_dict['model_state_dict'])
        
        return model
'''
    


class oldPredictor(nn.Module):
    """
    Prediction network
    """
    def __init__(self, input_size=28*2*2, num_classes=2, hidden_dim = 512, num_layers = 10):
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
    def save(self, save_loc):
        torch.save(self.layer,save_loc)
    def load(self,load_loc):
        self.layer = torch.load(load_loc)
        
def split_data(inputs, outputs, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split data into training, validation, and test sets.
    
    Parameters:
    inputs (torch.Tensor): Input tensor of shape (n_samples, ...)
    outputs (torch.Tensor): Output tensor of shape (n_samples, ...)
    train_ratio (float): Ratio of data to use for training (default: 0.7)
    val_ratio (float): Ratio of data to use for validation (default: 0.15)
    test_ratio (float): Ratio of data to use for testing (default: 0.15)
    seed (int): Random seed for reproducibility
    
    Returns:
    dict: Dictionary containing the split data
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"
    assert inputs.shape[0] == outputs.shape[0], "Number of inputs and outputs must match"
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    
    # Get total number of samples
    num_samples = inputs.shape[0]
    indices = torch.randperm(num_samples)
    
    # Calculate split points
    train_end = int(num_samples * train_ratio)
    val_end = train_end + int(num_samples * val_ratio)
    
    # Split indices
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    # Create splits
    train_data = {
        'inputs': inputs[train_indices],
        'outputs': outputs[train_indices]
    }
    
    val_data = {
        'inputs': inputs[val_indices],
        'outputs': outputs[val_indices]
    }
    
    test_data = {
        'inputs': inputs[test_indices],
        'outputs': outputs[test_indices]
    }
    
    split_info = {
        'train_size': len(train_indices),
        'val_size': len(val_indices),
        'test_size': len(test_indices)
    }
    
    return train_data, val_data, test_data, split_info
        
def train(predictor, train_data, nn_output, val_data, val_out, optimizer, device, 
          num_epochs=18, batch_size=100, show_progress=True, patience=5):
    criterion = nn.MSELoss()
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training Phase
        predictor.train()
        total_data_points = train_data.shape[0]
        num_it = total_data_points // batch_size
        epoch_losses = []
        
        # Shuffle training data
        shuffle_indices = torch.randperm(total_data_points)
        shuffled_data = train_data[shuffle_indices]
        shuffled_output = nn_output[shuffle_indices]
        
        clear_output(wait=True)
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training loop
        with tqdm(total=num_it, position=0, leave=True) as pbar:
            for it in range(num_it):
                optimizer.zero_grad()
                begin = it * batch_size
                end = min(begin + batch_size, total_data_points)
                
                context_inputs = shuffled_data[begin:end].flatten(start_dim=1).to(device)
                expected_outputs = shuffled_output[begin:end].unsqueeze(-1).to(device)
                
                outputs = predictor(context_inputs)
                loss = criterion(outputs, expected_outputs)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Invalid loss at iteration {it}")
                    continue
                    
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
                
                if show_progress:
                    pbar.update(1)
        
#         avg_train_loss = sum(epoch_losses) / len(epoch_losses)
#         avg_train_loss = np.mean(np.array(epoch_losses))
        avg_train_loss = np.mean(epoch_losses) if epoch_losses else float('nan')
        train_losses.append(avg_train_loss)
        
        # Validation Phase
        predictor.eval()
        val_epoch_losses = []
        with torch.no_grad():
            # Process validation data in batches
            val_data_points = val_data.shape[0]
            val_iterations = val_data_points // batch_size + (1 if val_data_points % batch_size != 0 else 0)
            
            for it in range(val_iterations):
                begin = it * batch_size
                end = min(begin + batch_size, val_data_points)
                
                val_inputs = val_data[begin:end].flatten(start_dim=1).to(device)
                val_expected = val_out[begin:end].unsqueeze(-1).to(device)
                
                val_outputs = predictor(val_inputs)
                val_loss = criterion(val_outputs, val_expected)
                val_epoch_losses.append(val_loss.item())
        
#         avg_val_loss = sum(val_epoch_losses) / len(val_epoch_losses)
#         avg_val_loss = np.mean(np.array(val_epoch_losses))
        avg_val_loss = np.mean(val_epoch_losses) if val_epoch_losses else float('nan')
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            # Validation phase
    
        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = predictor.state_dict().copy()  # Save a copy of the current state
            patience_counter = 0
        else:
            patience_counter += 1

        # Check early stopping condition
        if patience_counter >= patience:
            print(f"Early stopping triggered after epoch {epoch+1}")
            if best_model_state is not None:  # Safety check
                predictor.load_state_dict(best_model_state)  # Restore best model
                print("Loaded best model")
            break
    return train_losses, val_losses

def calculate_metrics(y_true, y_pred):

    # Convert inputs to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mse = np.mean((y_true - y_pred) ** 2)  # Mean Squared Error
    mae = np.mean(np.abs(y_true - y_pred))  # Mean Absolute Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error

    # R-squared calculation
    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r_squared': r_squared
    }

class SiPMSignalProcessor:
    def __init__(self, 
                 sampling_rate=40e9,  # 40 GHz sampling rate
                 tau_rise=1e-9,       # 1 ns rise time
                 tau_fall=10e-9,      # 50 ns fall time
                 window=200e-9,       # 200 ns time window
                 cfd_delay=5e-9,      # 5 ns delay for CFD
                 cfd_fraction=0.3):   # 30% fraction for CFD
        
        self.sampling_rate = sampling_rate
        self.tau_rise = tau_rise
        self.tau_fall = tau_fall
        self.window = window
        self.cfd_delay = cfd_delay
        self.cfd_fraction = cfd_fraction
        
        # Time array for single pulse shape
        self.time = np.arange(0, self.window, 1/self.sampling_rate)
        
        # Generate single pulse shape
        self.pulse_shape = self._generate_pulse_shape()
    
    def _generate_pulse_shape(self):
        """Generate normalized pulse shape for a single photon"""
        shape = (1 - np.exp(-self.time/self.tau_rise)) * np.exp(-self.time/self.tau_fall)
        return shape / np.max(shape)  # Normalize
    
    def generate_waveform(self, photon_times):
        """Generate waveform from list of photon arrival times"""
        # Initialize waveform array
        waveform = np.zeros_like(self.time)
        
        # Add pulse for each photon
        for t in photon_times:
            if 0 <= t < self.window:
                idx = int(t * self.sampling_rate)
                remaining_samples = len(self.time) - idx
                waveform[idx:] += self.pulse_shape[:remaining_samples]
        
        return self.time, waveform
    
    def integrate_charge(self, waveform, integration_start=0, integration_time=100e-9):
        """Integrate charge in specified time window"""
        start_idx = int(integration_start * self.sampling_rate)
        end_idx = int((integration_start + integration_time) * self.sampling_rate)
        
        # Integrate using trapezoidal rule
        charge = np.trapezoid(waveform[start_idx:end_idx], dx=1/self.sampling_rate)
        return charge
    def constant_threshold_timing(self,waveform,threshold):
        for i in range(len(self.time)):
            if(waveform[i] > threshold):
                return self.time[i]
        return -1
        
    def apply_cfd(self, waveform, use_interpolation=True):
        """Apply Constant Fraction Discrimination to the waveform.

        Parameters:
        -----------
        waveform : numpy.ndarray
            Input waveform to process
        use_interpolation : bool, optional
            If True, use linear interpolation for sub-sample precision
            If False, return the sample index of zero crossing
            Default is True

        Returns:
        --------
        tuple (numpy.ndarray, float)
            CFD processed waveform and the zero-crossing time in seconds.
            If use_interpolation is False, zero-crossing time will be aligned
            to sample boundaries.
        """
        # Calculate delay in samples
        delay_samples = int(self.cfd_delay * self.sampling_rate)

        # Create delayed and attenuated versions of the waveform
        delayed_waveform = np.pad(waveform, (delay_samples, 0))[:-delay_samples]
        attenuated_waveform = -self.cfd_fraction * waveform

        # Calculate CFD waveform
        cfd_waveform = delayed_waveform + attenuated_waveform

        # Find all zero crossings
        zero_crossings = np.where(np.diff(np.signbit(cfd_waveform)))[0]

        if len(zero_crossings) < 2:  # Need at least two crossings for valid CFD
            return cfd_waveform, None

        # Find the rising edge of the original pulse
        pulse_start = np.where(waveform > np.max(waveform) * 0.1)[0]  # 10% threshold
        if len(pulse_start) == 0:
            return cfd_waveform, None
        pulse_start = pulse_start[0]

        # Find the first zero crossing that occurs after the pulse starts
        valid_crossings = zero_crossings[zero_crossings > pulse_start]
        if len(valid_crossings) == 0:
            return cfd_waveform, None

        crossing_idx = valid_crossings[0]

        if not use_interpolation:
            # Simply return the sample index converted to time
            crossing_time = crossing_idx / self.sampling_rate
        else:
            # Use linear interpolation for sub-sample precision
            y1 = cfd_waveform[crossing_idx]
            y2 = cfd_waveform[crossing_idx + 1]

            # Calculate fractional position of zero crossing
            fraction = -y1 / (y2 - y1)

            # Calculate precise crossing time
            crossing_time = (crossing_idx + fraction) / self.sampling_rate

        return cfd_waveform, crossing_time


    def get_pulse_timing(self, waveform, threshold=0.1):
        """Get pulse timing using CFD method with additional validation.
        
        Parameters:
        -----------
        waveform : numpy.ndarray
            Input waveform to analyze
        threshold : float
            Minimum amplitude threshold for valid pulses (relative to max amplitude)
            
        Returns:
        --------
        float or None
            Timestamp of the pulse in seconds, or None if no valid pulse found
        """
        # Check if pulse amplitude exceeds threshold
        max_amplitude = np.max(waveform)
        if max_amplitude < threshold:
            return None
            
        # Apply CFD
        _, crossing_time = self.apply_cfd(waveform)
        
        return crossing_time