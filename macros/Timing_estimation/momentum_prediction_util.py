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
import normflows as nf
import datetime
from torch import nn
from scipy import signal
layer_map, super_layer_map = create_layer_map()

def new_prepare_nn_input(processed_data, normalizing_flow, batch_size=1024, device='cuda'):
    nn_input = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    nn_output = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    all_context = []
    all_time_pixels = []
    all_metadata = []
    
    print("Processing data...")
    for event_idx, event_data in tqdm(processed_data.items()):
        for layer, layer_data in event_data.items():
            for particle_id, particle_data in layer_data.items():
                primary_momentum = particle_data["primary_momentum"].item()
                base_context = torch.tensor([particle_data['z_pos'], particle_data['theta'], particle_data['momentum']], 
                                            dtype=torch.float32)
                base_time_pixels = torch.tensor([particle_data['time'], particle_data['num_pixels']], 
                                                dtype=torch.float32)
                
                for SiPM_idx in range(2):
                    z_pos = 1500 - particle_data['z_pos'] if SiPM_idx == 1 else particle_data['z_pos']
                    context = base_context.clone()
                    context[0] = z_pos
                    
                    all_context.append(context.repeat(particle_data['num_pixels'], 1))
                    all_time_pixels.append(base_time_pixels.repeat(particle_data['num_pixels'], 1))
                    all_metadata.extend([(event_idx, layer, SiPM_idx, particle_id, primary_momentum)] * particle_data['num_pixels'])

    all_context = torch.cat(all_context)
    all_time_pixels = torch.cat(all_time_pixels)
    
    print("Sampling data...")
    sampled_data = []
    for i in tqdm(range(0, len(all_context), batch_size)):
        batch_context = all_context[i:i+batch_size].to(device)
        batch_time_pixels = all_time_pixels[i:i+batch_size]
        
        with torch.no_grad():
            samples = abs(normalizing_flow.sample(num_samples=len(batch_context), context=batch_context)[0]).squeeze(1)
        
        adjusted_times = samples.cpu() + batch_time_pixels[:, 0]
        sampled_data.extend(adjusted_times)

    print("Reorganizing data...")
    for (event, layer, SiPM, particle, momentum), sample in zip(all_metadata, sampled_data):
        nn_input[event][layer][SiPM].append(sample)
        nn_output[event][layer][SiPM].append(torch.tensor([momentum]))

    return nn_input, nn_output

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

# In your main training script:
def train_neural_network(data_directory):
    # Load and concatenate the data
    training_data = load_and_concatenate_tensors(data_directory)

    # Your existing training code here
    # ...

# Example usage
if __name__ == "__main__":
    data_directory = "/path/to/your/tensor/files"
    train_neural_network(data_directory)

def process_root_file(file_path):
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
                    num_pixels = calculate_num_pixels_z_dependence(layer_particle_energy, hit_data["z_pos"])
#                     print(f"layer:\t\t{layer}\t|\tparticle id:\t{particle_id}\t|\tnum_pixels:\t{num_pixels}")
                    hit_data["num_pixels"] = int(np.floor(num_pixels))
                    hit_data["layer_energy"] = layer_particle_energy  # Store total layer energy for this particle
                    processed_data[event_idx][layer][particle_id.item()] = hit_data
    
    print("finished processing")
    return processed_data
def prepare_nn_input(processed_data, normalizing_flow, batch_size=1024):
    flattened_data = []
    event_indices = []
    layer_indices = []
    SiPM_indices = []
    particle_indices = []

    final_event_indices = []
    final_layer_indices = []
    final_SiPM_indices = []
    final_particle_indices = []
    
    momentum_list = []
    
    context_list = []
    running_pixel_idx = 0
    for event_idx, event_data in processed_data.items():
        for layer, layer_data in event_data.items():
            for particle_id, particle_data in layer_data.items():
                #Loop over twice for both SiPMs
                for SiPM_idx in range(2):
                    primary_momentum = particle_data["primary_momentum"]
                    if(SiPM_idx == 1):
                        z_pos = 1500 - particle_data['z_pos']
                    else:
                        z_pos = particle_data['z_pos']
                    context = torch.tensor([particle_data['z_pos'], particle_data['theta'], particle_data['momentum']], dtype=torch.float32).repeat(particle_data['num_pixels'], 1)
                    flattened_data.append(torch.tensor([particle_data['time'], particle_data['num_pixels']]).repeat(particle_data['num_pixels'],1))
                    context_list.append(context)
                    for pixel_repeat_idx in range(particle_data['num_pixels']):
                        final_event_indices.append(event_idx)
                        final_layer_indices.append(layer)
                        final_SiPM_indices.append(SiPM_idx)
                        final_particle_indices.append(particle_id)
                        momentum_list.append(primary_momentum.item())
    all_context = torch.cat(context_list).to(device)
    all_time_pixels = torch.cat(flattened_data)
    # Batch the flattened data
    max_its = int(np.ceil(all_context.shape[0] / batch_size))
    sampled_data = []
    print("Beginning sampling process")
    for batch_idx in tqdm(range(max_its)):
        begin = batch_idx * batch_size
        data_left = all_context.shape[0] - (batch_idx * batch_size)
        end = min(begin + batch_size,begin + data_left)
        add_times = all_time_pixels[begin:end]
        context_batch = all_context[begin:end].to(device)
        with torch.no_grad():
            samples = abs(normalizing_flow.sample(num_samples=context_batch.shape[0], context=context_batch)[0]).squeeze(1)
        adjusted_times = samples.detach().cpu() + add_times[:,0]
        sampled_data.extend(adjusted_times)
    # Reorganize sampled data
    nn_input = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    nn_output = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    print("Beginning reorganization process")
    t = tqdm(total = len(final_event_indices))
    for i, (event, layer, SiPM, particle) in enumerate(zip(final_event_indices, final_layer_indices, final_SiPM_indices,final_particle_indices)):
        t.update(event)
        nn_input[event][layer][SiPM].append(sampled_data[i])
        nn_output[event][layer][SiPM].append(torch.Tensor([momentum_list[i]]))
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
def prepare_prediction_input_pulse(nn_input, nn_output):
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
                    photon_times = torch.tensor(sorted(nn_input[event_idx][layer][SiPM_idx])) * 10 **(-9)
                    #get relative times
                    min_time = photon_times[0]
                    photon_times = photon_times - min_time

                    #calculate time and charge
                    time,waveform = processor.generate_waveform(photon_times)
                    charge = processor.integrate_charge(waveform)
                    timing = processor.cfd_timing(waveform)
                    prediction_input
                    if(not set_output):
                        prediction_output[curr_event_num] = nn_output[event_idx][layer][SiPM_idx][0]
                        set_output = True

                    prediction_input[curr_event_num][layer][SiPM_idx][0] = charge * 1e6
                    prediction_input[curr_event_num][layer][SiPM_idx][1] = (timing + min_time) * 1e8
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
                    num_pixels = calculate_num_pixels_z_dependence(layer_particle_energy, hit_data["z_pos"])
#                     print(f"layer:\t\t{layer}\t|\tparticle id:\t{particle_id}\t|\tnum_pixels:\t{num_pixels}")
                    hit_data["num_pixels"] = int(np.floor(num_pixels))
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
    
    print("Processing data...")
    for event_idx, event_data in tqdm(processed_data.items()):
        for layer, layer_data in event_data.items():
            for particle_id, particle_data in layer_data.items():
                primary_momentum = particle_data["primary_momentum"].item()
                base_context = torch.tensor([particle_data['z_pos'], particle_data['theta'], particle_data['momentum']], 
                                            dtype=torch.float32)
                base_time_pixels = torch.tensor([particle_data['time'], particle_data['num_pixels']], 
                                                dtype=torch.float32)
                
                for SiPM_idx in range(2):
                    z_pos = 1500 - particle_data['z_pos'] if SiPM_idx == 1 else particle_data['z_pos']
                    context = base_context.clone()
                    context[0] = z_pos
                    
                    all_context.append(context.repeat(particle_data['num_pixels'], 1))
                    all_time_pixels.append(base_time_pixels.repeat(particle_data['num_pixels'], 1))
                    all_metadata.extend([(event_idx, layer, SiPM_idx, particle_id, primary_momentum,particle_data['mc_hit_idx'],particle_data['pid'],particle_data['theta'],particle_data['phi'])] * particle_data['num_pixels'])

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
def prepare_prediction_input_pulse_for_greg(nn_input, nn_output):
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
                    min_time = photon_times[0]
                    photon_times = photon_times - min_time

                    #calculate time and charge
                    time,waveform = processor.generate_waveform(photon_times)
                    charge_times[SiPM_idx][0] = processor.integrate_charge(waveform) * 1e6
                    charge_times[SiPM_idx][1] = (processor.cfd_timing(waveform) + min_time) * 1e8
                    
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
        
        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = predictor.state_dict()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered after epoch {epoch+1}")
            predictor.load_state_dict(best_model_state)  # Restore best model
            print("loaded best model")
            break
    
    print('Finished Training')
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
                 tau_fall=50e-9,      # 50 ns fall time
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
    
    def cfd_timing(self, waveform):
        """Implement Constant Fraction Discrimination timing"""
        # Create delayed and attenuated versions
        delay_samples = int(self.cfd_delay * self.sampling_rate)
        delayed = np.roll(waveform, delay_samples)
        attenuated = waveform * self.cfd_fraction
        
        # CFD waveform
        cfd_signal = attenuated - delayed
        
        # Find zero crossing
        zero_crossings = np.where(np.diff(np.signbit(cfd_signal)))[0]
        
        if len(zero_crossings) > 0:
            # Linear interpolation for more precise timing
            idx = zero_crossings[0]
            t1, t2 = self.time[idx], self.time[idx+1]
            v1, v2 = cfd_signal[idx], cfd_signal[idx+1]
            
            # Time at zero crossing
            zero_time = t1 - v1 * (t2 - t1) / (v2 - v1)
            return zero_time
        else:
            return None