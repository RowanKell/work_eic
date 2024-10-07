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
            energy_per_layer_particle = defaultdict(lambda: defaultdict(float))
            first_hit_per_layer_particle = defaultdict(dict)
            primary_momentum = (momentum_x_MC[event_idx][0],
                            momentum_y_MC[event_idx][0],
                            momentum_z_MC[event_idx][0])
            primary_momentum_mag = np.linalg.norm(primary_momentum)
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
    particle_indices = []

    final_event_indices = []
    final_layer_indices = []
    final_particle_indices = []
    
    momentum_list = []
    
    context_list = []
    running_pixel_idx = 0
    for event_idx, event_data in processed_data.items():
        for layer, layer_data in event_data.items():
            for particle_id, particle_data in layer_data.items():
                primary_momentum = particle_data["primary_momentum"]
                context = torch.tensor([particle_data['z_pos'], particle_data['theta'], particle_data['momentum']], dtype=torch.float32).repeat(particle_data['num_pixels'], 1)
                flattened_data.append(torch.tensor([particle_data['time'], particle_data['num_pixels']]).repeat(particle_data['num_pixels'],1))
                context_list.append(context)
                for pixel_repeat_idx in range(particle_data['num_pixels']):
                    final_event_indices.append(event_idx)
                    final_layer_indices.append(layer)
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
    nn_input = defaultdict(lambda: defaultdict(list))
    nn_output = defaultdict(lambda: defaultdict(list))
    print("Beginning reorganization process")
    t = tqdm(total = len(final_event_indices))
    for i, (event, layer, particle) in enumerate(zip(final_event_indices, final_layer_indices, final_particle_indices)):
        t.update(event)
        nn_input[event][layer].append(sampled_data[i])
        nn_output[event][layer].append(torch.Tensor([momentum_list[i]]))
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
    prediction_input = torch.ones(len(nn_input),28,2) * 9999
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
                photon_times = torch.tensor(sorted(nn_input[event_idx][layer])) * 10 **(-9)
                #get relative times
                min_time = photon_times[0]
                photon_times = photon_times - min_time
                
                #calculate time and charge
                time,waveform = processor.generate_waveform(photon_times)
                charge = processor.integrate_charge(waveform)
                timing = processor.cfd_timing(waveform)
                prediction_input
                if(not set_output):
                    prediction_output[curr_event_num] = nn_output[event_idx][layer][0]
                    set_output = True
                
                prediction_input[curr_event_num][layer][0] = charge
                prediction_input[curr_event_num][layer][1] = timing + min_time
            else:
                charge = 0
                timing = 9999
                prediction_input[curr_event_num][layer][0] = charge
                prediction_input[curr_event_num][layer][1] = timing

        curr_event_num += 1
    return prediction_input, prediction_output

class Predictor(nn.Module):
    """
    Prediction network
    """
    def __init__(self, input_size=280, num_classes=2, hidden_dim = 512, num_layers = 10):
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
def train(predictor, train_data,nn_output,optimizer,device, num_epochs = 18, batch_size = 100, show_progress = True):
    
    criterion = nn.MSELoss()
    predictor.train()
    total_data_points = train_data.shape[0]
    num_it = total_data_points // batch_size

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
                end = min((begin + batch_size),(total_data_points - begin))
                context_inputs = train_data[begin:end].flatten(start_dim = 1).to(device)
                expected_outputs = nn_output[begin:end].to(device)
                outputs = predictor(context_inputs)
                loss = criterion(outputs, expected_outputs)
                # Do backprop and optimizer step
                if ~(torch.isnan(loss) | torch.isinf(loss)):
                    if(loss > 200):
                        continue
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