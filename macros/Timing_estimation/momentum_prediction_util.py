import uproot
import numpy as np
import torch
from collections import defaultdict
from util import get_layer, theta_func,phi_func,create_layer_map,calculate_num_pixels_z_dependence
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
import pandas as pd

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


def new_prepare_nn_input(processed_data, normalizing_flow, batch_size=1024, device='cuda'):
    nn_input = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))))
    nn_output = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))))
    
    all_context = []
    all_time_pixels = []
    all_metadata = []
    num_pixel_list = ["num_pixels_high_z","num_pixels_low_z"]
    print("Processing data in new_prepare_nn_input...")
    for event_idx, event_data in tqdm(processed_data.items()):
        for stave_idx, stave_data in event_data.items():
            for layer_idx, layer_data in stave_data.items():
                for segment_idx, segment_data in layer_data.items():
                    for particle_id, particle_data in segment_data.items():
                        primary_momentum = particle_data["primary_momentum"]
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
                            all_metadata.extend([(event_idx,stave_idx, layer_idx,segment_idx, SiPM_idx, int(particle_id), particle_data['momentum'],particle_data['mc_hit_idx'],particle_data['pid'],particle_data['theta'],particle_data['phi'])] * particle_data[num_pixel_tag])

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
    for (event,stave, layer,segment, SiPM, particle, momentum,mc_hit_idx,pid,theta,phi), sample in zip(all_metadata, sampled_data):
        nn_input[event][stave][layer][segment][SiPM].append(sample)
#         print(f"types of nn_output vals: ({type(momentum)},{type(particle)},{type(mc_hit_idx)},{type(pid)},{type(theta)},{type(phi)})")
#         print(f"vals of nn_output: ({momentum},{particle},{mc_hit_idx},{pid},{theta},{phi})")
        nn_output[event][stave][layer][segment][SiPM].append(torch.tensor([momentum,particle,mc_hit_idx,pid,theta,phi]))
    end = time.time()
    print(f"reorganizing took {end - begin} seconds")
    return nn_input, nn_output



def prepare_prediction_input_pulse(nn_input,nn_output,pixel_threshold = 3):
    processor = SiPMSignalProcessor()
    
    #note - some events do not have dictionaries in nn_input due to being empty
    #need to skip over these and condense tensor
    out_columns = ['event_idx','stave_idx','layer_idx','segment_idx','trueid','truePID','P','Theta','Phi','Charge1','Time1','Charge2','Time2']
    running_index = 0
    rows = []
    curr_event_num = 0
    for event_idx in tqdm(list(nn_input)):
        event_input = []
        set_output = False
        stave_keys = nn_input[event_idx].keys()
        for stave_idx in stave_keys:
#             print(f"nn_input[event_idx][stave_idx]: {nn_input[event_idx][stave_idx]}")
            layer_keys = nn_input[event_idx][stave_idx].keys()
            for layer_idx in layer_keys:
                segment_keys = nn_input[event_idx][stave_idx][layer_idx].keys()
                for segment_idx in segment_keys:
                    charge_times = torch.tensor([[0.0,0.0],[0.0,0.0]])
                    SiPM_keys = nn_input[event_idx][stave_idx][layer_idx][segment_idx].keys()
                    set_event_details = False
                    trigger = False
                    for SiPM_idx in SiPM_keys:
#                         print(f"SiPM_idx: {SiPM_idx}")
                        photon_times = torch.tensor(sorted(nn_input[event_idx][stave_idx][layer_idx][segment_idx][SiPM_idx])) * 10 **(-9)
                        #get relative times
                        if(len(photon_times) > 0):
                            #calculate time and charge
                            time_arr,waveform = processor.generate_waveform(photon_times)
                            timing = processor.get_pulse_timing(waveform,threshold = pixel_threshold)
                            if(timing is not None):
                                curr_charge = processor.integrate_charge(waveform) * 1e6
                                curr_timing = timing * 1e8
                                
                                charge_times[SiPM_idx][0] = processor.integrate_charge(waveform) * 1e6
                                charge_times[SiPM_idx][1] = timing * 1e8
#                                 print(f"SiPM idx {SiPM_idx} triggered, (time,charge) : ({curr_timing},{curr_charge})")
                                trigger = True
                            else: #no trigger, don't set details yet
                                continue
                            if(not set_event_details):
                                P = nn_output[event_idx][stave_idx][layer_idx][segment_idx][SiPM_idx][0][0]
                                particle_idx = nn_output[event_idx][stave_idx][layer_idx][segment_idx][SiPM_idx][0][2]
                                pid = nn_output[event_idx][stave_idx][layer_idx][segment_idx][SiPM_idx][0][3]
                                theta = nn_output[event_idx][stave_idx][layer_idx][segment_idx][SiPM_idx][0][4]
                                phi = nn_output[event_idx][stave_idx][layer_idx][segment_idx][SiPM_idx][0][5]
                                set_event_details = True
                        else: #no photons, no data
                            continue
                    if(not set_event_details):
                        continue
                    if (not trigger):
                        continue;
                    new_row = { 
                       out_columns[0] : event_idx,
                       out_columns[1] : stave_idx,
                       out_columns[2] : layer_idx,
                       out_columns[3] : segment_idx,
                       out_columns[4] : particle_idx, 
                       out_columns[5] : pid, 
                       out_columns[6] : P, 
                       out_columns[7] : theta, 
                       out_columns[8] : phi, 
                       out_columns[9] : charge_times[0,0], 
                      out_columns[10] : charge_times[0,1], 
                      out_columns[11] : charge_times[1,0], 
                      out_columns[12] : charge_times[1,1]
                    }
                    rows.append(new_row)
                    running_index += 1
        curr_event_num += 1
    return pd.DataFrame(rows)




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