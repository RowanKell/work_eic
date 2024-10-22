import argparse
import os
from collections import defaultdict
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from tqdm import tqdm

from momentum_prediction_util import (
    Predictor,
    split_data,
    load_and_concatenate_tensors,
    filter_tensors_by_values,
    train
)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Training script for momentum prediction')
    parser.add_argument(
        '--inputTensorPath',
        type=str,
        required=True,
        help='Directory of input tensors'
    )
    parser.add_argument(
        '--outputTensorPath',
        type=str,
        required=True,
        help='Directory of output tensors'
    )
    return parser.parse_args()

def create_output_directories(base_path: str) -> Dict[str, str]:
    """Create output directories for plots and models."""
    paths = {
        'loss': os.path.join(base_path, "plots/momentum_prediction/loss/"),
        'rmse': os.path.join(base_path, "plots/momentum_prediction/RMSE/"),
        'model': os.path.join(base_path, "models/Momentum_prediction/")
    }
    
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    
    return paths

def setup_model(num_layers: int, device: torch.device) -> Tuple[nn.Module, torch.optim.Optimizer]:
    """Initialize the model and optimizer."""
    num_input_features_per_layer = 2 * 2  # two sipms, 2 features (charge, time)
    input_size = num_layers * num_input_features_per_layer
    hidden_dim_factor = 2.4
    hidden_dim = int(input_size * hidden_dim_factor)
    
    model = Predictor(
        input_size=input_size,
        num_classes=1,
        hidden_dim=hidden_dim,
        num_layers=20,
        dropout_rate=0.00075,
        activation="elu"
    ).to(device)
    
    learning_rate = 0.00094
    weight_decay = 7.2e-6
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return model, optimizer

def calculate_energy(momentum: np.ndarray, mass: float = 0.139570) -> np.ndarray:
    """Calculate energy from momentum and mass."""
    return np.sqrt(np.square(momentum) + np.square(mass))

def bin_data(real_e: np.ndarray, model_e: np.ndarray, 
            bin_range: Tuple[float, float] = (0.8, 10), 
            num_bins: int = 20) -> Tuple[List[List[float]], List[List[float]], np.ndarray]:
    """Bin the energy data for analysis."""
    bin_edges = np.linspace(bin_range[0], bin_range[1], num_bins + 1)
    binned_model_e = [[] for _ in range(num_bins)]
    binned_real_e = [[] for _ in range(num_bins)]
    
    for model_val, real_val in zip(model_e, real_e):
        for j in range(1, len(bin_edges)):
            if real_val < bin_edges[j]:
                binned_model_e[j - 1].append(model_val)
                binned_real_e[j - 1].append(real_val)
                break
    
    bin_centers = bin_edges[1:] - (bin_edges[1] - bin_edges[0]) / 2
    return binned_model_e, binned_real_e, bin_centers

def calculate_rmse_metrics(binned_real_e: List[List[float]], 
                         binned_model_e: List[List[float]], 
                         bin_centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate RMSE and relative RMSE for binned data."""
    rmse = np.zeros(len(bin_centers))
    rel_rmse = np.zeros(len(bin_centers))
    
    for i in range(len(bin_centers)):
        mse = np.mean((np.array(binned_real_e[i]) - np.array(binned_model_e[i])) ** 2)
        rmse[i] = np.sqrt(mse)
        rel_rmse[i] = rmse[i] / bin_centers[i]
    
    return rmse, rel_rmse

def plot_training_loss(loss_hist: List[float], val_hist: List[float], save_path: str):
    """Plot and save training loss history."""
    plt.figure()
    plt.plot(loss_hist, label="Train")
    plt.plot(val_hist, label="Validation")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig(os.path.join(save_path, "training_loss_optimized_more_data.pdf"))
    plt.close()

def plot_rmse_results(bin_centers: np.ndarray, rmse: np.ndarray, 
                     rel_rmse: np.ndarray, real_e: np.ndarray, 
                     model_e: np.ndarray, save_path: str):
    """Plot and save RMSE analysis results."""
    # Combined RMSE plots
    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    
    axs[0].scatter(bin_centers, rmse)
    axs[0].set_xlabel("Primary Energy")
    axs[0].set_ylabel("RMSE")
    
    axs[1].scatter(bin_centers, rel_rmse)
    axs[1].set_xlabel("Primary Energy")
    axs[1].set_ylabel("Relative RMSE")
    
    axs[2].scatter(real_e, model_e, alpha=0.1, color="blue")
    axs[2].plot([0, 10], [0, 10], color="red")
    axs[2].set_ylabel("Learned Energy")
    axs[2].set_xlabel("Real Energy")
    axs[2].set_ylim(0, 10)
    axs[2].set_xlim(0, 10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "rmse_analysis_optimized_params_more_data.pdf"))
    plt.close()
    
    # Separate relative RMSE plot
    plt.figure()
    plt.scatter(bin_centers, rel_rmse)
    plt.xlabel("Primary Energy")
    plt.ylabel("Relative RMSE")
    plt.title(r"Relative RMSE vs Energy for $\pi^-$")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "relative_rmse_optimized_params_more_data.pdf"))
    plt.close()

def main():
    # Setup
    args = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    paths = create_output_directories("/hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/")
    
    # Load and prepare data
    inputs = load_and_concatenate_tensors(args.inputTensorPath)
    outputs = load_and_concatenate_tensors(args.outputTensorPath)
    filtered_inputs, filtered_outputs = filter_tensors_by_values(inputs, outputs)
    
    train_data, val_data, test_data, _ = split_data(
        filtered_inputs, 
        filtered_outputs,
        train_ratio=0.8, 
        val_ratio=0.1, 
        test_ratio=0.1
    )
    
    # Train model
    model, optimizer = setup_model(num_layers=28, device=device)
    loss_hist, val_hist = train(
        model,
        train_data['inputs'],
        train_data['outputs'],
        val_data['inputs'],
        val_data['outputs'],
        optimizer,
        device,
        num_epochs=200,
        batch_size=32,
        patience=15
    )
    
    # Save model and plot training loss
    model.save(os.path.join(paths['model'], "model.pth"))
    plot_training_loss(loss_hist, val_hist, paths['loss'])
    
    # Evaluate on test data
    model_out = np.array([
        model(inputs.flatten().to(device)).detach().cpu().item()
        for inputs in test_data['inputs']
    ])
    real_out = test_data['outputs'].numpy()
    
    # Calculate energies
    model_e = calculate_energy(model_out)
    real_e = calculate_energy(real_out)
    
    # Bin data and calculate metrics
    binned_model_e, binned_real_e, bin_centers = bin_data(real_e, model_e)
    rmse, rel_rmse = calculate_rmse_metrics(binned_real_e, binned_model_e, bin_centers)
    
    # Plot results
    plot_rmse_results(bin_centers, rmse, rel_rmse, real_e, model_e, paths['rmse'])

if __name__ == "__main__":
    main()