import optuna
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from datetime import datetime
import json
import os
from typing import Dict, Any, Literal, Optional, Union

from momentum_prediction_util import (
    Predictor,
    split_data,
    load_and_concatenate_tensors,
    filter_tensors_by_values
)



class HyperparameterOptimization:
    def __init__(self, base_path: str, input_tensor_path: str, output_tensor_path: str):
        self.base_path = base_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load and prepare data
        inputs = load_and_concatenate_tensors(input_tensor_path)
        outputs = load_and_concatenate_tensors(output_tensor_path)
        filtered_inputs, filtered_outputs = filter_tensors_by_values(inputs, outputs)
        
        # Split data with fixed validation set for consistent evaluation
        self.train_data, self.val_data, self.test_data, _ = split_data(
            filtered_inputs, 
            filtered_outputs,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1
        )
        
        # Create study directory
        self.study_dir = os.path.join(
            self.base_path,
            f"hyperparameter_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.study_dir, exist_ok=True)

    def create_model(self, trial: optuna.Trial) -> nn.Module:
        """Create model with trial-suggested hyperparameters."""
        num_layers = 28  # Fixed by your problem
        num_input_features_per_layer = 2 * 2
        input_size = num_layers * num_input_features_per_layer
        
        # Suggest hyperparameters for network architecture
        n_layers = trial.suggest_int('n_layers', 5, 30)
        hidden_dim_factor = trial.suggest_float('hidden_dim_factor', 2.0, 10.0)
        hidden_dim = int(input_size * hidden_dim_factor)
        
        # Additional architecture parameters
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
        activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'elu'])
        
        model = Predictor(
            input_size=input_size,
            num_classes=1,
            hidden_dim=hidden_dim,
            num_layers=n_layers,
            dropout_rate=dropout_rate,
            activation=activation
        ).to(self.device)
        
        return model

    def create_optimizer(self, trial: optuna.Trial, model: nn.Module) -> torch.optim.Optimizer:
        """Create optimizer with trial-suggested hyperparameters."""
        lr = trial.suggest_float('lr', 1e-6, 1e-3, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'RMSprop'])
        optimizer_class = getattr(torch.optim, optimizer_name)
        
        return optimizer_class(model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_and_evaluate(self, trial: optuna.Trial) -> float:
        """Train model and return validation loss."""
        # Create model and optimizer
        model = self.create_model(trial)
        optimizer = self.create_optimizer(trial, model)
        
        # Training hyperparameters
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        num_epochs = trial.suggest_int('num_epochs', 100, 300)
        
        # Early stopping parameters
        patience = trial.suggest_int('patience', 5, 20)
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_losses = []
            
            # Create batches
            indices = torch.randperm(len(self.train_data['inputs']))
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                inputs = self.train_data['inputs'][batch_indices].to(self.device)
                targets = self.train_data['outputs'][batch_indices].to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = nn.MSELoss()(outputs.squeeze(-1), targets)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation phase
            model.eval()
            val_losses = []
            with torch.no_grad():
                for i in range(0, len(self.val_data['inputs']), batch_size):
                    inputs = self.val_data['inputs'][i:i+batch_size].to(self.device)
                    targets = self.val_data['outputs'][i:i+batch_size].to(self.device)
                    
                    outputs = model(inputs)
                    loss = nn.MSELoss()(outputs.squeeze(-1), targets)
                    val_losses.append(loss.item())
            
            avg_val_loss = np.mean(val_losses)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # Save best model
                model_path = os.path.join(self.study_dir, f"model_trial_{trial.number}.pth")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'trial_params': trial.params,
                    'val_loss': best_val_loss
                }, model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
            
            # Report intermediate values
            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return best_val_loss

    def save_study_results(self, study: optuna.Study):
        """Save study results and best parameters."""
        results = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'best_trial': study.best_trial.number,
            'n_trials': len(study.trials),
            'datetime': datetime.now().isoformat()
        }
        
        # Save results
        results_path = os.path.join(self.study_dir, 'study_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        # Save optimization history plot
        try:
            import plotly
            fig = optuna.visualization.plot_optimization_history(study)
            fig.write_html(os.path.join(self.study_dir, 'optimization_history.html'))
            
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_html(os.path.join(self.study_dir, 'parallel_coordinate.html'))
            
            fig = optuna.visualization.plot_param_importances(study)
            fig.write_html(os.path.join(self.study_dir, 'param_importances.html'))
        except Exception as e:
            print(f"Warning: Could not create visualization plots: {e}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter optimization for momentum prediction')
    parser.add_argument('--inputTensorPath', type=str, required=True,
                      help='Directory of input tensors')
    parser.add_argument('--outputTensorPath', type=str, required=True,
                      help='Directory of output tensors')
    parser.add_argument('--basePath', type=str, required=True,
                      help='Base directory for saving study results')
    parser.add_argument('--n_trials', type=int, default=100,
                      help='Number of optimization trials to run')
    
    args = parser.parse_args()
    
    # Create optimizer instance
    optimizer = HyperparameterOptimization(
        args.basePath,
        args.inputTensorPath,
        args.outputTensorPath
    )
    
    # Create study
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(),
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Run optimization
    study.optimize(optimizer.train_and_evaluate, n_trials=args.n_trials)
    
    # Save results
    optimizer.save_study_results(study)
    
    print("Best trial:")
    print(f"  Value: {study.best_value}")
    print(f"  Params:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    main()