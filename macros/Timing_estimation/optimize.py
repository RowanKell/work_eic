import matplotlib.pyplot as plot
import pandas as pd
import os
os.environ["DGLBACKEND"] = "pytorch"
import dgl
import torch
from dgl.data import DGLDataset
import matplotlib.pyplot as plot
import numpy as np
import torch.nn as nn
import torch
import itertools
import dgl.data
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
from dgl.nn import GraphConv,SumPooling,GINConv,AvgPooling
from tqdm import tqdm
import matplotlib.pyplot as plot
from datetime import datetime as datetime
current_date = datetime.now().strftime("%B_%d")
from torch.utils.data.sampler import SubsetRandomSampler
from scipy.spatial import ConvexHull
from GNN_util import process_df_vectorized,create_directory,HitDataset,create_fast_edge_lists,visualize_detector_graph,GIN,train_GNN,test_GNN,calculate_bin_rmse,delete_files_in_dir
import argparse
from scipy.optimize import curve_fit
from PIL import Image
import imageio
import optuna
import optuna.visualization as vis
from plotly.io import show

study_num = 3
n_trials = 100

Timing_path = "/hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/"

num_dfs = 200
inputDataPref =  "/hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/data/df/naive_CFD_Feb_10_50events_run_1_"
dfs = []
for i in range(num_dfs):
    try:
        new_df = pd.read_csv(f"{inputDataPref}{i}.csv")
    except FileNotFoundError as e:
        # Skip files that failed for some reason...
        # I think these files fail due to DCC issues?
        print(f"skipping file #{i}...")
        continue
    new_df["file_idx"] = i
    dfs.append(new_df)
if(len(dfs) > 1):
    data = pd.concat(dfs)
else:
    data = dfs[0]
    
coneAngle = 40
kNN_k = 6
training_batch_size = 20
model_path = f"/hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/models/validation_study/study_{study_num}/"

modified_df = process_df_vectorized(data, cone_angle_deg = coneAngle)    

filter_events_flag = True
connection_mode = "kNN"
dataset = HitDataset(modified_df,filter_events_flag,connection_mode = connection_mode,k = kNN_k)
print("Finished Creating HitDataset")

train_frac = 0.7
val_frac = 0.15
num_train = int(np.floor(len(dataset) * train_frac))
num_val = int(np.floor(len(dataset) * val_frac))
num_examples = len(dataset)

train_sampler = SubsetRandomSampler(torch.arange(num_train))
val_sampler = SubsetRandomSampler(torch.arange(num_train, num_val + num_train))
test_sampler = SubsetRandomSampler(torch.arange(num_val + num_train, num_examples))

train_dataloader = GraphDataLoader(
    dataset, sampler=train_sampler, batch_size=training_batch_size, drop_last=False
)

val_dataloader = GraphDataLoader(
    dataset, sampler=val_sampler, batch_size=training_batch_size, drop_last=False
)

test_dataloader = GraphDataLoader(
    dataset, sampler=test_sampler, batch_size=training_batch_size, drop_last=False
)

def objective(trial) -> float:
    #get suggestions for each hyperparameter
    MLP_hidden_dim = trial.suggest_int("MLP_hidden_dim",16,100)
    linear_capacity = trial.suggest_int("linear_capacity",3,8)
    n_linear_layers = trial.suggest_int("n_linear_layers",4,12)
    n_conv_layers = trial.suggest_int("n_conv_layers",1,6)
    lr = trial.suggest_float("lr",1e-4,5e-2,log = True)
    n_epochs = 300
    early_stopping_limit = 3
    model = GIN(
        dataset.dim_nfeats,
        MLP_hidden_dim,
        dataset.dim_event_feats,
        n_conv_layers = n_conv_layers, 
        n_linear_layers = n_linear_layers,
        linear_capacity = linear_capacity)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    trained_model, train_losses, val_losses, optimizer,best_epoch = train_GNN(
        model,
        optimizer,
        criterion, 
        train_dataloader, 
        val_dataloader, 
        n_epochs, 
        early_stopping_limit,
        frame_plot_path =  "",
        model_path = model_path,
        log_status = False)
    return min(val_losses).item()

create_directory(f"{Timing_path}optimization/study_{study_num}/")
# Step 2: Run the optimization
study = optuna.create_study(direction="minimize",study_name = f"Study_{study_num}",pruner=optuna.pruners.MedianPruner())  # Minimize MSE
study.optimize(objective, n_trials=n_trials) 

# Step 3: Print results
print("Best trial:")
print(study.best_trial)


# Create a report string
report = f"Study Name: {study.study_name}\n"
report += f"Number of Trials: {len(study.trials)}\n\n"

# Best trial information
best_trial = study.best_trial
report += f"Best Trial ID: {best_trial.number}\n"
report += f"Best Value: {best_trial.value}\n"
report += "Best Parameters:\n"
for key, value in best_trial.params.items():
    report += f"  {key}: {value}\n"
report += "\n"

# Save all trials
report += "All Trials:\n"
for trial in study.trials:
    report += f"Trial {trial.number}: Value={trial.value}, Params={trial.params}\n"

# Save to a text file
with open(f"{Timing_path}optimization/study_{study_num}/study_{study_num}_optuna_results.txt", "w") as f:
    f.write(report)
import joblib

joblib.dump(study,f"{Timing_path}optimization/study_{study_num}/study.pkl")

import optuna
import plotly
import numpy as np
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_slice
from optuna.visualization import plot_contour
from optuna.visualization import plot_param_importances


def visualize_optuna_study(study, exclude_trial_ids=None, objective_percentile=None):
    # Create a new study object with filtered trials
    figures = {}
    
    # Generate plots using filtered study
    figures['slice'] = plot_slice(study)
    figures['parallel_coordinate'] = plot_parallel_coordinate(study)
    figures['importance'] = plot_param_importances(study)
    figures['contour'] = plot_contour(study)
    
    return figures

def save_visualization(figure, filename):
    """
    Save a plotly figure to an HTML file.
    
    Parameters:
    figure (plotly.graph_objects.Figure): The plotly figure to save
    filename (str): Output filename (should end with .html)
    """
    figure.write_html(filename)
    

figures = visualize_optuna_study(study)

for plot_type, figure in figures.items():
    save_visualization(figure, f"{Timing_path}optimization/study_{study_num}/optuna_{plot_type}_plot.html")