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
from pathlib import Path

parser = argparse.ArgumentParser(description = 'Training GNN to predict KLM momentum')

parser.add_argument('--inputDataPref', type=str, default="NA",
                        help='Whole path of file excluding \"i.csv\" at the end') 
parser.add_argument('--numDfs', type=int, default=1,
                        help='Number of csv files to read into DataFrames') 
parser.add_argument('--coneAngle', type=int, default=40,
                        help='Angle for either side of cluster cone')
parser.add_argument('--kNNk', type=int, default=6,
                        help='k value for k Nearest Neighbors clustering for HitDataset graphs')
parser.add_argument('--trainFrac', type=float, default=0.7,
                        help='Fraction of events to use for training GNN')
parser.add_argument('--valFrac', type=float, default=0.15,
                        help='Fraction of events to use for validation when training GNN')
parser.add_argument('--runNum', type=int, default=0,
                        help='Run number for plotting, saving models')
parser.add_argument('--geometryType', type=int, default=1,
                        help='Refers to the number of scintillator layers per iron layers. The newer geometry uses 1 (use geometry_type 1), but the older uses 2 (bc of the superlayer design)')
parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate: multiplies the gradient to determine how fast to update weights in NN model. Higher will train faster, but may result in local minima. Lower will train slower but more stable.')
parser.add_argument('--MLPHiddenDim', type=int, default=32,
                        help='MLPs are used to mix the node features during the graph aggregation steps of GNN forward. This variable sets the MLP hidden dimension (and output dimension). All MLPs have same hidden dimension to allow for summing over all MLP representations at end of GNN layers.')
parser.add_argument('--trainingBatchSize', type=int, default=20,
                        help='GNN is trained on multiple graphs at once. Use this variable to set the number of graphs in each batch (more takes longer, but is more accurate).')
parser.add_argument('--modelPath', type=str, default="/hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/models/unsorted/",
                        help='GNN is trained on multiple graphs at once. Use this variable to set the number of graphs in each batch (more takes longer, but is more accurate).')
parser.add_argument('--nEpochs', type=int, default=300,
                        help='See earlyStoppingLimit - One epoch refers to training the GNN over the whole training dataset once. This variable will train the GNN over the training dataset N times (or until early stopping stops first). Typically nEpochs will be set to a very large number as early stopping should stop training before it reaches nEpochs.')
parser.add_argument('--earlyStoppingLimit', type=int, default=5,
                        help='See nEpochs - This value is used for early stopping. If the model doesn\'t improve (validation loss doesn\'t decrease) N times in a row then training will stop and the best model (lowest val loss) will be loaded. If early stopping does not occur, the model will train until nEpochs is reached.')
parser.add_argument('--lossPlotPath', type=str, default="",
                        help='Full path to save loss plot image. If not wanting to save plot path, then do not set this (leave default)')
parser.add_argument('--testPlotPath', type=str, default="",
                        help='Full path to save test plot image. If not wanting to save plot path, then do not set this (leave default)')
parser.add_argument('--resultsPlotPath', type=str, default="",
                        help='Full path to save results plot image. If not wanting to save plot path, then do not set this (leave default)')
parser.add_argument('--framePlotPath', type=str, default="",
                        help='Full path to save each frame of the training gif. The epoch number and the extension will be appended to end. If not wanting to save plot path, then do not set this (leave default)')
parser.add_argument('--gifPlotPath', type=str, default="",
                        help='Full path to save the compiled gif with frames of different epochs. Leave as default to not save a gif.')
parser.add_argument('--resultsFilePath', type=str, default="",
                        help='File to write the RMSE and A/sqrt(E) value to')
parser.add_argument('--runName', type=str, default="",
                        help='Name to use for saving files')
parser.add_argument('--deleteDfs', action=argparse.BooleanOptionalAction,
                        help='If true, train_GNN will first train a GNN and, if successful, it will delete the dfs created for the training run. Set this to be false if you are not generating the data at the same time.') 
args = parser.parse_args()
inputDataPref = args.inputDataPref
num_dfs = args.numDfs
coneAngle = args.coneAngle
kNN_k = args.kNNk
train_frac = args.trainFrac
val_frac = args.valFrac
run_num = args.runNum
geometry_type = args.geometryType
lr = args.lr
MLP_hidden_dim = args.MLPHiddenDim
training_batch_size = args.trainingBatchSize
n_epochs = args.nEpochs
early_stopping_limit = args.earlyStoppingLimit
frame_plot_path = args.framePlotPath
test_plot_path = args.testPlotPath
loss_plot_path = args.lossPlotPath
results_plot_path = args.resultsPlotPath
gif_plot_path = args.gifPlotPath
model_path = args.modelPath
results_file_path = args.resultsFilePath
run_name = args.runName
deleteDfs = args.deleteDfs

    

#check directories
path_list = [frame_plot_path,test_plot_path,loss_plot_path,results_plot_path,gif_plot_path,model_path,results_file_path]
for path in path_list:
    if(path != ''):
        if(".txt" in path):
            continue
        create_directory(path)

# Delete gif frames from file path if any exist
delete_files_in_dir(frame_plot_path)
        
# Example inputDataPref: /hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/data/df/Feb_8_50events_run_0_
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
    
modified_df = process_df_vectorized(data, cone_angle_deg = coneAngle)    

filter_events_flag = True
connection_mode = "kNN"
dataset = HitDataset(modified_df,filter_events_flag,connection_mode = connection_mode,k = kNN_k)
print("Finished Creating HitDataset")


# Split dataset into train, val and test sets
# Train is used to calculate and propagate gradients
# Val is used to check overfitting at end of each epoch
# Test is used to check performance after training
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

#Define model, optimizer, criterion for training
model = GIN(dataset.dim_nfeats,MLP_hidden_dim,dataset.dim_event_feats)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# Implementation for gif frame path: f"plots/training_gif_frames/run_{run_num}/frame{epoch}.jpeg"
trained_model, train_losses, val_losses, optimizer,best_epoch = train_GNN(model,optimizer,criterion, train_dataloader, val_dataloader, n_epochs, early_stopping_limit,frame_plot_path,model_path)

if(loss_plot_path != ""):
    loss_fig, loss_axs = plot.subplots(1,1)
    loss_axs.plot(train_losses,label = "train")
    loss_fig.suptitle(f"Train and Validation loss throughout training, run {run_num}")
    loss_axs.plot(val_losses, label = "test")
    loss_axs.legend()
    loss_fig.tight_layout()
    loss_fig.savefig(f"{loss_plot_path}{run_name}.jpeg")

test_truths, test_preds, test_rmse = test_GNN(trained_model, test_dataloader)
if(test_plot_path != ""):
    test_fig, test_axs = plot.subplots(1,1)
    test_axs.plot([0,5],[0,5])
    test_fig.suptitle("Test dataset results")
    test_axs.scatter(test_truths,test_preds,alpha = 0.1)
    test_axs.set_xlabel("True E (GeV)")
    test_axs.set_ylabel("Predicted E (GeV)")
    test_fig.tight_layout()
    test_fig.savefig(f"{test_plot_path}{run_name}.jpeg")
rmse_per_bin = calculate_bin_rmse(test_dataloader, model)
bin_centers = np.array(list(rmse_per_bin.keys()))
rmse = np.array(list(rmse_per_bin.values()))
rel_rmse = rmse / bin_centers

def func(x, A):
    return A / np.sqrt(x)

params, cov = curve_fit(func, bin_centers, rel_rmse)
fit_A_value = params[0]

#write the important objective values to a file so that AID2E can use
if(results_file_path != ""):
    if(os.path.isdir(results_file_path)):
        with open(f"{results_file_path}{run_name}.txt", "w") as f:
            f.write(f"{test_rmse.item()}")
            print(f"writing RMSE: {test_rmse.item()}")
#             f.write(f"{fit_A_value}\n{test_rmse.item()}")
    else:
        with open(f"{results_file_path}", "w") as f:
            f.write(f"{test_rmse.item()}")
            print(f"writing RMSE: {test_rmse.item()}")
#             f.write(f"{fit_A_value}\n{test_rmse.item()}")
        
        
        
x_fit = np.linspace(1, 3, 100)
y_fit = func(x_fit, params)
if(results_plot_path != ""):
    fig,axs = plot.subplots(1,3,figsize = (12,4))
    fig.suptitle("RMSE, relative RMSE, and scatter plot for Test set")
    axs[0].scatter(rmse_per_bin.keys(),rmse_per_bin.values())
    axs[0].set(xlabel="Energy",ylabel = "RMSE")
    axs[1].scatter(rmse_per_bin.keys(),np.array(list(rmse_per_bin.values())) / np.array(list(rmse_per_bin.keys())))
    axs[1].plot(x_fit,y_fit)
    axs[1].set(xlabel="Energy",ylabel = "Relative RMSE")
    axs[1].text(2,0.2,f"A: {params[0]:.2f}")
    axs[1].text(2,0.22,f"f(x) = A/sqrt(E)")
    axs[2].scatter(test_truths,test_preds,alpha = 0.2)
    axs[2].plot([0.5,3.5],[0.5,3.5],color = "red")
    axs[2].set(xlabel = "truths",ylabel = "preds")
    fig.tight_layout()
    plot.savefig(f"{results_plot_path}{run_name}.pdf")
    

if(gif_plot_path != ""):
    # path joining version for other paths
    num_files = len([name for name in os.listdir(frame_plot_path) if os.path.isfile(os.path.join(frame_plot_path, name))])
    # List of JPEG files (make sure they are in order)
    jpeg_files = []
    for i in range(num_files):
        jpeg_files.append(f"{frame_plot_path}epoch{i}.jpeg")
    jpeg_files.append(f"{frame_plot_path}epoch{best_epoch}.jpeg")

    # Load images
    images = [Image.open(f) for f in jpeg_files]

    # Save as a GIF
    imageio.mimsave(f"{gif_plot_path}{run_name}.gif", images, format="GIF", duration=5)  # duration in seconds

if(deleteDfs):
    #Only delete dfs if we successfully trained a model and saved it to the best_model.pth in model_path
    final_model_file = Path(f"{model_path}best_model.pth")    
    if(final_model_file.is_file()):
        print(f"successfully saved best model")
        for i in range(num_dfs):
            df_file = Path(f"{inputDataPref}{i}.csv")
            if(df_file.is_file()):
                df_file.unlink()
                print(f"deleted df file {inputDataPref}{i}.csv")
print("finished train_GNN")
