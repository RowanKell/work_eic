# import matplotlib.pyplot as plot
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
from dgl.nn import GraphConv,SumPooling,GINConv,AvgPooling,MaxPooling
from dgl.dataloading import GraphDataLoader
from tqdm import tqdm
import matplotlib.pyplot as plot
from datetime import datetime as datetime
current_date = datetime.now().strftime("%B_%d")
from torch.utils.data.sampler import SubsetRandomSampler
from scipy.spatial import ConvexHull
from GNN_util import process_df_vectorized,create_directory,HitDataset,create_fast_edge_lists,visualize_detector_graph,GIN,train_GNN,test_GNN,calculate_bin_rmse



def objective(num_files) -> tuple:

    dfs = []
    num_dfs = int(num_files)
    for i in range(num_dfs):
        try:
            new_df = pd.read_csv(f"/hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/data/df/naive_CFD_Feb_10_50events_run_1_{i}.csv")
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

    # Plot the data
    modified_df = process_df_vectorized(data, cone_angle_deg = 40)

    dataset = HitDataset(modified_df,True,connection_mode = "kNN",k = 6)
    print("done")
    print(len(dataset))
    
    train_frac = 0.7 
    val_frac = 0.15 
    test_frac = 0.15 
    num_train = int(np.floor(len(dataset) * train_frac))
    # num_train = 5
    num_val = int(np.floor(len(dataset) * val_frac))
    num_test = int(np.floor(len(dataset) * test_frac))
    num_examples = len(dataset)

    train_sampler = SubsetRandomSampler(torch.arange(num_train))
    val_sampler = SubsetRandomSampler(torch.arange(num_train, num_val + num_train))
    test_sampler = SubsetRandomSampler(torch.arange(num_val + num_train, num_examples))

    train_dataloader = GraphDataLoader(
        dataset, sampler=train_sampler, batch_size=20, drop_last=False
    )

    val_dataloader = GraphDataLoader(
        dataset, sampler=val_sampler, batch_size=20, drop_last=False
    )

    test_dataloader = GraphDataLoader(
        dataset, sampler=test_sampler, batch_size=20, drop_last=False
    )

    run_num = 1
    geometry_type = 2


    hidden_dim = 43
    linear_capacity = 5
    early_stopping_limit = 5
    n_conv_layers = 1
    n_linear_layers = 7
    lr = 4e-4

    model = GIN(dataset.dim_nfeats,hidden_dim,dataset.dim_event_feats,n_conv_layers = n_conv_layers, n_linear_layers = n_linear_layers,linear_capacity = linear_capacity)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model_path = f"/hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/models/{current_date}/dataset_size_study/"

    n_epochs = 300
    trained_model, train_losses, val_losses, optimizer,best_epoch = train_GNN(model,optimizer,criterion, train_dataloader, val_dataloader, n_epochs, early_stopping_limit, model_path = model_path, log_status=False)

#     plot.plot(train_losses,label = "train")
#     plot.title("Train and Val loss throughout training (log scale)")
#     plot.plot(val_losses, label = "test")
#     plot.legend()
#     plot.tight_layout()

    test_truths, test_preds, test_rmse = test_GNN(trained_model, test_dataloader)

#     plot.plot([0,5],[0,5])
#     plot.title("Test dataset results")
#     plot.scatter(test_truths,test_preds,alpha = 0.1)
#     plot.xlabel("truths")
#     plot.ylabel("preds")
#     plot.tight_layout()
    return (num_train,test_rmse)

options = np.linspace(1,600,5)
# options = [1]
create_directory("/hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/plots/dataset_size_study/")
num_examples_list = []
rmse_list = []
for i in range(len(options)):
    num_train_examples, rmse = objective(options[i])
    num_examples_list.append(num_train_examples)
    rmse_list.append(rmse)
rmse_list = np.array(rmse_list).flatten()
plot.plot(num_examples_list,rmse_list);
plot.xlabel("Number of training examples (higher is more expensive)")
plot.ylabel("Test MSE (lower is better)")
plot.title("Test MSE as function of dataset size")
plot.savefig("/hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/plots/dataset_size_study/study_2_5_points.pdf")