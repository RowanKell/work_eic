# import matplotlib.pyplot as plot
import pandas as pd
import os
os.environ["DGLBACKEND"] = "pytorch"
import dgl
import torch
from dgl.data import DGLDataset
import numpy as np
import torch.nn as nn
import torch
import itertools
import dgl.data
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
from tqdm import tqdm
import matplotlib.pyplot as plot
import numpy as np
from datetime import datetime as datetime
current_date = datetime.now().strftime("%B_%d")
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
from dgl.nn import GraphConv,SumPooling,GINConv,AvgPooling

def process_df_vectorized(df, cone_angle_deg=45):
    # Grab positions to use as center of cone
    event_references = (
        df.groupby(['event_idx', 'file_idx'])
        .last()[['first_hit_strip_x', 'first_hit_strip_y']]
        .reset_index()
    )
    #Save angle of center/reference
    event_references['reference_angle'] = np.degrees(
        np.arctan2(event_references['first_hit_strip_y'],event_references['first_hit_strip_x'])
    )
    
    # Add these new columns to original df
    df = df.merge(event_references[['event_idx', 'file_idx', 'reference_angle']], 
                  on=['event_idx', 'file_idx'], how='left')
    
    # Calc angle of each hit and how far off from center
    df['hit_angle'] = np.degrees(np.arctan2(df['strip_y'] * 10, df['strip_x'] * 10))
    df['angle_diff'] = np.abs(df['hit_angle'] - df['reference_angle'])
    
    # Handle the wraparound at ±180 degrees
    df['angle_diff'] = np.minimum(df['angle_diff'], 360 - df['angle_diff'])
    
    # Label hits outside of the cone as noise (-1)
    df['ModifiedTrueID'] = df['trueID']
    df.loc[df['angle_diff'] > cone_angle_deg, 'ModifiedTrueID'] = -1
    
    # Drop intermediate columns if necessary
    return df


class HitDataset(DGLDataset):
    def __init__(self, data, filter_events,max_distance = 200):
        self.data = data
        self.filter_events = filter_events
        self.max_distance = max_distance
        super().__init__(name = "KLM_reco")
    def process(self):
        events_group = self.data.groupby(["event_idx","file_idx"])
        self.labels = []
        self.graphs = []
        it_idx = 0
        for event_idx in events_group.groups:
            curr_event = events_group.get_group(event_idx)
            nhits = len(curr_event)
            '''FIRST FILTER FOR EVENTS'''
            if(self.filter_events):
                ModifiedTrueID_unique = np.array(curr_event['ModifiedTrueID'].unique())
                valid_ModifiedTrueID_unique = ModifiedTrueID_unique[ModifiedTrueID_unique != -1]
                #skip events with multiple valid trueIDs
                if(len(valid_ModifiedTrueID_unique) > 1):
                    continue
            
                #skip events with no valid ModififiedTrueIDs
                if(len(valid_ModifiedTrueID_unique) == 0):
                    continue
            # Skip graphs with only 1 hit (or 0)
            if(nhits <2):
                continue;
            else:
                x = curr_event['strip_x'].values
                y = curr_event['strip_y'].values

                # Create coordinate matrices
                x_diff = x[:, np.newaxis] - x[np.newaxis, :]  # Creates a matrix of all x differences
                y_diff = y[:, np.newaxis] - y[np.newaxis, :]  # Creates a matrix of all y differences

                # Compute distances in one go
                distances = np.sqrt(x_diff**2 + y_diff**2)

                # Create mask for valid edges (upper triangle only to avoid duplicates)
                upper_mask = (distances < self.max_distance) & (np.triu(np.ones_like(distances), k=1) > 0)

                # Get edge indices for upper triangle
                src_upper, dst_upper = np.where(upper_mask)

                # Create the bidirectional edges
                sources = np.concatenate([src_upper, dst_upper])
                destinations = np.concatenate([dst_upper, src_upper])
            g = dgl.graph((sources, destinations), num_nodes=nhits)
            #Want to predict momentum/energy
            label = torch.tensor(curr_event["P"].to_numpy()[0])
            self.labels.append(label)
            '''FLATTENED VERSION'''
            #THIS VERSION KEEPS FEATURES IN ONE DIMENSION
            feats = np.stack((
                curr_event["strip_x"].to_numpy(),curr_event["strip_y"].to_numpy(),
                curr_event["Time"].to_numpy(),
                curr_event["Charge"].to_numpy()
            ),axis = -1)
#             feats = curr_event["Charge"].to_numpy()
            g.ndata["feat"] = torch.tensor(feats)#.unsqueeze(-1)
    

            #add graph to dataset
            self.graphs.append(g)
            it_idx += 1
        self.dim_nfeats = self.graphs[0].ndata["feat"].shape[1]
        self.labels = torch.tensor(self.labels, dtype = torch.float32)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)
    

def create_fast_edge_lists(curr_event, max_distance):
    # Convert coordinates to numpy arrays for faster computation
    x = curr_event['strip_x'].values
    y = curr_event['strip_y'].values
    
    # Create coordinate matrices
    x_diff = x[:, np.newaxis] - x[np.newaxis, :]  # Creates a matrix of all x differences
    y_diff = y[:, np.newaxis] - y[np.newaxis, :]  # Creates a matrix of all y differences
    
    # Compute distances in one go
    distances = np.sqrt(x_diff**2 + y_diff**2)
    
    # Create mask for valid edges (upper triangle only to avoid duplicates)
    upper_mask = (distances < max_distance) & (np.triu(np.ones_like(distances), k=1) > 0)
    
    # Get edge indices for upper triangle
    src_upper, dst_upper = np.where(upper_mask)
    
    # Create the bidirectional edges
    sources = np.concatenate([src_upper, dst_upper])
    destinations = np.concatenate([dst_upper, src_upper])
    
    return sources, destinations
def visualize_detector_graph(curr_event, sources, destinations, max_edges=1000, figsize=(6, 6),max_angle):
    """
    Visualizes the detector hits and their connections.
    
    Parameters:
    curr_event (pd.DataFrame): DataFrame containing 'strip_x' and 'strip_y' columns
    sources (np.array): Array of source node indices
    destinations (np.array): Array of destination node indices
    max_edges (int): Maximum number of edges to plot to avoid overcrowding
    figsize (tuple): Figure size in inches
    """
    colors = curr_event['ModifiedTrueID'].apply(lambda x: 'red' if x == -1 else 'blue')
    sizes = curr_event['Charge'].apply(lambda x: x)
    # Create figure
    plot.figure(figsize=figsize)
    
    # Plot nodes (hits)
    plot.scatter(curr_event['strip_x'], curr_event['strip_y'], 
               c=colors, s=sizes * 1, alpha=0.4, label='Detector hits')
    
    # If there are too many edges, randomly sample them
    n_edges = len(sources)
    if n_edges > max_edges:
        idx = np.random.choice(n_edges, max_edges, replace=False)
        sources = sources[idx]
        destinations = destinations[idx]
    
    # Plot edges
    for src, dst in zip(sources, destinations):
        x1, y1 = curr_event.iloc[src][['strip_x', 'strip_y']]
        x2, y2 = curr_event.iloc[dst][['strip_x', 'strip_y']]
        plot.plot([x1, x2], [y1, y2], 'gray', alpha=0.1, linewidth=0.5)
    # Add reference angle and highlight region
    reference_angle = curr_event['reference_angle'].iloc[0]  # Assuming one reference angle per event
    radius = 250  # Radius of the detector
    
    # Calculate the coordinates for the line
    x_ref = radius * np.cos(np.radians(reference_angle))
    y_ref = radius * np.sin(np.radians(reference_angle))
#     plot.plot([0, x_ref], [0, y_ref], color='black', linewidth=3, label='Reference angle')
    
    # Highlight the 11-degree region
    theta_min = reference_angle - 40
    theta_max = reference_angle + 40
    
    # Calculate the coordinates for the line
    x_min = radius * np.cos(np.radians(theta_min))
    y_min = radius * np.sin(np.radians(theta_min))
    plot.plot([0, x_min], [0, y_min], color='orange', linewidth=1.5, label='Lower bound')
    
    # Calculate the coordinates for the line
    x_max = radius * np.cos(np.radians(theta_max))
    y_max = radius * np.sin(np.radians(theta_max))
    plot.plot([0, x_max], [0, y_max], color='orange', linewidth=1.5, label='Upper bound')
    
    # Add labels and title
    plot.xlabel('X Position')
    plot.ylabel('Y Position')
    plot.title(f'Detector Graph Visualization\n{len(curr_event)} nodes, {n_edges//2} edges')
    
    # Add text with statistics
    stats_text = f'Total nodes: {len(curr_event)}\n'
    stats_text += f'Total edges: {n_edges//2}\n'  # Divide by 2 because edges are bidirectional
    stats_text += f'Average degree: {n_edges/len(curr_event):.1f}'
    plot.text(0.02, 0.98, stats_text,
             transform=plot.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plot.grid(True, alpha=0.3)
    plot.axis('equal')
    plot.tight_layout()
#     plot.xlim(120,250)
#     plot.ylim(25,130)
    plot.xlim(-250,250)
    plot.ylim(-250,250)
#     plot.savefig("plots/GNN/graph_only.pdf")
    
    
class GIN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes=1):
        super(GIN, self).__init__()
        
        # Define the MLP for the GINConv layers
        self.mlp1 = nn.Sequential(
            nn.Linear(in_feats, h_feats),
            nn.ReLU(),
            nn.Linear(h_feats, h_feats)
        )
        
        self.mlp2 = nn.Sequential(
            nn.Linear(h_feats, h_feats),
            nn.ReLU(),
            nn.Linear(h_feats, num_classes)
        )
        
        # Define the GINConv layers
        self.conv1 = GINConv(self.mlp1)
        self.conv2 = GINConv(self.mlp2)
        
        # Dropout for regularization
#         self.dropout = nn.Dropout(0.05)
        
        # Graph pooling layer
        self.pool = SumPooling()

    def forward(self, g, in_feat):
        # Apply the first GINConv layer
        h = self.conv1(g, in_feat)
        h = F.relu(h)
#         h = self.dropout(h)
        
        # Apply the second GINConv layer
        h = self.conv2(g, h)
        
        # Pool the graph-level representation
        hg = self.pool(g, h)
        
        return hg
def train_GNN(model,optimizer,criterion, train_dataloader, val_dataloader, n_epochs,early_stopping_limit,run_num = "0"):
    create_directory(f"models/GNN_Energy_prediction/{current_date}/")
    val_mse = []
    val_mse_all = []
    train_losses = []
    train_losses_all = []
    early_stopping_dict = {
            "lowest_loss" : -1,
            "best_model_path" : "",
            "num_upticks" : 0
    }
    # 0: loss; 1: path; 2: # hits

    for epoch in range(n_epochs):
        # Training phase
        model.train()
        num_train_batches = 0
        epoch_train_losses = 0.0

        for batched_graph, labels in train_dataloader:
            pred = model(batched_graph, batched_graph.ndata["feat"].float())

            # Compute loss and update the model
            loss = criterion(pred, labels.unsqueeze(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_losses+=loss.detach()
            train_losses_all.append(loss.detach())
            num_train_batches += 1

        # Average RMSE for the epoch
        this_epoch_loss = epoch_train_losses / num_train_batches
        train_losses.append(this_epoch_loss)
        print(f"Epoch {epoch + 1}/{n_epochs} - Train loss:\t {this_epoch_loss:.4f}")

        # Testing phase
        model.eval()
        epoch_val_mse = 0.0
        num_val_batches = 0

        with torch.no_grad():  # Disable gradients for evaluation
            for batched_graph, labels in val_dataloader:
                pred = model(batched_graph, batched_graph.ndata["feat"].float())
                # Calculate RMSE for this batch
                batch_mse = criterion(pred, labels.unsqueeze(-1))
                epoch_val_mse += batch_mse
                num_val_batches += 1
                val_mse_all.append(batch_mse)

        # Average RMSE for the test set
        epoch_val_mse /= num_val_batches
        val_mse.append(epoch_val_mse)
        print(f"Epoch {epoch + 1}/{n_epochs} - Validation MSE:\t {epoch_val_mse:.4f}\n")
        if(epoch_val_mse.item() < early_stopping_dict["lowest_loss"] or early_stopping_dict["lowest_loss"] == -1):
            early_stopping_dict["lowest_loss"] = epoch_val_mse
            early_stopping_dict["best_model_path"] = f"models/GNN_Energy_prediction/{current_date}/run_{run_num}_events50k_lr0_001_hiddendim6_epoch{epoch}.pth"
            torch.save(model.state_dict(),early_stopping_dict["best_model_path"])
        elif(epoch_val_mse.item() > early_stopping_dict["lowest_loss"]):
            early_stopping_dict["num_upticks"] += 1
            print("Test loss increased, adding uptick")
        if(early_stopping_dict["num_upticks"] >= early_stopping_limit):
            # Stop training, load best model
            model.load_state_dict(torch.load(early_stopping_dict["best_model_path"]))
            torch.save(model.state_dict(),f"models/GNN_Energy_prediction/{current_date}/run_{run_num}_events50k_lr0_001_hiddendim6_best.pth")
            print("Stopping early, loading best model...")
            break
    return model, train_losses, val_mse

def test_GNN(model, test_dataloader):
    truths = []
    preds = []
    summed_sqe = 0.0
    num_predictions = 0
    with torch.no_grad():
        for batched_graph, labels in test_dataloader:
            graphs = dgl.unbatch(batched_graph)
            for i in range(len(graphs)):
                graph = graphs[i]
                label = labels[i].item()
                pred = model(graph, graph.ndata["feat"].float()).detach().numpy()
                summed_sqe += pow(pred - label,2)
                num_predictions += 1

                preds.append(pred)
                truths.append(label)
    mse = summed_sqe / num_predictions
    print(f"MSE: {mse[0][0]}")
    return truths, preds

def calculate_bin_rmse(test_dataloader, model, bin_width=0.5, bin_min=1.0, bin_max=3.0):
    # Calculate the bin centers
    bin_centers = np.arange(bin_min + bin_width / 2, bin_max, bin_width)
    
    # Initialize dictionaries to store squared errors and counts per bin
    summed_sqe_per_bin = {bin_center: 0.0 for bin_center in bin_centers}
    bin_counts = {bin_center: 0 for bin_center in bin_centers}
    
    # Initialize lists to store predictions and truths
    preds = []
    truths = []
    
    # Process each batch of test data
    with torch.no_grad():
        for batched_graph, labels in test_dataloader:
            graphs = dgl.unbatch(batched_graph)
            for i in range(len(graphs)):
                graph = graphs[i]
                label = labels[i].item()
                pred = model(graph, graph.ndata["feat"].float()).detach().numpy()

                # Store predictions and truths
                preds.append(pred)
                truths.append(label)
                
                # Calculate the squared error
                squared_error = (pred - label) ** 2
                
                # Find the bin this label falls into and update corresponding squared error and count
                for bin_center in bin_centers:
                    bin_min_edge = bin_center - bin_width / 2
                    bin_max_edge = bin_center + bin_width / 2
                    if bin_min_edge <= label < bin_max_edge:
                        summed_sqe_per_bin[bin_center] += squared_error
                        bin_counts[bin_center] += 1
                        break  # Only assign to one bin
    
    # Calculate RMSE for each bin
    rmse_per_bin = {}
    for bin_center in bin_centers:
        if bin_counts[bin_center] > 0:  # Avoid division by zero if no predictions fall into the bin
            rmse_value = np.sqrt(summed_sqe_per_bin[bin_center] / bin_counts[bin_center])
            rmse_per_bin[bin_center] = float(rmse_value[0,0])  # Ensure it's a scalar
        else:
            rmse_per_bin[bin_center] = float('nan')  # Assign NaN if no predictions fall into the bin
    
    return rmse_per_bin