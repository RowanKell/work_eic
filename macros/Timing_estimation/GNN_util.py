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
from dgl.nn import GraphConv,SumPooling,GINConv,AvgPooling
from tqdm import tqdm
import matplotlib.pyplot as plot
import numpy as np
from datetime import datetime as datetime
from pathlib import Path
current_date = datetime.now().strftime("%B_%d")
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
from dgl.nn import GraphConv,SumPooling,GINConv,AvgPooling

def delete_files_in_dir(directory):
    for file_path in Path(directory).iterdir():
        if file_path.is_file():
            file_path.unlink()
def process_df_vectorized(df, cone_angle_deg=45):
    """
    Check if hits are within "cluster", and reject any that are outside.
    
    This function creates a "cone" around the first hit for each strip 
    and excludes all hits that are outside of this cone (in x, y).
    This works as a sort of clustering. This function also stores a value
    "ModifiedTrueID" for each hit, which will be -1 if the hit is outside of the cone.
    
    ...
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing hit information for entire dataset.
    cone_angle_deg : float
        Total angle of cone centered on the first hit.
    """
    
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
    df['hit_angle'] = np.degrees(np.arctan2(df['strip_y'], df['strip_x']))
    df['angle_diff'] = np.abs(df['hit_angle'] - df['reference_angle'])
    
    # Handle the wraparound at ±180 degrees
    df['angle_diff'] = np.minimum(df['angle_diff'], 360 - df['angle_diff'])
    
    # Label hits outside of the cone as noise (-1)
    df['ModifiedTrueID'] = df['trueID']
    df.loc[df['angle_diff'] > cone_angle_deg, 'ModifiedTrueID'] = -1
    
    # Drop intermediate columns if necessary
    return df

class HitDataset(DGLDataset):
    """
    A dataset class for predicting PID or energy of particles with GNNs
    
    ...
    
    Attributes
    ----------
    data : pd.DataFrame
        dataframe (usually modified by process_df_vectorized()) containing
        rows of hits, generated by momentum_prediction_util.newer_prepare_nn_input()
    filter_events : bool
        Flag to tell object if we want to filter out events or not. If True, then process will
        only create graphs from events where there are no more than 1 trueID associated with
        each hit. This part only matters if we generate multiple primary particles in ddsim.
        ALSO: if True, then process removes hits from outside of cluster cone.
    max_distance : float
        maximum distance between two nodes that will be connected by edges
    event_data : torch.tensor
        deprecated (not used)
    connection_mode : str
        Either "kNN" or "max distance". If kNN, builds edges by connecting 
        k nearest neighbors. If "max distance", then connects all nodes within
        max_distance.
    k : int
        k for kNN edge connecting.
    dfs : list
        list of event dataframes (unused)
    mass_dict : dict
        Dictionary for getting mass from PDG value
    
    Methods
    -------
    get_max_distance_edges(curr_event) -> sources, destinations
        creates edge matrix by connecting all nodes within self.max_distance from each other.
        Returns the source and destination lists.
    get_knn_edges(curr_event) -> sources, destinations
        creates edge matrix by connecting each node to its self.k nearest neighbors
        Resturns the source and destinations lists
    process()->
        required method for DGLDataset. Takes the input dataframe and creates a set of graphs
        and labels that can be accessed via a dataloader.
    """
    
    def __init__(self, data, filter_events,connection_mode = "kNN",max_distance = 0.5,k = 6):
        """
        Parameters
        ----------
        data: pd.DataFrame
            dataframe (usually modified by process_df_vectorized()) containing
            rows of hits, generated by momentum_prediction_util.newer_prepare_nn_input()
        filter_events : bool
            Flag to tell object if we want to filter out events or not. If True, then process will
            only create graphs from events where there are no more than 1 trueID associated with
            each hit. This part only matters if we generate multiple primary particles in ddsim.
            ALSO: if True, then process removes hits from outside of cluster cone.
        max_distance : float
            maximum distance between two nodes that will be connected by edges
        connection_mode : str
            Either "kNN" or "max distance". If kNN, builds edges by connecting 
            k nearest neighbors. If "max distance", then connects all nodes within
            max_distance.
        k : int
            k for kNN edge connecting.
        """
        
        self.data = data
        self.filter_events = filter_events
        self.max_distance = max_distance
        self.event_data = torch.tensor([])
        self.connection_mode = connection_mode
        self.k = k
        self.dfs = []
        self.mass_dict = {
            130 : 0.497611,
            2112  : 0.939565,
            211 : 0.139570,
            -211 : 0.139570,
            13 : 0.10566
                         }
        super().__init__(name = "KLM_reco")
    def get_max_distance_edges(self,curr_event):
        """
        Create source and destination lists for given x and y of each strip in event.
        Connects each node that is within self.max_distance of each other node.
        The edges of the graph are defined as being between
        source[i] and desination[i] for all i in the range(len(source)). 
        
        Parameters
        ----------
        curr_event : pd.DataFrame
            dataframe of hits in an event
        
        Returns
        -------
        sources : list
            list of indices of source nodes
        destinations : list
            list of indices of destination nodes
        """
        
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
        return sources,destinations
    def get_knn_edges(self,curr_event):
        """
        Create source and destination lists for given x and y of each strip in event
        Connects considers each node and connects to the nearest self.k nodes.
        The edges of the graph are defined as being between
        source[i] and desination[i] for all i in the range(len(source)). 
        
        Parameters
        ----------
        curr_event : pd.DataFrame
            dataframe of hits in an event
        
        Returns
        -------
        sources : list
            list of indices of source nodes
        destinations : list
            list of indices of destination nodes
        """
        
        x = curr_event['strip_x'].values
        y = curr_event['strip_y'].values
        n = len(x)

        # The first notation with "np.newaxis" is the same as tensor.unsqueeze(-1)
        # It puts each value in its own dimension, like
        # x = np.arrayy([[a],[b],[c]]), so size is (N,1) rather than (N)
        # The second notation (x[np.newaxis,:]) just puts the array in another array so that the size is (1,N
        # rather than (N)
        x_diff = x[:, np.newaxis] - x[np.newaxis, :]
        y_diff = y[:, np.newaxis] - y[np.newaxis, :]
        
        # distances has shape (N,N) - matrix where the diagonal is 0, each entry is the distance between
        # the ith (column idx) node and the jth (row idx) node
        distances = np.sqrt(x_diff**2 + y_diff**2)

        # Get the indices of the k nearest neighbors for each node (excluding self-connections)
        # argsort sorts each row by the distance and returns the sorted indices
        # We use the [:,1:k+1] to take the first k indices besides the lowest (which is the diagonal self connection)
        knn_indices = np.argsort(distances, axis=1)[:, 1:self.k+1]

        # Create source and destination lists
        sources = np.concatenate([np.repeat(np.arange(n), self.k),knn_indices.flatten()])
        destinations = np.concatenate([knn_indices.flatten(),np.repeat(np.arange(n), self.k)])

        return sources, destinations
        
    def process(self):
        """
        Required function for DGLDataset. Runs on construction of dataset.
        Takes input data and creates graphs and labels that can be used
        to train GNN
        """
        
        events_group = self.data.groupby(["event_idx","file_idx"])
        self.labels = torch.tensor([])
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
                    print("Too many valid ModifiedTrueID, skipping...")
                    continue
            
                #skip events with no valid ModififiedTrueIDs
#                 if(len(valid_ModifiedTrueID_unique) == 0):
#                     print("No valid ModifiedTrueIDs, skipping...")
#                     continue
                # Remove rows that are hits outside of the cone
                curr_event = curr_event[curr_event.ModifiedTrueID != -1]
                nhits = len(curr_event)
            # Skip graphs with only 1 hit (or 0)
            if(nhits <2):
#                 print("only 1 hit, skipping...")
                continue;
            elif(nhits <self.k):
                sources = np.concatenate([np.repeat(np.arange(nhits),nhits),np.tile(np.arange(nhits),nhits)])
                destinations = np.concatenate([np.tile(np.arange(nhits),nhits),np.repeat(np.arange(nhits),nhits)])
            else:
                if(self.connection_mode == "max distance"):
                    sources, destinations = self.get_max_distance_edges(curr_event)
                elif(self.connection_mode == "kNN"):
                    sources, destinations = self.get_knn_edges(curr_event)
            g = dgl.graph((sources, destinations), num_nodes=nhits)
            #Want to predict energy
            try:
                mass = self.mass_dict[curr_event["truePID"].to_numpy()[0]]
            except Exception as e:
                truePID = curr_event["truePID"].to_numpy()[0]
                print(f"Exception: {e}")
                print(f"Particle with truePID of {truePID} not in dictionary. Skipping...")
                continue
            momentum = curr_event["P"].to_numpy()[0]
            energy = np.sqrt(mass**2 + momentum**2)
            label = torch.tensor(energy)
            strip_x = (curr_event["strip_x"].to_numpy() / 3000)
            strip_y = (curr_event["strip_y"].to_numpy() / 3000)
            radial_distance = torch.tensor(np.sqrt( strip_x** 2 + strip_y ** 2))
            '''VERSION LABEL INCLUDING EVENT FEATURES'''
            # Since this is the version with both SiPM in one hit/node, we have 2 times and charges
            # I hope that doing this will avoid making the NN learn that two hits at the same position are 
            # closely related
            feats = np.stack((
                strip_x,strip_y, radial_distance,
                curr_event["Time0"].to_numpy() / 5,
                curr_event["Charge0"].to_numpy(),
                curr_event["Time1"].to_numpy() / 5,
                curr_event["Charge1"].to_numpy(),
                curr_event["stave_idx"].to_numpy(),
                curr_event["layer_idx"].to_numpy(),
                curr_event["segment_idx"].to_numpy()
            ),axis = -1)
            g.ndata["feat"] = torch.tensor(feats)#.unsqueeze(-1)
            self.dfs.append(curr_event)
            # Sort hits by time

            # Basic features
            total_charge = curr_event['Charge1'].sum() + curr_event['Charge0'].sum()
            max_charge = max([curr_event['Charge1'].max(),curr_event['Charge0'].max()])
            n_hits = len(curr_event)

            # Spatial features
#             hit_coords = curr_event[['strip_x', 'strip_y']].values
            # Feature vector for this event
            event_features = torch.from_numpy(np.stack((label,
                total_charge,
                max_charge,
                n_hits

                ),axis = -1))
            if(self.labels.shape[0] == 0):
                self.labels = event_features
            else:
                self.labels = torch.vstack((self.labels,event_features))
            #add graph to dataset
            self.graphs.append(g)
            it_idx += 1
        self.dim_nfeats = self.graphs[0].ndata["feat"].shape[1]
        self.dim_event_feats = self.labels.shape[1] - 1
        self.labels = self.labels.clone().detach().float()

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)
    

def create_fast_edge_lists(curr_event, max_distance):
    """
    UNUSED AS OF MAY 2 -Rowan
    """
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
def visualize_detector_graph(dataset,graph_idx = 0, max_edges=1000, figsize=(6, 6)):
    """
    Visualizes the detector hits and their connections. Not used for actual computing,
    just for looking and debugging
    
    Parameters:
    curr_event : pd.DataFrame 
        DataFrame containing 'strip_x' and 'strip_y' columns
    sources : np.array
        Array of source node indices
    destinations : np.array
        Array of destination node indices
    max_edges : int
        Maximum number of edges to plot to avoid overcrowding
    figsize : tuple 
        Figure size in inches
    """
    
    fig, axs = plot.subplots(1,2,figsize = (12,6))
    graph = dataset[graph_idx][0]
    curr_event = dataset.dfs[graph_idx]
    colors = curr_event['ModifiedTrueID'].apply(lambda x: 'red' if x == -1 else 'blue')
    sizes = curr_event['Charge0'] + curr_event['Charge1']
    # Create figure
    
    # Plot nodes (hits)
    axs[0].scatter(curr_event['strip_x'], curr_event['strip_y'], 
               c=colors, s=sizes * 1, alpha=0.4, label='Detector hits')
    axs[1].scatter(curr_event['strip_x'], curr_event['strip_y'], 
               c=colors, s=sizes * 1, alpha=0.4, label='Detector hits')
    sources,destinations = graph.edges()
    
    # Plot edges
    for src, dst in zip(sources, destinations):
        x1, y1 = curr_event.iloc[int(src)][['strip_x', 'strip_y']]
        x2, y2 = curr_event.iloc[int(dst)][['strip_x', 'strip_y']]
        axs[0].plot([x1, x2], [y1, y2], 'gray', alpha=0.1, linewidth=0.5)
        axs[1].plot([x1, x2], [y1, y2], 'gray', alpha=0.1, linewidth=0.5)
    # Add reference angle and highlight region
    reference_angle = curr_event['reference_angle'].iloc[0]  # Assuming one reference angle per event
    radius = 330  # Radius of the detector
    
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
    axs[0].plot([0, x_min], [0, y_min], color='orange', linewidth=1.5, label='Lower bound')
    
    # Calculate the coordinates for the line
    x_max = radius * np.cos(np.radians(theta_max))
    y_max = radius * np.sin(np.radians(theta_max))
    axs[0].plot([0, x_max], [0, y_max], color='orange', linewidth=1.5, label='Upper bound')
    
    # Add labels and title
    axs[0].set_xlabel('X Position')
    axs[0].set_ylabel('Y Position')
    n_edges = len(sources) / 2
    fig.suptitle(f'Detector Graph Visualization\n{len(curr_event)} nodes, {n_edges//2} edges')
    
    # Add text with statistics
    stats_text = f'Total nodes: {len(curr_event)}\n'
    stats_text += f'Total edges: {n_edges//2}\n'  # Divide by 2 because edges are bidirectional
    stats_text += f'Average degree: {n_edges/len(curr_event):.1f}'
    axs[1].text(0.02, 0.98, stats_text,
             transform=plot.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    axs[0].grid(True, alpha=0.3)
    axs[0].axis('equal')
    fig.tight_layout()
    axs[0].set_xlim(-radius,radius)
    axs[0].set_ylim(-radius,radius)
    
    
class GIN(nn.Module):
    """
    Graph Neural Network for EIC KLM neutron/K_L energy reconstruction, as well as mu/pi PID.
    
    Based on / inspired by https://github.com/mfmceneaney/Lambda-GNNs and https://arxiv.org/pdf/2302.05481.
    Main function: learn to predict the energy of a particle given a graph of the event, where
    the graph contains nodes with information provided by the SiPMs.
    
    For examples, look at the GNN_Energy.ipynb for neutron energy prediction, GNN_PID.ipynb for PID classification,
    or at train_GNN.py to see the implementation used for MOBO runs.
    
    ...
    
    Attributes
    ----------
    conv_list : nn.ModuleList
        list of GINConv layers applied first in forward propagation.
    linear_list : nn.ModuleList
        List of linear layers (dense layers) for applying after graph convolution
        and pooling. This includes the final layer which can be used for regression
        (predicting energy) or classification (PID).
    pool : dgl.PoolingLayer
        Pooling layer for converting graph structure to a fixed structure so that
        we can apply linear layers and get a fixed output dimension despite having
        a variable input dimension (determined by the number of nodes in a graph).
    
    Methods
    -------
    forward(g, in_feats, event_feats) -> total_feats
        Required function for pytorch Neural Network (nn.Module). Takes in the graph input and 
        computes the output (prediction),  while tracking gradients. Implemented as the 
        __call__ function for the nn.Module, so it does not need to be called explicitly, can instead
        be called as model(g,in_feats,event_feats)
    """
    
    def __init__(self, in_feats, h_feats,num_event_feats,n_conv_layers = 2, n_linear_layers = 7,linear_capacity = 5, num_classes=1,pooling_type = "avg"):
        super(GIN, self).__init__()
        # Define the MLP for the GINConv layers
        conv_list = []
        for i in range(n_conv_layers):
            first_in = (in_feats if i == 0 else h_feats)
            mlp = nn.Sequential(
                nn.Linear(first_in, h_feats),
                nn.ReLU(),
                nn.Linear(h_feats, h_feats)
            )
            conv_list.append(GINConv(mlp))
        self.conv_list = nn.ModuleList(conv_list)
        
        # Define the GINConv layers
        
        linear_list = []
        for i in range(n_linear_layers):
            if(i == 0):
                in_feats = h_feats + num_event_feats
                out_feats = pow(2,linear_capacity + (n_linear_layers // 2))
            elif(i == n_linear_layers - 1):
                in_feats = out_feats
                out_feats = num_classes
            else:
                in_feats = out_feats
                out_feats = out_feats // (2 if (i %2) else 1)
            linear_list.append(nn.Linear(in_feats,out_feats))
        self.linear_list = nn.ModuleList(linear_list)
        
        # Graph pooling layer
        if(pooling_type == "avg"):
            self.pool = AvgPooling()
        elif(pooling_type == "sum"):
            self.pool = SumPooling()
        elif(pooling_type == "max"):
            self.pool = MaxPooling()
        else:
            print(f"Selected pooling type \"{pooling_type}\" not found. Resorting to default: AvgPooling")
            self.pool = AvgPooling()

    def forward(self, g, in_feat,event_feats):
        """
        Forward function as required by nn.Module. Takes in graph inputs, applies convolution using
        self.conv_list, pools over the nodes, then concatenates the event_feats and applies
        linear layers to get prediction output.
        
        Parameters
        ----------
        g : dgl.heterograph.DGLGraph
            graph from dataset with node features, event features, and labels. node features are convolved together,
            event_features are concatenated after pooling, and labels are used for calculating loss
        in_feat : torch.tensor
            Input features that are node_features from graph. Used for graph convolution
        event_feats : torch.tensor
            Features that correspond to an event in general
        
        Returns
        -------
        total_feats : torch.tensor
            Output, either energy prediction or class probabilities for PID.
        """
        # Apply the first GINConv layer
        
        h = in_feat
        hidden_reps = []
        for i in range(len(self.conv_list)):
            h = self.conv_list[i](g,h)
            h = F.relu(h)
            hidden_reps.append(self.pool(g,h))
        
        # Pool the graph-level representation - gives one array of length n_feats
        hg = self.pool(g, h)
        for i in range(len(hidden_reps)):
            hg += hidden_reps[i]
        total_feats = torch.cat((hg,event_feats),axis = 1).float()
        for i in range(len(self.linear_list) - 1):
            total_feats = self.linear_list[i](total_feats)
            total_feats = F.relu(total_feats)
            
        #No activation on last linear layer:
        total_feats = self.linear_list[-1](total_feats)
        return total_feats
    
def train_GNN(model,optimizer,criterion, train_dataloader, val_dataloader, n_epochs,early_stopping_limit,frame_plot_path = "",model_path = "",log_status = True):
    """
    Training function for GIN model
    
    Trains a GNN for energy prediction or particle classification. Uses train dataloader for training
    and uses validation data to keep track of overfitting.
    
    ...
    
    Parameters
    ----------
    model : GIN
        Untrained graph neural network model to train
    optimizer : torch.optim...
        Either Adam or SGD optimizer to use gradients and update weights in GIN model
    criterion : nn...
        Either nn.BSELoss() or nn.MSELoss - loss function
    train_dataloader : dgl.dataloading.dataloader.GraphDataLoader
        Training data
    val_dataloader : dgl.dataloading.dataloader.GraphDataLoader
        Validation data for keeping track of overfitting. Not used for updating weights
    n_epochs : int
        Number of times to train the model on the full dataset
    early_stopping_limit : int
        (aka patience) Each time the model performs worse on the validation set than the previous lowest 
        loss, an uptick is registered. Once this parameter many upticks are registered, training
        is haulted.
    frame_plot_path : str
        Path to directory where frames for gif should be stored.
    model_path : str
        Path to directory where model should be stored.
    log_status : bool
        If True, metrics are printed. If false, nothing is printed. Set to True if you want to 
        stay updated on training. Set to False for training via slurm to keep output clean.
    
    Returns
    -------
     model, train_losses, val_mse, optimizer,early_stopping_dict["best_epoch"]
    model : GIN
        Trained GNN model
    train_losses : list
        List of training losses for plotting loss curve.
    val_mse : list
        Mean squared error of the validation predictions for each epoch.
    optimizer : torch.optim...
        optimizer attached to trained weights (not really needed).
    early_stopping_dict["best_epoch"] : str
        Path of the model that had the lowest epoch loss during training.    
    """
    
    create_directory(model_path)
    val_mse = []
    val_mse_all = []
    train_losses = []
    train_losses_all = []
    early_stopping_dict = {
            "lowest_loss" : -1,
            "best_model_path" : "",
            "num_upticks" : 0,
            "best_epoch": 0
    }

    for epoch in range(n_epochs):
        model.train()
        num_train_batches = 0
        epoch_train_losses = 0.0
        train_preds = torch.tensor([])
        train_truths = torch.tensor([])
        
        for batched_graph, labels_w_event_feats in train_dataloader:
            labels = labels_w_event_feats[:,0]
            event_feats = labels_w_event_feats[:,1:]
            pred = model(batched_graph, batched_graph.ndata["feat"].float(),event_feats)
            loss = criterion(pred, labels.unsqueeze(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_train_losses+=loss.detach()
            train_losses_all.append(loss.detach())
            num_train_batches += 1
            train_preds = torch.cat([train_preds,pred.detach()])
            train_truths = torch.cat([train_truths,labels])
        # Average RMSE for the epoch
        this_epoch_loss = epoch_train_losses / num_train_batches
        train_losses.append(this_epoch_loss)
        # Testing phase
        model.eval()
        epoch_val_mse = 0.0
        num_val_batches = 0
        val_preds = torch.tensor([])
        val_truths = torch.tensor([])
        with torch.no_grad():  # Disable gradients for evaluation
            for batched_graph, labels_w_event_feats in val_dataloader:
                labels = labels_w_event_feats[:,0]
                event_feats = labels_w_event_feats[:,1:]
                pred = model(batched_graph, batched_graph.ndata["feat"].float(),event_feats)
                # Calculate RMSE for this batch
                batch_mse = criterion(pred, labels.unsqueeze(-1))
                epoch_val_mse += batch_mse
                num_val_batches += 1
                val_mse_all.append(batch_mse)
                val_preds = torch.cat([val_preds,pred])
                val_truths = torch.cat([val_truths,labels])

        # Average RMSE for the test set
        epoch_val_mse /= num_val_batches
        val_mse.append(epoch_val_mse)
        if(epoch %1 == 0):
            if(log_status):
                print(f"Epoch {epoch + 1}/{n_epochs} - Train loss:\t {this_epoch_loss:.4f}")
                print(f"Epoch {epoch + 1}/{n_epochs} - Validation MSE:\t {epoch_val_mse:.4f}\n")
            if(frame_plot_path != ""):
                frame_fig, frame_axs = plot.subplots(1,1)
                frame_axs.plot([0,5],[0,5])
                frame_fig.suptitle("Test dataset results")
                frame_axs.scatter(val_truths,val_preds,alpha = 0.05,color = "red",label = "val")
    #             plot.scatter(train_truths,train_preds,alpha = 0.01,color = "blue",label = "train")
                frame_axs.set_xlabel("truths")
                frame_axs.set_ylabel("preds")
                frame_axs.text(3.1,1.3, f"Epoch #{epoch + 1}\nTrain, val loss: ({this_epoch_loss:.4f},{epoch_val_mse:.4f})")
                frame_fig.tight_layout()
                frame_fig.savefig(f"{frame_plot_path}epoch{epoch}.jpeg")
        
        if(epoch_val_mse.item() < early_stopping_dict["lowest_loss"] or early_stopping_dict["lowest_loss"] == -1):
            early_stopping_dict["lowest_loss"] = epoch_val_mse
            early_stopping_dict["best_model_path"] = f"{model_path}epoch_{epoch}.pth"
            early_stopping_dict["num_upticks"] = 0
            early_stopping_dict["best_epoch"] = epoch
            
            torch.save(model.state_dict(),early_stopping_dict["best_model_path"])
        elif(epoch_val_mse.item() > early_stopping_dict["lowest_loss"]):
            early_stopping_dict["num_upticks"] += 1
            if(log_status):
                print("Test loss increased, adding uptick")
        if(early_stopping_dict["num_upticks"] >= early_stopping_limit):
            # Stop training, load best model
            model.load_state_dict(torch.load(early_stopping_dict["best_model_path"]))
            torch.save(model.state_dict(),f"{model_path}best_model.pth")
            if(log_status):
                print("Stopping early, loading current model...")
            break
    return model, train_losses, val_mse, optimizer,early_stopping_dict["best_epoch"]

def test_GNN(model, test_dataloader):
    """
    Evaluate GNN performance on unseen test dataset
    
    ...
    
    Parameters
    ----------
    model : GIN
        Trained model
    test_dataloader : dgl.dataloading.dataloader.GraphDataLoader
        Testing data inside a dataloader that allows batching
    
    Returns
    -------
    truths, preds, rmse
    truths : list
        List of true PID/Energy.
    preds : list
        List of predictions from GNN
    rmse : float
        Root mean squared error for predicted energies vs true energies.
    """
    
    truths = []
    preds = []
    summed_sqe = 0.0
    num_predictions = 0
    with torch.no_grad():
        for batched_graph, labels in test_dataloader:
            graphs = dgl.unbatch(batched_graph)
            for i in range(len(graphs)):
                graph = graphs[i]
                labels_w_event_feats = labels[i]
                label = labels_w_event_feats[0].item()
                event_feats = labels_w_event_feats[1:].unsqueeze(0)
                pred = model(graph, graph.ndata["feat"].float(),event_feats).detach().numpy()[0][0]
                summed_sqe += pow(pred - label,2)
                num_predictions += 1

                preds.append(pred)
                truths.append(label)
    rmse = np.sqrt(summed_sqe / num_predictions)
    print(f"RMSE: {rmse}")
    return truths, preds, rmse
def test_GNN_binned(model, test_dataloader):
    """
    See test_GNN
    
    This testing function calculates the RMSE for two different
    energy ranges.
    """
    truths = []
    preds = []
    summed_sqe = 0.0
    num_predictions = 0
    summed_sqe_binned = np.zeros(2)
    num_predictions_binned = np.zeros(2)
    with torch.no_grad():
        for batched_graph, labels in test_dataloader:
            graphs = dgl.unbatch(batched_graph)
            for i in range(len(graphs)):
                graph = graphs[i]
                labels_w_event_feats = labels[i]
                label = labels_w_event_feats[0].item()
                event_feats = labels_w_event_feats[1:].unsqueeze(0)
                pred = model(graph, graph.ndata["feat"].float(),event_feats).detach().numpy()[0][0]
                summed_sqe += pow(pred - label,2)
                num_predictions += 1
                if(label < 2.75):
                    summed_sqe_binned[0] += pow(pred - label,2)
                    num_predictions_binned[0] += 1
                else:
                    summed_sqe_binned[1] += pow(pred - label,2)
                    num_predictions_binned[1] += 1

                preds.append(pred)
                truths.append(label)
    rmse = np.sqrt(summed_sqe / num_predictions)
    binned_rmse = np.sqrt(summed_sqe_binned / num_predictions_binned)
    print(f"RMSE: {rmse}")
    print(f"RMSE for E < 2.75GeV: {binned_rmse[0]}; E > 2.75GeV: {binned_rmse[1]}")
    return truths, preds, rmse, binned_rmse

def get_text_positions(bin_centers,rel_rmse,test_truths,test_preds):
    """
    Calculates approximately "good" positions for text in 
    neutron RMSE plot.
    
    See GNN_Energy.ipynb
    
    Parameters
    ----------
    bin_centers : np.array
        Centers of the energy bins.
    rel_rmse : 
    """
    
    rel_rmse_x = np.mean(bin_centers) 
    rel_rmse_y = np.mean(rel_rmse) +  np.max(rel_rmse) / 10
    
    scatter_x = np.min(test_truths)
    scatter_y = np.max(test_preds) - 0.5
    return rel_rmse_x,rel_rmse_y,scatter_x,scatter_y

def get_min_max_of_graph_dataset(dataset):
    curr_max = -1
    curr_min = -1
    for graph, labels in dataset:
        true_E = labels[0]
        if(curr_max == -1):
            curr_max = true_E
            curr_min = true_E
        elif(curr_max < true_E):
            curr_max = true_E
        elif(curr_min > true_E):
            curr_min = true_E
    return curr_min, curr_max

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
                labels_w_event_feats = labels[i]
                label = labels_w_event_feats[0].item()
                event_feats = labels_w_event_feats[1:].unsqueeze(0)
                pred = model(graph, graph.ndata["feat"].float(),event_feats).detach().numpy()

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