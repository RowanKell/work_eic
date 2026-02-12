import matplotlib.pyplot as plot
import pandas as pd
import os
os.environ["DGLBACKEND"] = "pytorch"
import dgl
import torch
from dgl.data import DGLDataset
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
from dgl.nn import GINConv, AvgPooling
from datetime import datetime as datetime
current_date = datetime.now().strftime("%B_%d")
from torch.utils.data.sampler import SubsetRandomSampler
from GNN_util import process_df_vectorized, create_directory, HitDataset, GIN, delete_files_in_dir
import argparse
from pathlib import Path
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser(description='Training GNN classifier for mu/pi PID')

parser.add_argument('--inputDataPrefMu', type=str, default="NA",
                        help='Whole path of muon CSV files excluding "i.csv" at the end')
parser.add_argument('--inputDataPrefPi', type=str, default="NA",
                        help='Whole path of pion CSV files excluding "i.csv" at the end')
parser.add_argument('--numDfs', type=int, default=1,
                        help='Number of csv files per particle to read into DataFrames')
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
parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
parser.add_argument('--MLPHiddenDim', type=int, default=32,
                        help='MLP hidden dimension for GIN conv layers')
parser.add_argument('--trainingBatchSize', type=int, default=20,
                        help='Number of graphs in each training batch')
parser.add_argument('--modelPath', type=str, default="/hpc/group/vossenlab/rck32/eic/work_eic/macros/Timing_estimation/models/unsorted/",
                        help='Path to save trained model')
parser.add_argument('--nEpochs', type=int, default=300,
                        help='Maximum number of training epochs')
parser.add_argument('--earlyStoppingLimit', type=int, default=5,
                        help='Number of epochs without improvement before stopping')
parser.add_argument('--lossPlotPath', type=str, default="",
                        help='Full path to save loss plot image')
parser.add_argument('--testPlotPath', type=str, default="",
                        help='Full path to save test ROC plot image')
parser.add_argument('--resultsFilePath', type=str, default="",
                        help='File to APPEND the AUC values to (mode "a")')
parser.add_argument('--runName', type=str, default="",
                        help='Name to use for saving files')
parser.add_argument('--deleteDfs', action=argparse.BooleanOptionalAction,
                        help='If true, delete the CSV files after successful training')

args = parser.parse_args()

inputDataPrefMu = args.inputDataPrefMu
inputDataPrefPi = args.inputDataPrefPi
num_dfs = args.numDfs
coneAngle = args.coneAngle
kNN_k = args.kNNk
train_frac = args.trainFrac
val_frac = args.valFrac
run_num = args.runNum
lr = args.lr
MLP_hidden_dim = args.MLPHiddenDim
training_batch_size = args.trainingBatchSize
n_epochs = args.nEpochs
early_stopping_limit = args.earlyStoppingLimit
loss_plot_path = args.lossPlotPath
test_plot_path = args.testPlotPath
model_path = args.modelPath
results_file_path = args.resultsFilePath
run_name = args.runName
deleteDfs = args.deleteDfs

# Check directories
path_list = [loss_plot_path, test_plot_path, model_path]
for path in path_list:
    if(path != ''):
        create_directory(path)


class GIN_Classifier(nn.Module):
    """Wraps the GIN from GNN_util with a sigmoid activation for binary classification."""
    def __init__(self, in_feats, h_feats, num_event_feats, n_conv_layers=2, n_linear_layers=7, linear_capacity=5, num_classes=1, pooling_type="avg"):
        super(GIN_Classifier, self).__init__()
        self.gin = GIN(in_feats, h_feats, num_event_feats, n_conv_layers, n_linear_layers, linear_capacity, num_classes, pooling_type)
        self.sig = nn.Sigmoid()

    def forward(self, g, in_feat, event_feats):
        out = self.gin(g, in_feat, event_feats)
        return self.sig(out)


def train_classifier(model, optimizer, criterion, train_dataloader, val_dataloader, n_epochs, early_stopping_limit, model_path="", log_status=True):
    """Train the GNN classifier with BCELoss and early stopping. Mirrors train_GNN structure."""
    create_directory(model_path)
    val_loss = []
    train_losses = []
    early_stopping_dict = {
        "lowest_loss": -1,
        "best_model_path": "",
        "num_upticks": 0,
        "best_epoch": 0
    }

    for epoch in range(n_epochs):
        model.train()
        num_train_batches = 0
        epoch_train_losses = 0.0
        for batched_graph, labels_w_event_feats in train_dataloader:
            labels = labels_w_event_feats[:, 0]
            event_feats = labels_w_event_feats[:, 1:]
            pred = model(batched_graph, batched_graph.ndata["feat"].float(), event_feats)
            loss = criterion(pred, labels.unsqueeze(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_losses += loss.detach()
            num_train_batches += 1

        this_epoch_loss = epoch_train_losses / num_train_batches
        train_losses.append(this_epoch_loss)

        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        num_val_batches = 0
        num_correct = 0
        num_total = 0
        with torch.no_grad():
            for batched_graph, labels_w_event_feats in val_dataloader:
                labels = labels_w_event_feats[:, 0]
                event_feats = labels_w_event_feats[:, 1:]
                pred = model(batched_graph, batched_graph.ndata["feat"].float(), event_feats)
                batch_loss = criterion(pred, labels.unsqueeze(-1))
                epoch_val_loss += batch_loss
                num_val_batches += 1

                argmax_prediction = pred.squeeze() > 0.5
                num_correct += torch.sum(argmax_prediction == labels)
                num_total += len(labels)

        epoch_val_loss /= num_val_batches
        epoch_acc = num_correct / num_total
        val_loss.append(epoch_val_loss)

        if(log_status and epoch % 1 == 0):
            print(f"Epoch {epoch + 1}/{n_epochs} - Train loss:\t {this_epoch_loss:.4f}")
            print(f"Epoch {epoch + 1}/{n_epochs} - Validation Loss:\t {epoch_val_loss:.4f}")
            print(f"Epoch {epoch + 1}/{n_epochs} - Validation Accuracy:\t {epoch_acc:.4f}\n")

        if(epoch_val_loss.item() < early_stopping_dict["lowest_loss"] or early_stopping_dict["lowest_loss"] == -1):
            early_stopping_dict["lowest_loss"] = epoch_val_loss
            early_stopping_dict["best_model_path"] = f"{model_path}epoch_{epoch}.pth"
            early_stopping_dict["num_upticks"] = 0
            early_stopping_dict["best_epoch"] = epoch
            torch.save(model.state_dict(), early_stopping_dict["best_model_path"])
        elif(epoch_val_loss.item() > early_stopping_dict["lowest_loss"]):
            early_stopping_dict["num_upticks"] += 1
            if(log_status):
                print("Validation loss increased, adding uptick")
        if(early_stopping_dict["num_upticks"] >= early_stopping_limit):
            model.load_state_dict(torch.load(early_stopping_dict["best_model_path"]))
            torch.save(model.state_dict(), f"{model_path}best_model.pth")
            if(log_status):
                print("Stopping early, loading best model...")
            break
    return model, train_losses, val_loss, optimizer, early_stopping_dict["best_epoch"]


def test_classifier_binned(model, test_dataloader, dataset, test_indices):
    """
    Evaluate classifier on test set and compute ROC AUC in two energy bins.

    Mirrors test_GNN_binned but computes AUC instead of RMSE.
    Energy threshold: 2.75 GeV (matching neutron RMSE binning).
    Energy = sqrt(mass^2 + P^2), using momentum and PID from dataset dataframes.
    """
    mass_dict = {
        130: 0.497611,
        2112: 0.939565,
        211: 0.139570,
        -211: 0.139570,
        13: 0.10566
    }

    truths_low = []
    probs_low = []
    truths_high = []
    probs_high = []
    truths_all = []
    probs_all = []

    num_correct = 0
    num_total = 0

    test_energies = []
    for idx in test_indices:
        df = dataset.dfs[idx]
        truePID = df["truePID"].to_numpy()[0]
        momentum = df["P"].to_numpy()[0]
        mass = mass_dict[truePID]
        energy = np.sqrt(mass**2 + momentum**2)
        test_energies.append(energy)

    event_counter = 0
    with torch.no_grad():
        for batched_graph, labels in test_dataloader:
            graphs = dgl.unbatch(batched_graph)
            for i in range(len(graphs)):
                graph = graphs[i]
                labels_w_event_feats = labels[i]
                label = labels_w_event_feats[0].item()
                event_feats = labels_w_event_feats[1:].unsqueeze(0)
                pred = model(graph, graph.ndata["feat"].float(), event_feats).detach().numpy()[0][0]

                pred_class = int(pred >= 0.5)
                num_correct += pred_class == int(label)
                num_total += 1

                energy = test_energies[event_counter]
                event_counter += 1

                truths_all.append(int(label))
                probs_all.append(pred)

                if energy < 2.75:
                    truths_low.append(int(label))
                    probs_low.append(pred)
                else:
                    truths_high.append(int(label))
                    probs_high.append(pred)

    accuracy = num_correct / num_total
    print(f"Test accuracy: {accuracy:.4f}")

    overall_auc = roc_auc_score(truths_all, probs_all)
    print(f"Overall AUC: {overall_auc:.4f}")

    low_auc = roc_auc_score(truths_low, probs_low) if len(set(truths_low)) > 1 else -1.0
    high_auc = roc_auc_score(truths_high, probs_high) if len(set(truths_high)) > 1 else -1.0

    print(f"Low energy (E < 2.75 GeV) AUC: {low_auc:.4f} ({len(truths_low)} events)")
    print(f"High energy (E >= 2.75 GeV) AUC: {high_auc:.4f} ({len(truths_high)} events)")

    return truths_all, probs_all, accuracy, low_auc, high_auc


# ============================================================
#                     DATA LOADING
# ============================================================

# Load muon CSVs
dfs = []
for i in range(num_dfs):
    try:
        new_df = pd.read_csv(f"{inputDataPrefMu}{i}.csv")
    except FileNotFoundError as e:
        print(f"skipping muon file #{i}...")
        continue
    except pd.errors.EmptyDataError as e:
        print(f"found error: {e}")
        print(f"df index: {i}\ninputDataPrefMu: {inputDataPrefMu}")
        raise Exception(f"not sure why this error occurs: {e}")
    new_df["file_idx"] = i
    dfs.append(new_df)

# Load pion CSVs with offset file_idx to avoid collision
for j in range(num_dfs):
    try:
        new_df = pd.read_csv(f"{inputDataPrefPi}{j}.csv")
    except FileNotFoundError as e:
        print(f"skipping pion file #{j}...")
        continue
    except pd.errors.EmptyDataError as e:
        print(f"found error: {e}")
        print(f"df index: {j}\ninputDataPrefPi: {inputDataPrefPi}")
        raise Exception(f"not sure why this error occurs: {e}")
    new_df["file_idx"] = num_dfs + j
    dfs.append(new_df)

if(len(dfs) > 1):
    data = pd.concat(dfs)
else:
    data = dfs[0]

print(f"Loaded {len(dfs)} CSV files total ({num_dfs} per particle)")

modified_df = process_df_vectorized(data, cone_angle_deg=coneAngle)

filter_events_flag = True
connection_mode = "kNN"
dataset = HitDataset(modified_df, filter_events_flag, connection_mode=connection_mode, k=kNN_k, function='PID')
print("Finished Creating HitDataset")
print(f"Dataset size: {len(dataset)} events")

# ============================================================
#            STRATIFIED TRAIN / VAL / TEST SPLIT
# ============================================================

# Get labels for stratified split
labels = torch.tensor([dataset[i][1][0] for i in range(len(dataset))])

class_0_indices = torch.where(labels == 0)[0].tolist()
class_1_indices = torch.where(labels == 1)[0].tolist()

np.random.shuffle(class_0_indices)
np.random.shuffle(class_1_indices)

num_train_0 = int(len(class_0_indices) * train_frac)
num_val_0 = int(len(class_0_indices) * val_frac)

num_train_1 = int(len(class_1_indices) * train_frac)
num_val_1 = int(len(class_1_indices) * val_frac)

train_indices = class_0_indices[:num_train_0] + class_1_indices[:num_train_1]
val_indices = (
    class_0_indices[num_train_0: num_train_0 + num_val_0]
    + class_1_indices[num_train_1: num_train_1 + num_val_1]
)
test_indices = class_0_indices[num_train_0 + num_val_0:] + class_1_indices[num_train_1 + num_val_1:]

np.random.shuffle(train_indices)
np.random.shuffle(val_indices)
np.random.shuffle(test_indices)

print(f"Split: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test")
print(f"Class balance - class 0 (pion): {len(class_0_indices)}, class 1 (muon): {len(class_1_indices)}")

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_dataloader = GraphDataLoader(
    dataset, sampler=train_sampler, batch_size=training_batch_size, drop_last=False
)
val_dataloader = GraphDataLoader(
    dataset, sampler=val_sampler, batch_size=training_batch_size, drop_last=False
)
test_dataloader = GraphDataLoader(
    dataset, sampler=test_sampler, batch_size=training_batch_size, drop_last=False
)

# ============================================================
#                  TRAINING (multiple runs)
# ============================================================

best = {
    'model': None,
    'accuracy': None,
    'train_losses': None,
    'val_losses': None,
    'low_auc': None,
    'high_auc': None
}
binned_aucs_sum = np.zeros(2)
num_trainings = 3
for i in range(num_trainings):
    model = GIN_Classifier(dataset.dim_nfeats, MLP_hidden_dim, dataset.dim_event_feats)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    trained_model, train_losses, val_losses, optimizer, best_epoch = train_classifier(
        model, optimizer, criterion, train_dataloader, val_dataloader, n_epochs, early_stopping_limit, model_path
    )
    test_truths, test_probs, accuracy, low_auc, high_auc = test_classifier_binned(
        trained_model, test_dataloader, dataset, test_indices
    )
    print(f"training #{i}: low_auc={low_auc:.4f}, high_auc={high_auc:.4f}")
    binned_aucs_sum[0] += low_auc
    binned_aucs_sum[1] += high_auc
    if(best['model'] is None or accuracy > best['accuracy']):
        best['model'] = trained_model
        best['train_losses'] = train_losses
        best['val_losses'] = val_losses
        best['accuracy'] = accuracy
        best['low_auc'] = low_auc
        best['high_auc'] = high_auc

binned_aucs_avg = binned_aucs_sum / num_trainings
trained_model = best['model']
train_losses = best['train_losses']
val_losses = best['val_losses']

print(f"\nAverage binned AUCs over {num_trainings} trainings:")
print(f"  Low energy AUC:  {binned_aucs_avg[0]:.4f}")
print(f"  High energy AUC: {binned_aucs_avg[1]:.4f}")

# ============================================================
#                       PLOTS
# ============================================================

if(loss_plot_path != ""):
    loss_fig, loss_axs = plot.subplots(1, 1)
    loss_axs.plot(train_losses, label="train")
    loss_fig.suptitle(f"Train and Validation loss throughout training, run {run_num}")
    loss_axs.plot(val_losses, label="val")
    loss_axs.legend()
    loss_fig.tight_layout()
    loss_fig.savefig(f"{loss_plot_path}{run_name}.jpeg")

if(test_plot_path != ""):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(test_truths, test_probs)
    roc_auc = auc(fpr, tpr)
    roc_fig, roc_axs = plot.subplots(1, 1)
    roc_axs.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    roc_axs.set_xlim([0.0, 1.0])
    roc_axs.set_ylim([0.0, 1.05])
    roc_axs.set_xlabel('False Positive Rate', fontsize=20)
    roc_axs.set_ylabel('True Positive Rate', fontsize=20)
    roc_axs.legend(loc="lower right", fontsize=20)
    roc_axs.grid(True)
    roc_fig.tight_layout()
    roc_fig.savefig(f"{test_plot_path}{run_name}.jpeg")

# ============================================================
#                  WRITE RESULTS (append)
# ============================================================

if(results_file_path != ""):
    if(os.path.isdir(results_file_path)):
        results_write_path = f"{results_file_path}{run_name}.txt"
    else:
        results_write_path = f"{results_file_path}"
    with open(results_write_path, "a") as f:
        writeString = f"\n{binned_aucs_avg[0]}\n{binned_aucs_avg[1]}"
        f.write(writeString)
        print(f"appending AUC: {writeString}")

# ============================================================
#                  CLEANUP
# ============================================================

if(deleteDfs):
    final_model_file = Path(f"{model_path}best_model.pth")
    if(final_model_file.is_file()):
        print(f"successfully saved best model")
        for i in range(num_dfs):
            mu_file = Path(f"{inputDataPrefMu}{i}.csv")
            if(mu_file.is_file()):
                mu_file.unlink()
                print(f"deleted muon df file {inputDataPrefMu}{i}.csv")
            pi_file = Path(f"{inputDataPrefPi}{i}.csv")
            if(pi_file.is_file()):
                pi_file.unlink()
                print(f"deleted pion df file {inputDataPrefPi}{i}.csv")
print("finished train_GNN_classifier")
