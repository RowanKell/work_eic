


from datetime import date
date = date.today().strftime("%b_%d")

# Import packages
import torch
import numpy as np
import normflows as nf

import uproot as up

from matplotlib import pyplot as plot
import math
from tqdm import tqdm
from NF_util import PVect, theta_func, r_func, checkdir
from IPython.display import clear_output
import time
import os
def checkdir(path):
    if not os.path.exists(path): 
        os.makedirs(path)
import argparse
parser = argparse.ArgumentParser(description = 'NF training')
parser.add_argument('--K', type=int, default=1,
                        help='# of flows')
parser.add_argument('--hu', type=int, default=100,
                        help='# hidden units')
parser.add_argument('--hl', type=int, default=6,
                        help='# hidden layers')
parser.add_argument('--bs', type=int, default=2000,
                        help='# datapoints per sample')
parser.add_argument('--run_num', type=int, default=0,
                        help='run number (of the day)')
parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate for optimizer')
parser.add_argument('--num_epochs', type=int, default=6,
                        help='# epochs to train')
parser.add_argument('--useArgs', action=argparse.BooleanOptionalAction,
                        help='If True, uses argparse arguments')
    
args = parser.parse_args()
useArgs = args.useArgs
# Define flows
if(useArgs):
    run_num = args.run_num
    K = args.K
    hidden_units = args.hu
    hidden_layers = args.hl
    batch_size = args.bs
    lr = args.lr
    num_epochs = args.num_epochs
else:
    run_num = 1
    K = 2
    hidden_units = 64
    hidden_layers = 8
    batch_size = 2000
    lr = 1e-5
    num_epochs = 6

import logging
import os
log_file = f'/hpc/group/vossenlab/rck32/eic/work_eic/macros/NF_timing_modeling/slurm/logs/train_run_{run_num}.log'
if os.path.exists(log_file):
    os.remove(log_file)


# Configure logging
logging.basicConfig(filename=log_file, level=logging.DEBUG)
# Log messages
logging.debug('Start of train.py')
    
    
# Get device to be used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


Timing_path = "/hpc/group/vossenlab/rck32/eic/work_eic/macros/NF_timing_modeling/"
logging.debug("before loading data")

combined_inputs_already = True
if(not combined_inputs_already):

    raw_inputs = torch.load(Timing_path + "data/January_30/Run_1/Vary_p_events_file_0_600_z_pos_fixed_z_hit_pos.pt")
    for i in range(600):
        clear_output(wait=True)
        logging.debug(f"loaded file #{i+1} of 600")
        raw_inputs = torch.cat((raw_inputs, torch.load(Timing_path + f"data/January_30/Run_1/Vary_p_events_file_{i+1}_600_z_pos_fixed_z_hit_pos.pt")),0)
    inputs = raw_inputs[np.logical_and(raw_inputs[:,4] < 100,raw_inputs[:,3] < 0.06)]
    indexes = torch.randperm(inputs.shape[0])
    dataset = inputs[indexes]
    train_frac = 0.16
    test_frac = 0.02
    val_frac = 0.02
    train_lim = int(np.floor(dataset.shape[0] * train_frac))
    test_lim = train_lim + int(np.floor(dataset.shape[0] * test_frac))
    val_lim = test_lim + int(np.floor(dataset.shape[0] * val_frac))
    train_data = dataset[:train_lim]
    test_data = dataset[train_lim:test_lim]
    val_data = dataset[test_lim:val_lim]

    checkdir(Timing_path + "data/combined/January_30/")

#     torch.save(train_data,Timing_path + "data/combined/January_30/onesix_600_z_pos_train.pt")
#     torch.save(test_data,Timing_path + "data/combined/January_30/onesix_600_z_pos_test.pt")
#     torch.save(val_data,Timing_path + "data/combined/January_30/onesix_600_z_pos_val.pt")
else:
    train_data = torch.load(Timing_path + "data/combined/January_30/onesix_600_z_pos_train.pt")
    test_data = torch.load(Timing_path + "data/combined/January_30/onesix_600_z_pos_test.pt")
    val_data = torch.load(Timing_path + "data/combined/January_30/onesix_600_z_pos_val.pt")


logging.debug("Finished Loading Data")
    
'''NOTE: originally (in august) we used 2000 bs, but that is taking pretty long, so Im gonna try 20k to start'''


latent_size = 1
context_size = 3
num_context = 3
#strings for file paths
lr_str = str(lr)
K_str = str(K)
hidden_units_str = str(hidden_units)
hidden_layers_str = str(hidden_layers)
batch_size_str = str(batch_size)
num_context_str = str(num_context)
run_num_str = str(run_num)
# input_fig, input_axs = plot.subplots(1,3,figsize=(18,8))
# input_fig.suptitle("train.py training inputs")
# input_axs[0].hist(train_data[:100000,0],bins = 100)
# input_axs[0].set_title("hit z")
# input_axs[1].hist(train_data[:100000,1],bins = 100)
# input_axs[1].set_title("theta (deg)")
# input_axs[2].hist(train_data[:100000,2],bins = 100)
# input_axs[2].set_title("momentum")
# input_fig.savefig("/cwork/rck32/eic/work_eic/macros/Timing_estimation/plots/inputs/August_5/inputs train_" + run_num_str+  ".jpeg")
logging.debug("finished making inputs plot")
flows = []
for i in range(K):
    flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units, 
                                                             num_context_channels=3)]
    flows += [nf.flows.LULinearPermute(latent_size)]

# Set base distribution
from normflows.distributions import BaseDistribution
class DiagGaussian(BaseDistribution):
    """
    Multivariate Gaussian distribution with diagonal covariance matrix
    """

    def __init__(self, shape, trainable=True):
        """Constructor

        Args:
          shape: Tuple with shape of data, if int shape has one dimension
          trainable: Flag whether to use trainable or fixed parameters
        """
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        if isinstance(shape, list):
            shape = tuple(shape)
        self.shape = shape
        self.n_dim = len(shape)
        self.d = np.prod(shape)
        if trainable:
            self.loc = nn.Parameter(torch.zeros(1, *self.shape))
            self.log_scale = nn.Parameter(torch.zeros(1, *self.shape))
        else:
            self.register_buffer("loc", 4.5 * torch.ones(1, *self.shape))
            self.register_buffer("log_scale", torch.zeros(1, *self.shape))
        self.temperature = None  # Temperature parameter for annealed sampling

    def forward(self, num_samples=1, context=None):
        eps = torch.randn(
            (num_samples,) + self.shape, dtype=self.loc.dtype, device=self.loc.device
        )
        if self.temperature is None:
            log_scale = self.log_scale
        else:
            log_scale = self.log_scale + np.log(self.temperature)
        z = self.loc + torch.exp(log_scale) * eps
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            log_scale + 0.5 * torch.pow(eps, 2), list(range(1, self.n_dim + 1))
        )
        return z, log_p

    def log_prob(self, z, context=None):
        if self.temperature is None:
            log_scale = self.log_scale
        else:
            log_scale = self.log_scale + np.log(self.temperature)
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            log_scale + 0.5 * torch.pow((z - self.loc) / torch.exp(log_scale), 2),
            list(range(1, self.n_dim + 1)),
        )
        return log_p
q0 = DiagGaussian(1, trainable=False) #Typical use

# Construct flow model
model = nf.ConditionalNormalizingFlow(q0, flows)

# Move model on GPU if available
model = model.to(device)

import datetime

x = datetime.datetime.now()
today = x.strftime("%B_%d")
# model_path = "models/context_3/" + today + "/"
# checkdir(model_path)

model_path = "../models/" + today + "/"
checkdir(model_path)

loss_path = "../plots/loss/" + today + "/"
checkdir(loss_path)

test_data_path = "../data/test/" + today + "/"
checkdir(test_data_path)

run_info = "run_" + run_num_str+"_"+num_context_str+ "context_"+ K_str + "flows_" + hidden_layers_str + "hl_" + hidden_units_str + "hu_" + batch_size_str + "bs"

# Train model
num_context = 4
max_iter = int(np.floor(train_data.shape[0] / batch_size))
train_loss_hist = np.array([])
val_loss_hist = np.array([])
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

validation_frequency = 2000  # Perform validation every 100 training steps
logging.debug("beginning training loop")

# Early stopping
max_ticks = 10
early_stopping_dict = {"val_loss":-1,"upticks":0}

global_step = 0
for epoch in range(num_epochs):
    if(early_stopping_dict["upticks"] > max_ticks):
        break
    logging.debug(f"\n\nBeginning epoch #{epoch}")
    model.train()  # Set model to training mode
    for it in range(max_iter):
        optimizer.zero_grad()
        # Get training samples
        begin = it * batch_size
        end = (it + 1) * batch_size
        it_data = train_data[begin:end]
        context = torch.empty(it_data.size()[0], 3)
        context[:,0] = it_data[:,0]
        context[:,1] = it_data[:,1]
        context[:,2] = it_data[:,2]
        context = context.to(device)
        samples = (it_data[:,4] - it_data[:,3]).unsqueeze(1).to(device)
        # Compute loss
        loss = model.forward_kld(samples, context)
        logging.debug(f"Iteration {it} loss: {loss}")
        # Do backprop and optimizer step
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer.step()
        # Log loss
        train_loss_hist = np.append(train_loss_hist, loss.to('cpu').data.numpy())

        global_step += 1

        # Validation step every 100 training steps
        if global_step == validation_frequency:
            model.eval()  # Set model to evaluation mode
            val_loss = 0
            val_iter = min(100, int(np.floor(val_data.shape[0] / batch_size)))  # Limit validation to 100 batches
            with torch.no_grad():
                for val_it in range(val_iter):
                    begin = val_it * batch_size
                    end = (val_it + 1) * batch_size
                    it_data = val_data[begin:end]
                    context = torch.empty(it_data.size()[0], 3)
                    context[:,0] = it_data[:,0]
                    context[:,1] = it_data[:,1]
                    context[:,2] = it_data[:,2]
                    context = context.to(device)
                    samples = (it_data[:,4] - it_data[:,3]).unsqueeze(1).to(device)
                    loss = model.forward_kld(samples, context)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / val_iter
            val_loss_hist = np.append(val_loss_hist, avg_val_loss)
            
            logging.debug(f"\n\nStep {it} - Train Loss: {train_loss_hist[-1]:.4f}, Val Loss: {avg_val_loss:.4f}\n\n")
            
            model.train()  # Set model back to training mode
            global_step = 0
            
            if(avg_val_loss < early_stopping_dict["val_loss"] or early_stopping_dict["val_loss"] == -1):
                early_stopping_dict["val_loss"] = avg_val_loss
                model.save(model_path + run_info + f"_epoch{epoch}_step{it}.pth")
            if(avg_val_loss > early_stopping_dict["val_loss"]):
                model.save(model_path + run_info + f"_epoch{epoch}_step{it}_uptick.pth")
                early_stopping_dict["upticks"] += 1
            if(early_stopping_dict["upticks"] > max_ticks):
                print("Hit the max upticks, stopping now...")
                break
    model.save(model_path + run_info + f"_checkpoint_e{epoch}.pth")
    logging.debug(f"Epoch {epoch} completed.")
    
model.save(model_path + run_info + "_finished.pth")
# Plot loss
plot.figure(figsize=(5, 5))
plot.scatter(range(len(train_loss_hist)),train_loss_hist, np.ones(len(train_loss_hist)) * 0.01, label='loss', alpha = 1)
plot.scatter(np.linspace(0,len(train_loss_hist),len(val_loss_hist)),val_loss_hist, np.ones(len(val_loss_hist)) * 0.1, label='loss', alpha = 1)
plot.legend()
plot.savefig( loss_path + run_info + ".jpeg")

# torch.save(test_data, test_data_path + "full_test_data_run_" + run_num_str+ "_"+ K_str + "flows_" + hidden_layers_str + "hl_" + hidden_units_str + "hu_" + batch_size_str + "bs.pt")