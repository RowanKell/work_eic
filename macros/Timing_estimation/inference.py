## Setup

# Import packages
import torch
import numpy as np
import normflows as nf

import uproot as up

from matplotlib import pyplot as plot
import math
from tqdm import tqdm
from util import PVect, theta_func, r_func
from IPython.display import clear_output


# Get device to be used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
## Neural Spline Flow

# Define flows
K = 8

latent_size = 1
hidden_units = 64
hidden_layers = 6
context_size = 4
num_context = 4

flows = []
for i in range(K):
    flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units, 
                                                             num_context_channels=context_size)]
    flows += [nf.flows.LULinearPermute(latent_size)]

# Set base distribution
q0 = nf.distributions.DiagGaussian(1, trainable=False)
    
# Construct flow model
model = nf.ConditionalNormalizingFlow(q0, flows)

# Move model on GPU if available
model = model.to(device)


model.load("models/July_11/nspline_8flows_6hl_64hu_2000bs.pth")

test_data = torch.load("data/test/July_11/full_test_data_8flows_6hl_64hu_2000bs.pt")

min_time = min(test_data[:, 4])
skipped = np.array([])
eval_batch_size = 200000
eval_max_iter = test_data.shape[0] // eval_batch_size
eval_test_data = test_data[:eval_max_iter * eval_batch_size]
eval_test_data = eval_test_data.to("cpu")
model = model.to("cpu")
samples = torch.empty(eval_test_data.shape[0])

for i in tqdm(range(eval_max_iter)):
    begin = eval_batch_size * i
    end = eval_batch_size * (i + 1)
    context = eval_test_data[begin:end, :num_context]
    
    # Initialize a mask for valid samples
    valid_samples = torch.zeros(eval_batch_size, dtype=torch.bool)
    batch_samples = torch.empty(eval_batch_size)
    
    while not valid_samples.all():
        # Generate samples for the invalid positions
        new_samples = model.sample(num_samples=(~valid_samples).sum(), context=context[~valid_samples])[0].cpu().detach().squeeze(1)
        
        # Update the samples and the valid_samples mask
        batch_samples[~valid_samples] = new_samples
        valid_samples = batch_samples >= min_time
    
    samples[begin:end] = batch_samples

# Count and report the number of resampled points
num_resampled = (samples < min_time).sum().item()
# print(f"Number of resampled points: {num_resampled}")

torch.save(samples,"data/samples/July_11/samples_8_flows_6hl_64hu_2000_bs.pt")
torch.save(eval_test_data,"data/test/July_11/eval_8_flows_6hl_64hu_2000_bs.pt")

sample_fig, sample_axs = plot.subplots(1,1,figsize=(6,6))
sample_fig.suptitle("photon hit timing (p varied) 6 hidden layers, 8 flows, 64 units, bs = 2000")
sample_axs.hist(samples,bins = 200, alpha = 0.5,color = 'b', label = 'learned', density = True)
sample_axs.set_title("learned and true distributions")
sample_axs.set_xlabel("time (ns)")
sample_axs.set_ylabel("counts")
sample_axs.hist(eval_test_data[:,num_context],bins = 200, color = 'r', alpha = 0.5, label = 'true', density = True)
sample_axs.legend(loc='upper right')
sample_fig.show()

sample_fig.savefig("plots/test_distributions/July_11/nspline_8_flows_6hl_64hu_2000bs_normalized_resample.pdf")

data_tensor = eval_test_data
num_bins = 5

sample_means = np.empty((10,10))
data_means = np.empty((10,10))

# Extract the relevant features
feature1 = data_tensor[:, 0].numpy()
feature2 = data_tensor[:, 2].numpy()
target_feature = data_tensor[:, 4].numpy()

samples_np = samples.numpy()

# Calculate bin edges for feature1 and feature2
f1_bins = np.linspace(feature1.min(), feature1.max(), num_bins + 1)
f2_bins = np.linspace(feature2.min(), feature2.max(), num_bins + 1)

# Create the figure and subplots
fig, axs = plot.subplots(num_bins, num_bins, figsize=(30, 30), sharex=True)
fig.suptitle(r"8 flows Histograms of 5th Feature Binned by z hit pos (x) and $\theta$ (y)", fontsize=16)

# Iterate through the grid
for i in range(num_bins):
    for j in range(num_bins):
        # Select data points in the current bin
        mask = (
            (feature1 >= f1_bins[j]) & (feature1 < f1_bins[j+1]) &
            (feature2 >= f2_bins[i]) & (feature2 < f2_bins[i+1])
        )
        bin_data = target_feature[mask]
        bin_samples_np = samples_np[mask]
        
        sample_means[i,j] = np.mean(bin_samples_np)
        data_means[i,j] = np.mean(bin_data)
        
        data_max = max(bin_data)
        data_min = min(bin_data)
        
        samples_max = max(bin_samples_np)
        samples_min = min(bin_samples_np)
        
        if(samples_max - samples_min > data_max - data_min):
            n_bins_samples = 80
            bin_width = (samples_max - samples_min) / n_bins_samples
            n_bins_data = int(round((data_max - data_min) / bin_width))
        else:
            n_bins_data = 80
            bin_width = (data_max - data_min) / n_bins_data
            n_bins_samples = int(round((samples_max - samples_min) / bin_width))
        # Plot histogram in the current subplot
        axs[i, j].hist(bin_data, bins=n_bins_data, color = 'red',alpha = 0.5)
        axs[i, j].hist(bin_samples_np, bins=n_bins_samples, color = 'blue',alpha = 0.5)
#         axs[i, j].set_title(fr'Hit z pos: [{f1_bins[j]:.2f}, {f1_bins[j+1]:.2f})\n $\theta$: [{f2_bins[i]:.2f}, {f2_bins[i+1]:.2f})')

# Set labels for the outer subplots
for ax in axs[-1, :]:
    ax.set_xlabel('photon hit time')
for ax in axs[:, 0]:
    ax.set_ylabel('Count')

# Add overall x and y labels
# fig.text(0.5, 0.04, '1st Feature Bins', ha='center', va='center', fontsize=14)
# fig.text(0.06, 0.5, '2nd Feature Bins', ha='center', va='center', rotation='vertical', fontsize=14)

plot.tight_layout()
plot.show()
fig.savefig("plots/test_distributions/Binned/July_11/8_flows_run_3_5x5_binned_no_normalized.jpeg")
# Example usage:
# Assuming you have a PyTorch tensor named 'data_tensor' with shape [x, 5]
# create_histogram_grid(data_tensor)

f2_plot_y = np.empty(100)
f1_plot_x = f1_bins[:10]
for i in range(100):
    f2_plot_y[i] = f2_bins[i // 10]
#     print(i // 10)
for i in range(9):
    f1_plot_x = np.append(f1_plot_x, f1_bins[:10])

fig_means, axs_means = plot.subplots(1,2,figsize = (12,6))
fig_means.suptitle("data distribution vs sampled distribution")
axs_means[0].scatter(f1_plot_x,f2_plot_y,c = sample_means.flatten(),cmap = 'gray',s = 900)
axs_means[0].set_title("DATA timing means in z_pos and theta binnings")
axs_means[0].set_xlabel("z hit position (mm)")
axs_means[0].set_ylabel("hit theta (degrees)")
axs_means[1].scatter(f1_plot_x,f2_plot_y,c = data_means.flatten(),cmap = 'gray',s = 900)
axs_means[1].set_title("SAMPLED timing means in z_pos and theta binnings")
axs_means[1].set_xlabel("z hit position (mm)")
axs_means[1].set_ylabel("hit theta (degrees)")
fig_means.savefig("plots/test_distributions/2d/July_11/run_3_10x10means.jpeg")