import uproot
import numpy as np
import torch
from collections import defaultdict
from util import get_layer, theta_func,create_layer_map
from reco import calculate_num_pixels_z_dependence
import matplotlib.pyplot as plot
import time
from collections import defaultdict
# Get device to be used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os
def checkdir(path):
    if not os.path.exists(path): 
        os.makedirs(path)
from IPython.display import clear_output
from tqdm import tqdm
import normflows as nf
import datetime
torch.cuda.set_device(0)  # Use the first GPU
torch.backends.cudnn.enabled = False
from momentum_prediction_util import process_root_file,prepare_nn_input,prepare_prediction_input,Predictor,train

layer_map, super_layer_map = create_layer_map()

x = datetime.datetime.now()
today = x.strftime("%B_%d")

run_num = 10
hidden_dim_factor = 4
num_hlayers = 10


pref = "/hpc/group/vossenlab/rck32/"


num_files = 20
for i in range(num_files):
    if(i == 0):
        prediction_input = torch.load(pref + f"eic/work_eic/macros/Timing_estimation/data/momentum_prediction/sept_12_5k_n_input_slurm_0.pt")
        prediction_output = torch.load(pref + f"eic/work_eic/macros/Timing_estimation/data/momentum_prediction/sept_12_5k_n_output_slurm_0.pt")
    else:
        prediction_input = torch.cat((prediction_input,torch.load(pref + f"eic/work_eic/macros/Timing_estimation/data/momentum_prediction/sept_12_5k_n_input_slurm_{i}.pt")))
        prediction_output = torch.cat((prediction_output,torch.load(pref + f"eic/work_eic/macros/Timing_estimation/data/momentum_prediction/sept_12_5k_n_output_slurm_{i}.pt")))
num_layers = 28
num_pixels_per_layer = 10


model = Predictor(input_size=num_layers * num_pixels_per_layer, num_classes=1, hidden_dim = 280 * hidden_dim_factor, num_layers = num_hlayers)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-7, weight_decay=1e-5)
from torch import nn
def train(predictor, train_data,nn_output,optimizer,device, num_epochs = 18, batch_size = 100, show_progress = True):
    
    criterion = nn.MSELoss()
    predictor.train()
    total_data_points = train_data.shape[0]
    num_it = total_data_points // batch_size

    show_progress = True
    loss_hist = []
    curr_losses = []
    for i in range(num_epochs):
        clear_output(wait=True)
        print(f"Training epoch #{i}")
        epoch_hist = np.array([])
        val_epoch_hist = np.array([])
        with tqdm(total=num_it, position=0, leave=True) as pbar:
            for it in range(num_it):
                optimizer.zero_grad()
                begin = it * batch_size
                end = min((begin + batch_size),(total_data_points - begin))
                context_inputs = train_data[begin:end].flatten(start_dim = 1).to(device)
                expected_outputs = nn_output[begin:end].unsqueeze(-1).to(device)
                outputs = predictor(context_inputs)
                if(i == 19 and it == 0):
                    print("expected_outputs:")
                    print(expected_outputs)
                    print("actual outputs:")
                    print(outputs)
                loss = criterion(outputs, expected_outputs)
                # Do backprop and optimizer step
                if ~(torch.isnan(loss) | torch.isinf(loss)):
                    if(loss > 200):
                        continue
                    loss.backward()
                    optimizer.step()
                # Log loss
                if~(torch.isnan(loss)):
                    curr_losses.append(loss.to('cpu').data.numpy())
                    if(not (it % 5)):
                        loss_hist.append(sum(curr_losses) / len(curr_losses))
                        curr_losses = []
                if(show_progress):
                    pbar.update(1)
        

    print('Finished Training')
    return loss_hist
loss_hist = train(model,prediction_input,prediction_output,optimizer,device,num_epochs = 20, batch_size = 128)
model.save(pref + f"eic/work_eic/macros/Timing_estimation/models/Momentum_prediction/run_{run_num}_20_epoch_128bs_20files_5kevents_n_0_8_10GeV.pth")
plot.plot(loss_hist);
plot.savefig(pref + f"eic/work_eic/macros/Timing_estimation/plots/train_momentum_prediction/run_{run_num}_hidden_dim_factor_{hidden_dim_factor}_num_layers_{num_hlayers}_loss_20_epochs.jpeg")


#testing
test_inputs = torch.load(pref + f"eic/work_eic/macros/Timing_estimation/data/momentum_prediction/sept_12_5k_n_input_slurm_{num_files}.pt").to(device)[:1000]
test_outputs = torch.load(pref + f"eic/work_eic/macros/Timing_estimation/data/momentum_prediction/sept_12_5k_n_output_slurm_{num_files}.pt")[:1000]

model.eval()
predictions = model(test_inputs.flatten(start_dim = 1)).detach().cpu()
criterion = nn.MSELoss()
print(f"test_loss: {criterion(predictions,test_outputs)}")

fig_test, axs_test = plot.subplots(1,1)
axs_test.scatter(test_outputs,predictions,alpha = 0.1)
axs_test.set_xlabel("Expected")
axs_test.set_ylabel("Predicted")
axs_test.plot(range(10),range(10),color = 'r')
fig_test.suptitle("Real vs Learned Momentums")
fig_test.savefig(pref + f"eic/work_eic/macros/Timing_estimation/plots/train_momentum_prediction/predictions_vs_real_run_{run_num}_hidden_dim_factor_{hidden_dim_factor}_num_layers_{num_hlayers}.pdf")

'''from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(model, test_in,test_out, device):
    model.eval()  # Set the model to evaluation mode
    max_index = test_data.shape[1] - 1
    
    with torch.no_grad():
        samples = test_in.float().to(device)
        true_values = test_out.float().cpu().numpy()
        
        predictions = model(samples).cpu().numpy().flatten()
    
    # Calculate metrics
    mae = mean_absolute_error(true_values, predictions)
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_values, predictions)
    
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R-squared Score: {r2:.4f}")
    plot_predictions(true_values, predictions)
    return mae, mse, rmse, r2

#test_inputs = torch.load(pref + f"eic/work_eic/macros/Timing_estimation/data/momentum_prediction/sept_12_5k_n_input_slurm_11.pt")[:128]
#test_outputs = torch.load(pref + f"eic/work_eic/macros/Timing_estimation/data/momentum_prediction/sept_12_5k_n_output_slurm_11.pt")[:128]
# evaluate_model(model,test_inputs,test_outputs,device)
'''