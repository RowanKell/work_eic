import torch
import numpy as np

Timing_path = "/cwork/rck32/eic/work_eic/macros/Timing_estimation/"

#training data produced by preprocess
raw_inputs = torch.load(Timing_path + "data/July_24/Run_1/Vary_p_events_file_0_July_23_600_z_pos.pt")
for i in range(1,600):
    raw_inputs = torch.cat((raw_inputs, torch.load(Timing_path + f"data/July_24/Run_1/Vary_p_events_file_{i + 1}_July_23_600_z_pos.pt")),0)
    
inputs = raw_inputs[np.logical_and(raw_inputs[:,4] < 100,raw_inputs[:,3] < 0.06)]

indexes = torch.randperm(inputs.shape[0])
dataset = inputs[indexes]
train_frac = 0.08
test_frac = 0.01
val_frac = 0.01
train_lim = int(np.floor(dataset.shape[0] * train_frac))
test_lim = train_lim + int(np.floor(dataset.shape[0] * test_frac))
val_lim = test_lim + int(np.floor(dataset.shape[0] * val_frac))
train_data = dataset[:train_lim]
test_data = dataset[train_lim:test_lim]
val_data = dataset[test_lim:val_lim]

train_data.save(Timing_path + "data/combined/July_23/tenth_600_z_pos_train.pt")
test_data.save(Timing_path + "data/combined/July_23/tenth_600_z_pos_test.pt")
val_data.save(Timing_path + "data/combined/July_23/tenth_600_z_pos_val.pt")