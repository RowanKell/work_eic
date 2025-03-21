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
import pathlib

from momentum_prediction_util import process_root_file,prepare_nn_input,prepare_prediction_input,Predictor,train,prepare_prediction_input_pulse,new_prepare_nn_input


import argparse
parser = argparse.ArgumentParser(description = 'Preparing data for momentum prediction training')

parser.add_argument('--filePathName', type=str, default="NA",
                        help='directory of root file') 
parser.add_argument('--inputTensorPathName', type=str, default="NA",
                        help='directory of input tensors') 
parser.add_argument('--outputTensorPathName', type=str, default="NA",
                        help='directory of output tensors') 
args = parser.parse_args()
filePathName = args.filePathName
inputTensorPathName = args.inputTensorPathName
outputTensorPathName = args.outputTensorPathName

Tensor_parent = str(pathlib.Path(outputTensorPathName).parent)

# file_name = f"n_5kevents_0_8_to_10GeV_90theta_origin_file_{file_num}.edm4hep.root"

layer_map, super_layer_map = create_layer_map()

x = datetime.datetime.now()
today = x.strftime("%B_%d")

run_num = 7
run_num_str = str(run_num)

#NF Stuff

K = 8 #num flows

latent_size = 1 #dimension of PDF
hidden_units = 256 #nodes in hidden layers
hidden_layers = 26
context_size = 3 #conditional variables for PDF
num_context = 3

K_str = str(K)
batch_size= 2000
hidden_units_str = str(hidden_units)
hidden_layers_str = str(hidden_layers)
batch_size_str = str(batch_size)
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
# model_date = "August_03"
# today = "August_03"
# model_path = "models/" + model_date + "/"
# checkdir(model_path)

model_path = "/hpc/group/vossenlab/rck32/NF_time_res_models/"

checkdir(Tensor_parent)

model.load(model_path + "run_" + run_num_str + "_" + str(num_context)+ "context_" +K_str +  "flows_" + hidden_layers_str+"hl_" + hidden_units_str+"hu_" + batch_size_str+"bs.pth")
model = model.to(device)
model_compile = torch.compile(model,mode = "reduce-overhead")
model_compile = model_compile.to(device)
print("Starting process_root_file")
start = time.time()
processed_data = process_root_file(filePathName,max_events = 500)
end = time.time()

print(f"Finished running process_root_file, took {(end - start) / 60} minutes")


# print("Starting optimized prepare_nn_input")
begin = time.time()
nn_input, nn_output = optimized_prepare_nn_input(processed_data, model_compile,batch_size = 50000)
end = time.time()

print(f"optimized method took {(end - begin) / 60} minutes")
# print("starting original prepare_nn_iput")
begin = time.time()
nn_input, nn_output = new_prepare_nn_input(processed_data, model_compile,batch_size = 50000)
end = time.time()
print(f"original method took {(end - begin) / 60} minutes")


print("Starting prepare_prediction_input")
begin = time.time()
prediction_input, prediction_output= prepare_prediction_input_pulse(nn_input,nn_output)
end = time.time()
print(f"prepare_prediction_input_pulse took {(end - begin) / 60} minutes")
torch.save(prediction_input,inputTensorPathName)
torch.save(prediction_output,outputTensorPathName)
print("finished job")