import uproot
import numpy as np
import torch
from collections import defaultdict
from util import get_layer, theta_func,create_layer_map, calculate_num_pixels_z_dependence
from time_res_util import get_compiled_NF_model
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
import datetime
import pathlib
import pandas as pd
import json

from momentum_prediction_util import Predictor,train,prepare_prediction_input_pulse,new_prepare_nn_input
import argparse



def create_nested_defaultdict():
    """Recreate the nested defaultdict structure."""
    return defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
def convert_dict_to_defaultdict(d, factory):
    """Convert a dictionary back to nested defaultdict."""
    result = factory()
    
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = convert_dict_to_defaultdict(v, factory)
        else:
            result[k] = v
    return result
def load_defaultdict(filename):
    """Load data from JSON file into nested defaultdict."""
    # Read the JSON file
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Convert back to nested defaultdict
    return convert_dict_to_defaultdict(data, create_nested_defaultdict)

parser = argparse.ArgumentParser(description = 'Preparing data for momentum prediction training')

parser.add_argument('--inputProcessedData', type=str, default="NA",
                        help='directory of input np dict') 
parser.add_argument('--outputDataframePathName', type=str, default="NA",
                        help='directory of output df') 
args = parser.parse_args()
outputDataframePathName = args.outputDataframePathName
inputProcessedData = args.inputProcessedData


# file_name = f"n_5kevents_0_8_to_10GeV_90theta_origin_file_{file_num}.edm4hep.root"

layer_map, super_layer_map = create_layer_map()

x = datetime.datetime.now()
today = x.strftime("%B_%d")

model_compile = get_compiled_NF_model()

# print("Starting process_root_file")
# begin = time.time()
# processed_data = process_root_file(filePathName)
# end = time.time()
# print(f"process_root_file took {(end - begin) / 60} minutes")
processed_data = load_defaultdict(inputProcessedData)
print("Starting prepare_nn_input")
begin = time.time()
nn_input,nn_output = new_prepare_nn_input(processed_data, model_compile,batch_size = 50000)
end = time.time()
print(f"new_prepare_nn_input took {(end - begin) / 60} minutes")

print("Starting prepare_prediction_input")
begin = time.time()
df = prepare_prediction_input_pulse(nn_input,nn_output)
end = time.time()
print(f"prepare_prediction_input_pulse took {(end - begin) / 60} minutes")

df.to_csv(outputDataframePathName)

print("finished job")