import uproot
import numpy as np
import torch
from collections import defaultdict
from util import get_layer, theta_func, calculate_num_pixels_z_dependence
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
from pathlib import Path

from momentum_prediction_util import Predictor,train,prepare_prediction_input_pulse,create_nested_defaultdict,convert_dict_to_defaultdict,load_defaultdict,newer_prepare_nn_input_slab
import argparse

parser = argparse.ArgumentParser(description = 'Preparing data for momentum prediction training')

parser.add_argument('--inputProcessedData', type=str, default="NA",
                        help='directory of input np dict') 
parser.add_argument('--outputDataframePathName', type=str, default="NA",
                        help='directory of output df') 
parser.add_argument('--scintThickness', type=str, default="2cm",
                        help='Thickness of scintillator in geometry') 
parser.add_argument('--batchSize', type=int, default=50000,
                        help='Size of batches for inference') 
parser.add_argument('--useCFD', action=argparse.BooleanOptionalAction,
                        help='If true, uses constant fraction discrimination. Otherwise, uses leading edge.') 
parser.add_argument('--deleteJSON', action=argparse.BooleanOptionalAction,
                        help='.') 

args = parser.parse_args()
outputDataframePathName = args.outputDataframePathName
inputProcessedData = args.inputProcessedData
scintThickness = args.scintThickness
useCFD = args.useCFD
batch_size = args.batchSize
deleteJSON = args.deleteJSON

model_compile = get_compiled_NF_model(thickness = scintThickness)

processed_data = load_defaultdict(inputProcessedData)

print("Starting prepare_nn_input")
begin = time.time()
ret_df = newer_prepare_nn_input_slab(processed_data, model_compile,batch_size = batch_size,useCFD = useCFD)
end = time.time()
print(f"new_prepare_nn_input took {(end - begin) / 60} minutes")

ret_df.to_csv(outputDataframePathName)
df_file = Path(outputDataframePathName)
print(f"saved df at {outputDataframePathName}")
# Ensure data is stored in df before deleting root/process files
if(deleteJSON):
    if(df_file.is_file()):
        process_file = Path(inputProcessedData)
        if(process_file.is_file()):
            process_file.unlink()
            print(f"deleted process file at {inputProcessedData}")


print("finished job")
