import datetime
def print_w_time(message):
    current_time = datetime.datetime.now().strftime('%H:%M:%S')
    print(f"{current_time} {message}")

print_w_time("began analyze_data")

import uproot
import numpy as np
from  torch import device as torchdevice
from torch.cuda import is_available as torchcudaisavailable
import matplotlib.pyplot as plot
import time
# Get device to be used
device = torchdevice('cuda' if torchcudaisavailable() else 'cpu')
from os.path import exists as ospathexists
from os import makedirs as osmakedirs
def checkdir(path):
    if not ospathexists(path): 
        osmakedirs(path)
from tqdm import tqdm
import datetime
from  pandas import read_csv as pd_read_csv
print_w_time("finished library imports in analyze data")
from util import get_layer, theta_func,create_layer_map, calculate_num_pixels_z_dependence
from time_res_util import get_compiled_NF_model
from momentum_prediction_util import Predictor,train,prepare_prediction_input_pulse,new_prepare_nn_input,create_nested_defaultdict,convert_dict_to_defaultdict,load_defaultdict,generateSiPMOutput
import argparse
print_w_time("finished local imports in analyze data")
parser = argparse.ArgumentParser(description = 'Preparing data for momentum prediction training')

parser.add_argument('--inputProcessedData', type=str, default="NA",
                        help='directory of input np dict') 
parser.add_argument('--outputDataframePathName', type=str, default="NA",
                        help='directory of output df') 
args = parser.parse_args()
outputDataframePathName = args.outputDataframePathName
inputProcessedData = args.inputProcessedData
print_w_time("got to main part of analyze_data")
'''MEMORY PROFILING'''
import linecache
import os
import tracemalloc

def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))

tracemalloc.start()
    
'''MEMORY PROFILING SETUP END}'''

model_compile = get_compiled_NF_model()
print_w_time("loaded nf model")
processed_data = pd_read_csv(inputProcessedData)
print_w_time("Opened processed data... Starting generateSiPMOutput")
begin = time.time()
df = generateSiPMOutput(processed_data, model_compile,batch_size = 50000)
end = time.time()
print_w_time(f"generateSiPMOutput took {(end - begin) / 60} minutes")

def print_df_info(df):
    print(f"DataFrame has {len(df)} rows.")
    print("Column names:")
    for col in df.columns:
        print(f"- {col}")
print_df_info(df)
print_w_time("finished job")
print_w_time("analyzing memory snapshot")

snapshot = tracemalloc.take_snapshot()

display_top(snapshot)