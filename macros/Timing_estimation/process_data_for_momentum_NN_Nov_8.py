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

from momentum_prediction_util import Predictor,train,prepare_prediction_input_pulse,new_prepare_nn_input,create_nested_defaultdict,convert_dict_to_defaultdict,load_defaultdict,generateSiPMOutput
import argparse

parser = argparse.ArgumentParser(description = 'Preparing data for momentum prediction training')

parser.add_argument('--inputProcessedData', type=str, default="NA",
                        help='directory of input np dict') 
parser.add_argument('--outputDataframePathName', type=str, default="NA",
                        help='directory of output df') 
args = parser.parse_args()
outputDataframePathName = args.outputDataframePathName
inputProcessedData = args.inputProcessedData

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

layer_map, super_layer_map = create_layer_map()

x = datetime.datetime.now()
today = x.strftime("%B_%d")

model_compile = get_compiled_NF_model()

processed_data = pd.read_csv(inputProcessedData)
print("Starting prepare_nn_input")
begin = time.time()
df = generateSiPMOutput(processed_data, model_compile,batch_size = 50000)
end = time.time()
print(f"new_prepare_nn_input took {(end - begin) / 60} minutes")
df.to_csv(outputDataframePathName)

'''
print("Starting prepare_prediction_input")
begin = time.time()
df = prepare_prediction_input_pulse(nn_input,nn_output)
end = time.time()
print(f"prepare_prediction_input_pulse took {(end - begin) / 60} minutes")

df.to_csv(outputDataframePathName)
'''
print("finished job")
print("analyzing memory snapshot")

snapshot = tracemalloc.take_snapshot()

display_top(snapshot)