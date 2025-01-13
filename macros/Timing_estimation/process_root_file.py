import datetime
def print_w_time(message):
    current_time = datetime.datetime.now().strftime('%H:%M:%S')
    print(f"{current_time} {message}")

import numpy as np
import time
import pickle
from ExtractCellID import process_root_file_to_csv,process_root_file_old
from collections import defaultdict
import argparse
import json
import pandas as pd

parser = argparse.ArgumentParser(description = 'Preparing data for momentum prediction training')

parser.add_argument('--filePathName', type=str, default="NA",
                        help='directory of root file') 
parser.add_argument('--processedDataPath', type=str, default="NA",
                        help='directory to output np dict')
args = parser.parse_args()
filePathName = args.filePathName
processedDataPath = args.processedDataPath
# print("Starting process_root_file")
# begin = time.time()
# processed_data = process_root_file(filePathName)
# end = time.time()
# print(f"process_root_file took {(end - begin) / 60} minutes")

print_w_time("Starting process_root_file")
begin = time.time()
processed_data_csv = process_root_file_to_csv(filePathName)
end = time.time()
print_w_time(f"process_root_file took {(end - begin) / 60} minutes")

processed_data_csv.to_csv(processedDataPath)
print_w_time("saved processed root file data to csv")