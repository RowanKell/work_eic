import numpy as np
import time
import pickle
from ExtractCellID import process_root_file_to_csv
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

print("Starting process_root_file")
begin = time.time()
processed_data = process_root_file_to_csv(filePathName)
end = time.time()
print(f"process_root_file took {(end - begin) / 60} minutes")

processed_data.to_csv(processedDataPath)