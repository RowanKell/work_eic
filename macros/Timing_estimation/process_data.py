import numpy as np
import time
import pickle
from ExtractCellID import process_root_file
from collections import defaultdict
import argparse
import json

def convert_key_to_serializable(key):
    """Convert numpy types to native Python types."""
    if isinstance(key, np.integer):
        return int(key)
    if isinstance(key, np.floating):
        return float(key)
    if isinstance(key, np.ndarray):
        return key.tolist()
    return key

def convert_defaultdict_to_dict(d):
    """Convert a nested defaultdict to a regular dictionary with serializable keys."""
    if isinstance(d, defaultdict):
        # Convert each key-value pair, ensuring keys are serializable
        return {convert_key_to_serializable(k): convert_defaultdict_to_dict(v) 
                for k, v in d.items()}
    elif isinstance(d, dict):
        # Handle regular dictionaries the same way
        return {convert_key_to_serializable(k): convert_defaultdict_to_dict(v) 
                for k, v in d.items()}
    elif isinstance(d, np.ndarray):
        return d.tolist()
    elif isinstance(d, (np.integer, np.floating)):
        return convert_key_to_serializable(d)
    return d



def save_defaultdict(data, filename):
    """Save nested defaultdict to a JSON file."""
    # Convert defaultdict to regular dict for serialization
    regular_dict = convert_defaultdict_to_dict(data)
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(regular_dict, f)
        


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
processed_data = process_root_file(filePathName)
end = time.time()
print(f"process_root_file took {(end - begin) / 60} minutes")

save_defaultdict(processed_data,processedDataPath)