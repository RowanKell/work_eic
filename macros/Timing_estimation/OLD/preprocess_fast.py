print("started preprocess.py")

'''
User defined params
'''
import argparse

parser = argparse.ArgumentParser(description='Preprocessing program to prepare data for NF')
parser.add_argument('--outfile', type=str, default="data/test.pt",
                        help='name of pt file to store tensor in (default: data/test.pt)') 
parser.add_argument('--infile', type=str, default="NA",
                        help='name of root file to read in') 
parser.add_argument('--parallel', type=int, default=0,
                        help='name of root file to read in') 
parser.add_argument('--file_num', type=int, default=0,
                        help='name of root file to read in') 
parser.add_argument('--num_files', type=int, default=0,
                        help='name of root file to read in') 
args = parser.parse_args()

file_num = args.file_num
data_save_path = args.outfile
infile = args.infile
break_limit = 4000 #max 4000
num_files_process = 2 #max 21
theta_test = False


import numpy as np
import uproot
import torch
from tqdm import tqdm
from util import PVect, time_func, z_func

def process_file(file_path, break_limit=-1):
    with uproot.open(file_path) as events:
        # Load all required branches at once
        branches = events.arrays([
            "HcalBarrelHits.time", "HcalBarrelHits.position.x", "HcalBarrelHits.position.z",
            "_HcalBarrelHits_MCParticle.index", "MCParticles.vertex.x", "MCParticles.vertex.z",
            "MCParticles.PDG", "MCParticles.mass", "MCParticles.momentum.x",
            "MCParticles.momentum.y", "MCParticles.momentum.z"
        ], library="np")

        # Extract primary particle info
        primary_px = np.array([event[0] for event in branches["MCParticles.momentum.x"]])
        primary_py = np.array([event[0] for event in branches["MCParticles.momentum.y"]])
        primary_pz = np.array([event[0] for event in branches["MCParticles.momentum.z"]])
        primary_m = np.array([event[0] for event in branches["MCParticles.mass"]])
        vertex_x = np.array([event[0] for event in branches["MCParticles.vertex.x"]])
        vertex_z = np.array([event[0] for event in branches["MCParticles.vertex.z"]])

        # Compute derived quantities
        primary = PVect()
        primary.setVector(primary_px, primary_py, primary_pz, primary_m)
        theta = primary.theta
        P = primary.P
        mu_incident_time = time_func(primary.px, primary.M, 1770.3 - vertex_x)
        hit_z = z_func(vertex_z, theta)

        # Apply cuts
        valid_events = (
            (hit_z >= -735) & (hit_z <= 770) &
            (theta >= 0) & (theta <= 180) &
            (mu_incident_time >= 0) &
            (P >= 0.1) & (P <= 10)
        )

        if break_limit > 0:
            valid_events = valid_events & (np.arange(len(valid_events)) < break_limit)

        all_hit_data = []
        for i, valid in enumerate(valid_events):
            if valid:
                hit_pdg = branches["MCParticles.PDG"][i][branches["_HcalBarrelHits_MCParticle.index"][i]]
                valid_hits = (hit_pdg == -22)
                hit_times = branches["HcalBarrelHits.time"][i][valid_hits]
                
                if len(hit_times) > 0:
                    hit_data = np.column_stack([
                        np.full(len(hit_times), hit_z[i]),
                        np.full(len(hit_times), theta[i]),
                        np.full(len(hit_times), P[i]),
                        np.full(len(hit_times), mu_incident_time[i]),
                        hit_times
                    ])
                    all_hit_data.append(hit_data)
#                     if i == 0:print(hit_data) 

        return np.vstack(all_hit_data) if all_hit_data else np.array([])

# Process all files
file_path = infile + f"vary_p_z_th_events_{file_num}_600_z_vals.edm4hep.root:events"
file_data = process_file(file_path, break_limit)

# Combine all data and convert to PyTorch tensor
inputs = torch.tensor(file_data, dtype=torch.float32)
# print(inputs.shape)
# Save the tensor
torch.save(inputs, data_save_path)