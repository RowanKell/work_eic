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

if(args.parallel == 1):
    num_files_process_start = args.file_num
    num_files_process_end = args.file_num + 1
else:
    num_files_process_start = args.file_num
    num_files_process_end = args.file_num + args.num_files

data_save_path = args.outfile
break_limit = 2000 #max 4000
num_files_process = 21 #max 21
theta_test = False

# Import packages
import os
import torch
import numpy as np
import normflows as nf

import uproot as up

from matplotlib import pyplot as plot
import math
from tqdm import tqdm
from util import PVect, theta_func, r_func,z_func,time_func
from IPython.display import clear_output
# Get device to be used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("finished imports")


'''
Data preprocessing
'''
err_z = []
err_theta = []
err_mu_time = []
err_P = []
init = False
inputs = torch.ones(1,5) if not theta_test else torch.ones(1,4)
for file_num in range(num_files_process_start,num_files_process_end):
    print(f"Beginning file #{file_num}")
    uproot_path = f"/cwork/rck32/eic/work_eic/root_files/July_2/slurm/mu_vary_p_z_theta_no_save_all/vary_p_8000events_{file_num}_cos_theta_distribution.edm4hep.root:events"
    with up.open(uproot_path) as events:
        hit_time_branch = events["HcalBarrelHits.time"].array(library='np')
        hit_x_pos_branch = events["HcalBarrelHits.position.x"].array(library='np')
        hit_z_pos_branch = events["HcalBarrelHits.position.z"].array(library='np')
        Hits_MC_idx_branch = events["_HcalBarrelHits_MCParticle.index"].array(library='np')


        x_pos_branch = events["MCParticles/MCParticles.vertex.x"].array(library='np')
        z_pos_branch = events["MCParticles/MCParticles.vertex.z"].array(library='np')
        PDG_branch = events["MCParticles.PDG"].array(library='np')
        mass_branch = events["MCParticles.mass"].array(library='np')

        momentum_x_branch = events["MCParticles.momentum.x"].array(library='np')
        momentum_y_branch = events["MCParticles.momentum.y"].array(library='np')
        momentum_z_branch = events["MCParticles.momentum.z"].array(library='np')

        emission_time_branch = events["MCParticles/MCParticles.time"].array(library='np')
        daughter_end_branch = events["MCParticles/MCParticles.daughters_end"].array(library='np')
        daughter_begin_branch = events["MCParticles/MCParticles.daughters_begin"].array(library='np')
        
        for event_idx in range(len(PDG_branch)):
            print(f"Working on file #{file_num}: {event_idx} events done")
            if(break_limit > 0 and event_idx > break_limit):
                break
            #Primary stuff
            primary = PVect()
            primary.setVector(momentum_x_branch[event_idx][0],momentum_y_branch[event_idx][0],momentum_z_branch[event_idx][0],mass_branch[event_idx][0])
            theta = primary.theta
            P = primary.P
            px = primary.px
            m = primary.M
            vertex_x = x_pos_branch[event_idx][0]
            vertex_z = z_pos_branch[event_idx][0]
            mu_incident_time = time_func(px,m,1770.3 - vertex_x) #for one_segment, bar starts at 1770.3 mm
            hit_z = z_func(vertex_z, theta)
            
            '''
            cuts to avoid wacky data
            '''
            if (hit_z > 770 or hit_z < -735):#should be between -732->767
                err_z.append(hit_z)
                continue
            if(theta > 180 or theta < 0): #should be between 0 180
                err_theta.append(theta)
                continue
            if(mu_incident_time < 0): #only positive times...
                err_mu_time.append(mu_incident_time)
                continue
            if(P < 0.1 or P > 10): #rn only shooting between 0.1 and 1
                err_P.append(P)
                continue
            
            if(not theta_test):
                for hit_idx in range(len(hit_z_pos_branch[event_idx])):
                    if(PDG_branch[event_idx][Hits_MC_idx_branch[event_idx][hit_idx]] != -22):
                        continue
                    #conditionals
                    hit_tensor = torch.ones(1,5) * 10
                    hit_tensor[0,0] = float(hit_z)
                    hit_tensor[0,1] = float(mu_incident_time)
                    hit_tensor[0,2] = float(theta)
                    hit_tensor[0,3] = float(P)
                    #features
                    sensor_time = hit_time_branch[event_idx][hit_idx]
                    hit_tensor[0,4] = float(sensor_time)
                    if(not init):
                        inputs[0] = hit_tensor[0]
                        init = True
                    else:
                        inputs = torch.cat((inputs,hit_tensor),0)
            else:
                hit_tensor = torch.ones(1,4) * 10
                hit_tensor[0,0] = float(hit_z)
                hit_tensor[0,1] = float(mu_incident_time)
                hit_tensor[0,2] = float(theta)
                hit_tensor[0,3] = float(P)
                if(not init):
                    inputs[0] = hit_tensor[0]
                    init = True
                else:
                    inputs = torch.cat((inputs,hit_tensor),0)
torch.save(inputs,data_save_path)
print(f"error P values: {len(err_P)}\n")
for i in range(len(err_P)):
    print(f"P val #{i}: {err_P[i]}")
    
print(f"error z values: {len(err_z)}\n")
for i in range(len(err_z)):
    print(f"z val #{i}: {err_z[i]}")
    
print(f"error theta values: {len(err_theta)}\n")
for i in range(len(err_theta)):
    print(f"theta val #{i}: {err_theta[i]}")
    
print(f"error mu_time values: {len(err_mu_time)}\n")
for i in range(len(err_mu_time)):
    print(f"mu_time val #{i}: {err_mu_time[i]}")