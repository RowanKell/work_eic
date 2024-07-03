'''
 General Set of Python Functions and Code saved for later
'''

'''
IMPORTS
'''

#General imports
import numpy as np
import uproot as up
import pandas as pd
import matplotlib.pyplot as plot

#Math
import sympy #acos and atan2
import math

#Updating progress bar/output:
from IPython.display import clear_output

#Timing
import time

#Misc
from scipy.stats import norm

#ML
import torch
import torch.nn as nn

from tqdm import tqdm

#roc-curve calculations
from sklearn.metrics import roc_auc_score, roc_curve

# Fetching the device that will be used throughout this notebook
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)

'''
VARIABLES AND FILES
'''

#setting configuration to update other functions
particle = "mu"
energy = "5"
color_dict = {
    "pi" : "red",
    "mu" : "blue"
}
'''
#general uproot path
uproot_path = f"/cwork/rck32/eic/work_eic/root_files/June_13/variation/full_{particle}_{energy}GeV_10k.edm4hep.root:events"
events = up.open(uproot_path)

#Some events branches
x_pos_branch = events["HcalBarrelHits/HcalBarrelHits.position.x"].array(library='np')
EDep_branch = events["HcalBarrelHits.EDep"].array(library='np')
Pathlength_branch = events["HcalBarrelHits.pathLength"].array(library='np')
Hits_MC_idx_branch = events["_HcalBarrelHits_MCParticle.index"].array(library='np')
MC_parents = events["_MCParticles_parents.index"].array(library='np')
MC_daughters = events["MCParticles.daughters_end"].array(library='np')
PDG_branch = events["MCParticles.PDG"].array(library='np')
MC_endpoint_x_branch = events["MCParticles.endpoint.x"].array(library='np')
MC_px = events["MCParticles.momentum.x"].array(library='np')
MC_py = events["MCParticles.momentum.y"].array(library='np')
MC_pz = events["MCParticles.momentum.z"].array(library='np')
MC_m = events["MCParticles.mass"].array(library='np')
'''
'''
LAYER CODE
'''
#Approx layer map (outdated, use calculation now)
layer_map = [1830.8000, 1841.4000, 1907.5, 1918.1,1984.1999, 1994.8000, 2060.8999, 2071.5, 2137.6001, 2148.1999, 2214.3000, 2224.8999, 2291, 2301.6001, 2367.6999, 2378.3000, 2444.3999, 2455, 2521.1001, 2531.6999, 2597.8000, 2608.3999, 2674.5, 2685.1001, 2751.1999, 2761.8000, 2827.8999, 2838.5]    

#layer_map calculation based on start, distance between super and minor layers
# RETURNS [layer_map,super_layer_map]
def create_layer_map(begin = ((1830.8 + 1841.4) / 2),dis_between_super_layers = 76.7,dis_between_internal_layers = 10.6):
    begin = (1830.8 + 1841.4) / 2
    super_layer_map_calc = np.empty(14)
    layer_map_calc = np.empty(28)
    for i in range(14):
        super_layer_map_calc[i] = begin + dis_between_super_layers * i
    for i in range(28):
        layer_map_calc[i] = (super_layer_map_calc[i // 2] + (i % 2) * dis_between_internal_layers ) - (dis_between_internal_layers / 2)
    return [layer_map_calc,super_layer_map_calc]
layer_map, super_layer_map = create_layer_map()

#Functions to calculate which layer an x position is in without loops
def get_super_layer(x_pos,s_map,s_dis):
    rel_dist = x_pos - (s_map[0] - (s_dis / 2))
    if(rel_dist < -2):
        return -1
    return int(np.floor(rel_dist / (s_dis)))

def get_internal_layer(x_pos,super_layer_idx,s_map):
    rel_dist = x_pos - s_map[super_layer_idx]
    return int((math.copysign(1,rel_dist) + 1) /2)

def get_layer(x_pos,s_map = super_layer_map):
    s_dis = s_map[1] - s_map[0]
    super_layer_idx = get_super_layer(x_pos,s_map,s_dis)
    internal_layer_idx = get_internal_layer(x_pos,super_layer_idx,s_map)
    return int(super_layer_idx * 2 + internal_layer_idx)

# OLD CALCULATION OF LAYER NUMBER:
def get_num_layers_traversed(x_pos):
    #This map contains the midpoints of each layer - each layer is 1cm in width, so we can check the 0.5cm on each side to see if a hit was in the layer
    layer_map = [1830.8000, 1841.4000, 1907.5, 1918.1,1984.1999, 1994.8000, 2060.8999,2071.5,2137.6001,2148.1999,2214.3000,2224.8999,2291,2301.6001,2367.6999,2378.3000,2444.3999,2455,2521.1001,2531.6999,2597.8000,2608.3999,2674.5,2685.1001,2751.1999,2761.8000,2827.8999,2838.5]
    for layer_idx in range(len(layer_map)):
        #check if particle hit within layer
        if ((x_pos >= layer_map[layer_idx] - 5) and (x_pos <= layer_map[layer_idx] + 5)):
            return layer_idx
    #if no layers hit, send error code
    return -1

'''
Particles and angles
'''
#theta = 0: px, py = 0
#theta - xz plane
#phi - xy plane
#https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
def theta_func(px,py,pz):
    return sympy.acos(pz / np.sqrt(px ** 2 + py ** 2 + pz ** 2)) * 180 / np.pi
def phi_func(px,py,pz):
    return sympy.atan2(py,px) * 180 / np.pi
def Efunc(px,py,pz,m):
    return np.sqrt(px**2 + py**2 + pz**2 + m**2)

#My own class similar to give me a four momenta with limited functionality
class PVect:
    def __init__(self):
        self.px = 0
        self.py = 0
        self.pz = 0
        self.theta = 0
        self.phi = 0
        self.E = 0
        self.M = 0
        self.P = 0
    def setVector(self,px,py,pz,m):
        self.px = px
        self.py = py
        self.pz = pz
        self.M = m
        self.E = Efunc(px,py,pz,m)
        self.P = r_func(self.px,self.py,self.pz)
        self.theta = theta_func(px,py,pz)
        self.phi = phi_func(px,py,pz)
def r_func(x,y,z):
    return np.sqrt(x**2 + y**2 + z**2)
#Currently used max and min angles        
theta_min = 67
theta_max = 113
phi_min = -22
phi_max = 22

#fit functions
def exp_curve(x,a,b,c):
    return a * np.exp(-x * b) + c
def poly_8d(x,a,b,c,d,e,f,g,h,i):
    return a * x ** 8 + b * x ** 7 + c * x ** 6 + d * x ** 5 + e * x ** 4 + f * x ** 3 + g * x ** 2 + h * x + i 
def poly_2d(x,a,b,c):
    return a * x ** 2 + b * x + c
def inverse(x,a,b,c):
    return a  / (x + b) + c

def p_func(x,y,z):
    return np.sqrt((x ** 2) + (y ** 2) + (z ** 2))
def calculate_num_pixels(energy_dep):
    efficiency = 0.005
    return 10 * energy_dep * (1000 * 1000) * efficiency

def calculate_num_pixels_z_dependence(energy_dep, z_hit):
    efficiency = inverse(770 - z_hit,494.98,9.9733,-0.16796)
    return 10 * energy_dep * (1000 * 1000) * efficiency

part_dict = {
    -211 : 1,
    13 : 0
}
'''
Loops
'''
#loop and print angle
def calculate_and_print_angle(EDep_branch,MC_px,MC_py,MC_pz):
    for event_idx in range(len(EDep_branch)):
        if(event_idx > 100):
            break
        num_MC = 0
        primary = PVect()
        primary.setVector(MC_px[event_idx][0],MC_py[event_idx][0],MC_pz[event_idx][0],MC_m[event_idx][0])
        print(f"event # {event_idx}: theta = {primary.theta}; phi = {primary.phi}")
#Loop and print % of photons getting hits
def calculate_percent_photons_hitting(EDep_branch,PDG_branch,MC_daughters):
    hits_per_photon = []
    for event_idx in range(len(EDep_branch)):
        num_MC = 0
        for i in range(len(PDG_branch[event_idx])):
            if(PDG_branch[event_idx][i] == -22 and (i < MC_daughters[event_idx][0])):num_MC+= 1
        if(num_MC == 0):
            print(f"skipping event #{event_idx}, no optph found")
        else:
            hits_per_photon.append(len(EDep_branch[event_idx]) / num_MC)
    print(f"% of photons hitting = {sum(hits_per_photon) / len(hits_per_photon) * 100}%")
    
'''
Binning
'''
#general setup
n_bins = 20
theta_range = theta_max - theta_min
phi_range = phi_max - phi_min
#these arrays will hold the bin edges
phi_bins = np.linspace(phi_min,phi_max,n_bins+1)
theta_bins = np.linspace(theta_min,theta_max,n_bins+1)

#Bin centers:
theta_bin_centers = np.empty(n_bins)
phi_bin_centers = np.empty(n_bins)
for i in range(len(theta_bins) - 1):
    theta_bin_centers[i] = (theta_bins[i] + theta_bins[i+1]) / 2
    phi_bin_centers[i] = (phi_bins[i] + phi_bins[i+1]) / 2

#Find the bin from the value - works only for evenly spaced bins
# bins is the bin edges
def findBin(val,bins):
    diff = bins[1] - bins[0]
    rel_dist = val - bins[0]
    mod = np.floor(rel_dist / diff)
    return int(mod)

#Big function / loop to calculate the % of photons hitting and bin these values
#RETURNS (<list of phi percentage values>,<list of theta percent values>)
def bin_percent_theta_phi(EDep_branch,MC_px,MC_py,MC_pz):
    theta_percent = [[] for i in range(n_bins)]
    phi_percent = [[] for i in range(n_bins)]

    theta_MC = np.zeros(n_bins)
    phi_MC = np.zeros(n_bins)

    theta_hits = np.zeros(n_bins)
    phi_hits = np.zeros(n_bins)

    theta_counts = np.zeros(n_bins)
    phi_counts = np.zeros(n_bins)

    num_MC = 0
    num_hits = 0
    energy_dep = 0
    dep_count = 0
    hits_per_photon = []
    num_events = len(EDep_branch)
    break_val = 10000
    for event_idx in range(len(EDep_branch)):
        if(not (event_idx % (break_val // 100))):
            clear_output(wait = True)
            print(f"{event_idx // (break_val // 100)}% done")
    #     if(event_idx > break_val):
    #         break
        num_MC = 0
        primary = PVect()
        primary.setVector(MC_px[event_idx][0],MC_py[event_idx][0],MC_pz[event_idx][0],MC_m[event_idx][0])
        phi_bin = findBin(primary.phi,phi_bins)
        theta_bin = findBin(primary.theta,theta_bins)
        phi_counts[phi_bin]+=1
        theta_counts[theta_bin]+=1
    #     print(f"theta, phi: {primary.theta}, {primary.phi} | px,pz = {primary.px,primary.pz} | theta, phi bins: {theta_bin},{phi_bin}")
        for i in range(len(PDG_branch[event_idx])):
            if(PDG_branch[event_idx][i] == -22 and (i < MC_daughters[event_idx][0])):
                theta_MC[theta_bin] += 1
                phi_MC[phi_bin] += 1
        for hit in range(len(EDep_branch[event_idx])):
            #check if hit is from optph
            if(PDG_branch[event_idx][Hits_MC_idx_branch[event_idx][hit]] == -22):
                theta_hits[theta_bin] += 1
                phi_hits[phi_bin] += 1
    for i in range(n_bins):
        #if no MCParticles, then just set to -1
        if(not len(theta_MC)):
            theta_percent[i] = -1
        else:
            theta_percent[i] = (theta_hits[i] / theta_MC[i]) * 100
        if(not len(phi_MC)):
            phi_percent[i] = -1
        else:
            phi_percent[i] = (phi_hits[i] / phi_MC[i]) * 100
    return (theta_percent,phi_percent)

#Plot percentages
def plot_percent(phi_bin_centers,theta_bin_centers,phi_percent,theta_percent, output_path = "plots/percentage/June_14_mu_5GeV_10k_phi_theta_20_bins.jpeg"):
    fig, (ax1,ax2) = plot.subplots(1,2, figsize=(10,4))
    fig.suptitle("Percentage of photons reaching sensor as a fraction of total photons generated")
    ax1.scatter(phi_bin_centers,phi_percent,color="red")
    ax1.set_xlabel("phi (deg)")
    ax1.set_ylabel("percentage")
    ax2.scatter(theta_bin_centers,theta_percent,color = "blue")
    ax2.set_xlabel("theta (deg)")
    fig.show()
    fig.savefig(output_path)
    

    
'''
Energy deposited
'''
#Find total energy deposited by primary or all particles in each layer avg by EVENT
# RETURNS (layer_EDep,super_layer_EDep)
def energy_dep_event(x_pos_branch,EDep_branch,Hits_MC_idx_branch,particle = "mu",energy = 5,scope = "primary",save_path = "plots/June_13/avg_event_dep/"):
    #initialize list that we will fill with # of layers traversed for each event
    layers_traversed = []
    skip_count = 0
    #loop over each event
    layer_EDep = np.zeros(28)

    for event_idx in range(len(x_pos_branch)):
        event_x_pos = x_pos_branch[event_idx]
        event_layer_hits = []
        event_EDep = EDep_branch[event_idx]
        layer_hit_bool = np.zeros(28)
        #for each event, loop over the particles to find mu/pi
        for hit_idx in range(len(event_x_pos)):
            if(scope == "primary" and (Hits_MC_idx_branch[event_idx][hit_idx] != 0)):
                continue
            current_x_pos = event_x_pos[hit_idx]
            current_EDep = event_EDep[hit_idx]
            layer_hit = get_num_layers_traversed(current_x_pos)
    #         if(layer_hit_bool[hit_idx] == 1):
    #             continue
    #         layer_hit_bool[hit_idx] = 1
            if(layer_hit == -1):
                skip_count += 1
                continue
            layer_EDep[layer_hit] += current_EDep
    super_layer_EDep = np.zeros(14)
    for i in range(len(layer_EDep)):
        super_layer_num = int(np.floor(i / 2))
        super_layer_EDep[super_layer_num] += layer_EDep[i]
    super_layer_EDep = super_layer_EDep / len(x_pos_branch)
    layer_EDep = layer_EDep / len(x_pos_branch)
    print(f"skipped {skip_count} events")

    fig,(ax1,ax2) = plot.subplots(1,2,figsize=(12,6))
    fig.suptitle(f"{energy}GeV {particle}-: energy deposited by layer avg over 5k events by {scope}")
    ax1.set_title("Super layer energy deposition")
    ax2.set_title("Individual layer energy deposition")

    ax1.set_xlabel("superlayer number")
    ax2.set_xlabel("layer number")

    ax1.set_ylabel("avg energy deposited per event (GeV)")
    ax1.scatter(range(14),super_layer_EDep,10)
    ax2.scatter(layer_map,layer_EDep,10,color = 'r',marker='o')
    fig.show()
    fig.savefig(save_path + f"{particle}_{energy}GeV_{scope}_avg_event.jpeg")
    return (layer_EDep,super_layer_EDep)

# Energy deposited by particle avg by HIT
# RETURNS (energy_means,super_layer_means)
def energy_dep_hit(x_pos_branch,EDep_branch,Hits_MC_idx_branch,gen_status_branch,particle = "mu",energy = 5,scope = "primary",save_path = "plots/June_13/avg_event_dep/"):

    #initialize list that we will fill with # of layers traversed for each event
    layers_traversed = []
    skip_count = 0
    #loop over each event
    layer_EDep = np.zeros(28)
    energies = [[] for i in range(28)]
    energy_means = np.empty(28)
    for event_idx in range(len(x_pos_branch)):
        event_x_pos = x_pos_branch[event_idx]
        event_layer_hits = []
        event_EDep = EDep_branch[event_idx]
        layer_hit_bool = np.zeros(28)
        #for each event, loop over the particles to find mu/pi
        for hit_idx in range(len(event_x_pos)):
            if(scope == "primary" and (not gen_status_branch[event_idx][Hits_MC_idx_branch[event_idx][hit_idx]])):continue
            current_x_pos = event_x_pos[hit_idx]
            current_EDep = event_EDep[hit_idx]
            layer_hit = get_num_layers_traversed(current_x_pos)
            if(layer_hit == -1):
                skip_count += 1
                continue
            energies[layer_hit].append(current_EDep)
    print(f"skipped {skip_count}")
    for i in range(len(energies)):
        if(len(energies[i]) == 0):
            energy_means[i] = 0
            print(f"skipped layer #{i}")
        else:
            energy_means[i] = np.mean(energies[i])
    super_layer_means = np.zeros(14)
    for i in range(len(layer_EDep)):
        super_layer_num = int(np.floor(i / 2))
        super_layer_means[super_layer_num] += energy_means[i]
    fig,(ax1,ax2) = plot.subplots(1,2,figsize=(12,6))
    fig.suptitle(f"{energy}GeV {particle}-: energy deposited by layer avg over hits by {scope}")
    ax1.set_title("Super layer energy deposition")
    ax2.set_title("Individual layer energy deposition")

    ax1.set_xlabel("superlayer number")
    ax2.set_xlabel("layer number")

    ax1.set_ylabel("avg energy deposited per event (GeV)")
    ax1.scatter(range(14),super_layer_means,10,color = 'b',marker='o')
    ax2.scatter(layer_map,energy_means,10,color = 'r',marker='o')
    fig.show()
    fig.savefig(save_path = f"{particle}_{energy}GeV_{scope}_avg_hit.jpeg")
    return (energy_means,super_layer_means)

'''
fitting
'''



#process
def fit_to_angle(xdata, ydata, function):
    return curve_fit(function,xdata,ydata)

#plot
def plot_fit(xdata, ydata,function, popt):
    fig_theta, ax_theta = plot.subplots(1,1)
    ax_theta.plot(xdata, function(xdata,*popt), 'g--')
    fig_theta.suptitle("Theta dependence of % photons reaching sensor")
    ax_theta.scatter(xdata,ydata,color="red")
    ax_theta.set_xlabel("theta (deg)")
    ax_theta.set_ylabel("% photons reaching sensor")
    
'''
Machine Learning Stuff
'''

#Input root file path and get tensor with first index event and second index feature with the labels being index 56 (last)
def create_data(uproot_path):
    events = up.open(uproot_path)

    x_pos_branch = events["HcalBarrelHits/HcalBarrelHits.position.x"].array(library='np')
    EDep_branch = events["HcalBarrelHits.EDep"].array(library='np')
    Hits_MC_idx_branch = events["_HcalBarrelHits_MCParticle.index"].array(library='np')
    PDG_branch = events["MCParticles.PDG"].array(library='np')

    num_events = len(x_pos_branch)
    num_features = 56
    num_layers = 28

    hits_per_layer = np.zeros((num_events,28))
    EDep_per_layer = np.zeros((num_events,28))
    label = np.zeros((num_events,1))
    skip_count = 0
    for event_idx in range(len(x_pos_branch)):
        if(not event % (len(x_pos_branch / 100))):
            print(f"on event #{event_idx} for current file")
        event_x_pos = x_pos_branch[event_idx]
        event_EDep = EDep_branch[event_idx]
        #for each event, loop over the particles to find mu/pi
        for hit_idx in range(len(event_x_pos)):
            current_x_pos = event_x_pos[hit_idx]
            current_EDep = event_EDep[hit_idx]
            layer_hit = get_layer(current_x_pos,super_layer_map)
            if(layer_hit == -1):
                skip_count += 1
                continue
            EDep_per_layer[event_idx][layer_hit] += current_EDep
            hits_per_layer[event_idx][layer_hit] += 1
        label[event_idx][0] = part_dict[PDG_branch[event_idx][0]]
    return torch.cat((torch.tensor(hits_per_layer),torch.tensor(EDep_per_layer),torch.tensor(label)),1)
def create_data_depth(uproot_path, file_num = 0, particle = "pion"):
    events = up.open(uproot_path)

    x_pos_branch = events["HcalBarrelHits/HcalBarrelHits.position.x"].array(library='np')
    z_pos_branch = events["HcalBarrelHits/HcalBarrelHits.position.z"].array(library='np')
    EDep_branch = events["HcalBarrelHits.EDep"].array(library='np')
    Hits_MC_idx_branch = events["_HcalBarrelHits_MCParticle.index"].array(library='np')
    PDG_branch = events["MCParticles.PDG"].array(library='np')

    x_momentum_branch = events["MCParticles.momentum.x"].array(library='np')
    y_momentum_branch = events["MCParticles.momentum.y"].array(library='np')
    z_momentum_branch = events["MCParticles.momentum.z"].array(library='np')
    num_events = len(x_pos_branch)
    num_features = 56
    num_layers = 28

    hits_per_layer = np.zeros((num_events,28))
    EDep_per_layer = np.zeros((num_events,28))
    pixels_per_layer = np.zeros((num_events,28))
    label = np.zeros((num_events,1))
    layers_traversed = np.zeros((num_events,1))
    primary_momentum = np.zeros((num_events,1))
    primary_theta = np.zeros((num_events,1))
    primary_phi = np.zeros((num_events,1))
    skip_count = 0
    for event_idx in range(len(x_pos_branch)):
        if(not event_idx % (len(x_pos_branch) // 10)):
            clear_output(wait = True)
            print(f"on event #{event_idx} for file #{file_num} for {particle}")
        event_x_pos = x_pos_branch[event_idx]
        event_EDep = EDep_branch[event_idx]
        
        current_px = x_momentum_branch[event_idx][0]
        current_py = y_momentum_branch[event_idx][0]
        current_pz = z_momentum_branch[event_idx][0]
        current_theta = theta_func(current_px,current_py,current_pz)
        current_phi = phi_func(current_px,current_py,current_pz)

        primary_momentum[event_idx][0] = p_func(current_px,current_py,current_pz)
        primary_theta[event_idx][0] = theta_func(current_px,current_py,current_pz)
        primary_phi[event_idx][0] = phi_func(current_px,current_py,current_pz)
        #for each event, loop over the particles to find mu/pi
        hit_layers = np.zeros(28)
        for hit_idx in range(len(event_x_pos)):
            current_x_pos = event_x_pos[hit_idx]
            current_EDep = event_EDep[hit_idx]
            current_z_pos = z_pos_branch[event_idx][hit_idx]

            layer_hit = get_layer(current_x_pos,super_layer_map)
            if(layer_hit == -1):
                skip_count += 1
                continue
            hit_layers[layer_hit] += 1
            EDep_per_layer[event_idx][layer_hit] += current_EDep
            hits_per_layer[event_idx][layer_hit] += 1
            pixels_per_layer[event_idx][layer_hit] += calculate_num_pixels(current_EDep,current_z_pos)
        for i in range(29):
            if(i == 28):
                layers_traversed[event_idx][0] = 28
                break
            curr_pixels = calculate_num_pixels(EDep_per_layer[event_idx][i])
            if(curr_pixels < 2):
                layers_traversed[event_idx][0] = i
                break
        label[event_idx][0] = part_dict[PDG_branch[event_idx][0]]
    return torch.cat((torch.tensor(pixels_per_layer),torch.tensor(primary_momentum),torch.tensor(primary_theta),torch.tensor(primary_phi),torch.tensor(layers_traversed),torch.tensor(label)),1)

#Classifier network from NF project - works for doubles
class Classifier(nn.Module):
    """
    Classifier for normalized tensors
    """
    def __init__(self, input_size=28, num_classes=2, hidden_dim = 512, num_layers = 10):
        super(Classifier, self).__init__()
        self.layer = nn.Sequential()
        for i in range(num_layers):
            if(i == 0):
                self.layer.append(
                nn.Linear(input_size, hidden_dim)
                )
                self.layer.append(
                    nn.LeakyReLU(inplace=True)
                )
            elif(i == num_layers - 1):
                self.layer.append(
                nn.Linear(hidden_dim, num_classes)
                )
            else:
                self.layer.append(
                    nn.Linear(hidden_dim, hidden_dim)
                )
                self.layer.append(
                    nn.LeakyReLU(inplace=True)
                )
        self.name = "Classifier"
        self.double()
        
    def forward(self, h):
        c = self.layer(h)
        return c
    
    # @property
    def name(self):
        """
        Name of model.
        """
        return self.name
    
def train(classifier, train_data,optimizer, num_epochs = 18, batch_size = 100, show_progress = True):
    max_index = train_data.shape[1] - 1
    
    criterion = nn.CrossEntropyLoss()
    classifier.train()
    num_it = train_data.shape[0] // batch_size

    show_progress = True
    loss_hist = []
    curr_losses = []
    for i in range(num_epochs):
        clear_output(wait=True)
        print(f"Training epoch #{i}")
        epoch_hist = np.array([])
        val_epoch_hist = np.array([])
        with tqdm(total=num_it, position=0, leave=True) as pbar:
            for it in range(num_it):
                optimizer.zero_grad()
                #randomly sample the latent space
                begin = it * batch_size
                end = (it + 1) * batch_size
                it_data = train_data[begin:end]
                samples = it_data[:,:max_index]
                labels = it_data[:,max_index]#.unsqueeze(1)
    #             print(labels)
                samples = samples.to(device)
                labels = (labels.type(torch.LongTensor)).to(device)
                # forward + backward + optimize
                outputs = classifier(samples)
    #             print(outputs)
                loss = criterion(outputs, labels)
                # Do backprop and optimizer step
                if ~(torch.isnan(loss) | torch.isinf(loss)):
                    loss.backward()
                    optimizer.step()
                # Log loss
                if~(torch.isnan(loss)):
                    curr_losses.append(loss.to('cpu').data.numpy())
                    if(not (it % 5)):
                        loss_hist.append(sum(curr_losses) / len(curr_losses))
                        curr_losses = []
                if(show_progress):
                    pbar.update(1)
        

    print('Finished Training')
    return loss_hist
    
def test(classifier, test_data, test_batch_size = 10, show_progress = True, return_outputs = False):
    max_index = test_data.shape[1] - 1
    
    test_batch_size = 10
    test_num_it = test_data.shape[0] // test_batch_size
    classifier.eval()

    outputs = torch.empty(test_data.shape[0],2)    
    with tqdm(total=test_num_it, position=0, leave=True) as pbar:
        for it in range(test_num_it):
            #randomly sample the latent space
            begin = it * test_batch_size
            end = (it + 1) * test_batch_size
            it_data = test_data[begin:end]
            samples = it_data[:,:max_index]
            labels = it_data[:,max_index]#.unsqueeze(1)
            samples = samples.to(device)
            labels = (labels.type(torch.LongTensor)).to(device)
            # forward + backward + optimize
            output_batch = classifier(samples)
            for i in range(test_batch_size):
                outputs[it*test_batch_size + i] = output_batch[i]
            if(show_progress):
                pbar.update(1)
    test_Y     = test_data[:,max_index].clone().detach().float().view(-1, 1).to("cpu")
    probs_Y = torch.softmax(outputs, 1)
    argmax_Y = torch.max(probs_Y, 1)[1].view(-1,1)
    test_acc = (test_Y == argmax_Y.float()).sum().item() / len(test_Y)
    print(f"Accuracy: {test_acc * 100}")  
    if(return_outputs):
        return test_Y,probs_Y
    else:
        return
def plot_roc_curve(test_Y, probs_Y):
    verbose = True
    # Get ROC curve
    pfn_fp, pfn_tp, threshs = roc_curve(test_Y, probs_Y[:,1].detach().numpy())

    auc = roc_auc_score(np.squeeze(test_Y), probs_Y[:,1].detach().numpy())
    print(f'AUC = {auc:.4f}')
    f = plot.figure()

    plot.plot(pfn_tp, 1-pfn_fp, '-', color='black')

    # axes labels
    plot.xlabel('True positive efficiency')
    plot.ylabel('False positive Rejection')

    # axes limits
    plot.xlim(0, 1)
    plot.ylim(0, 1)

    f.show()
    
'''
mu, pi histograms
'''

'''
Preprocessing data for timing NF
'''
def z_func(z_vertex, theta):
    return z_vertex + 1 * math.tan(math.pi / 2 - theta)
# return time in ns for GeV/c, GeV/c^2 and mm inputs
c = 299792458 # 2.998 * 10 ^ 8 m/s
c_n = 1 #c = 1 in natural units
def time_func(p,m,dx):
    p_div_m = p / m
    vc = p_div_m * np.sqrt(1 / (1 + ((p_div_m) ** 2) * (1 / (c_n ** 2)))) # in terms of c
    v = vc * c #now in m/s
    v_mm = v * 1000 # in mm/s
    v_mmpns = v_mm / (10 ** (9)) # in mm/ns
    return dx / v_mmpns