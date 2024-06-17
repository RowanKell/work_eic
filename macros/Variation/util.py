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

def get_layer(x_pos,s_map = super_layer_map, s_dis = dis_between_super_layers):
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
    def setVector(self,px,py,pz,m):
        self.px = px
        self.py = py
        self.pz = pz
        self.M = m
        self.E = Efunc(px,py,pz,m)
        self.theta = theta_func(px,py,pz)
        self.phi = phi_func(px,py,pz)
        
#Currently used max and min angles        
theta_min = 67
theta_max = 113
phi_min = -22
phi_max = 22

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
def findBin(val,bins):
    diff = bins[1] - bins[0]
    rel_dist = val - bins[0]
    mod = np.floor(rel_dist / diff)
    return int(mod)

#Big function / loop to calculate the % of photons hitting and bin these values
#RETURNS (<list of phi percentage values>,<list of theta percent values>)
def bin_percent_theta_phi(EDep_branch,MC_px,MC_py,MC_pz)
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
def plot_percent(phi_bin_centers,theta_bin_centers,phi_percent,theta_percent, output_path = "plots/percentage/June_14_mu_5GeV_10k_phi_theta_20_bins.jpeg")
    fig, (ax1,ax2) = plot.subplots(1,2, figsize=(10,4))
    fig.suptitle("Percentage of photons reaching sensor as a fraction of total photons generated")
    ax1.scatter(phi_bin_centers,phi_percent,color="red")
    ax1.set_xlabel("phi (deg)")
    ax1.set_ylabel("percentage")
    ax2.scatter(theta_bin_centers,theta_percent,color = "blue")
    ax2.set_xlabel("theta (deg)")
    fig.show()
    fig.savefig(output_path)