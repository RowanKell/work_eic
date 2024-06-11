#imports
import numpy as np
import uproot as up
import pandas as pd
import matplotlib.pyplot as plot

#USER SET VARIABLES

#Set these according to the simulation
select_pdg = 13 #pdg value of particle gun (mu- = 13; pi- = 211)

uproot_path = f"~/eic/work_eic/root_files/June_10_tracker/pi_1GeV_10k_full_layer.edm4hep.root:events"
#uproot_path = f"~/eic/work_eic/root_files/June_10_tracker/mu_1GeV_10k.edm4hep.root:events"
#uproot_path = f"~/eic/work_eic/root_files/June_10_tracker/pi_1GeV_10k.edm4hep.root:events"
events = up.open(uproot_path)

#x_pos_branch = events["HcalBarrelHits/HcalBarrelHits.position.x"].array(library='np')
x_pos_branch = events["MCParticles.endpoint.x"].array(library='np')
PDG_branch = events["MCParticles.PDG"].array(library='np')
def get_num_layers_traversed(x_pos):
    #This map contains the midpoints of each layer - each layer is 1cm in width, so we can check the 0.5cm on each side to see if a hit was in the layer
    layer_map = [1830.8000, 1841.4000, 1907.5, 1918.1,1984.1999, 1994.8000, 2060.8999,2071.5,2137.6001,2148.1999,2214.3000,2224.8999,2291,2301.6001,2367.6999,2378.3000,2444.3999,2455,2521.1001,2531.6999,2597.8000,2608.3999,2674.5,2685.1001,2751.1999,2761.8000,2827.8999,2838.5]
    for layer_idx in range(len(layer_map)):
        #check if particle hit within layer
        if ((x_pos >= layer_map[layer_idx] - 5) and (x_pos <= layer_map[layer_idx] + 5)):
            return layer_idx
    #if no layers hit, send error code
    return -1

#initialize list that we will fill with # of layers traversed for each event
layers_traversed = []


#loop over each event

for event_idx in range(len(x_pos_branch)):
    event_x_pos = x_pos_branch[event_idx]
    event_layer_hits = []
    
    #for each event, loop over the particles to find mu/pi
    for hit_idx in range(len(event_x_pos)):
        if(PDG_branch[event_idx][hit_idx] != -211):continue
        current_x_pos = event_x_pos[hit_idx]
        layer_hit = get_num_layers_traversed(current_x_pos)
        if(layer_hit == -1):
            print(f"No hits within layer volumes for #{event_idx}... skipping")
            continue
        event_layer_hits.append(layer_hit)#add to list
    if(len(event_layer_hits) != 0):
        super_layer_hit = np.floor(max(event_layer_hits) / 2)
        layers_traversed.append(super_layer_hit)
plot.hist(layers_traversed,bins=100)
plot.savefig("./pi_1GeV_10k_full_layer_MC.jpeg")
