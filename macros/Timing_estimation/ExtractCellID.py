import dd4hep
import ROOT
import ctypes
import matplotlib.pyplot as plot
import uproot
import numpy as np

from collections import defaultdict

def theta_func(px,py,pz):
    return np.arccos(pz / np.sqrt(px ** 2 + py ** 2 + pz ** 2)) * 180 / np.pi
def phi_func(px,py,pz):
    return np.arctan2(py,px) * 180 / np.pi
def inverse(x,a,b,c):
    return a  / (x + b) + c
def calculate_num_pixels_z_dependence(energy_dep, z_hit):
    efficiency = inverse(770 - z_hit,494.98,9.9733,-0.16796) * 0.5 #0.5 for QE of 50% (approx from simulation)
    return 10 * energy_dep * (1000 * 1000) * efficiency / 100

def find_parent(PDG_branch,parent_idx_branch,parent_begin_branch,parent_end_branch,generatorStatus_branch,particle_instance_idx):
    if(parent_end_branch[particle_instance_idx] - parent_begin_branch[particle_instance_idx] > 1):
        print("hit particle has multiple parents... setting trueID to -1")
        return -1
    parent_begin_idx = parent_begin_branch[particle_instance_idx] #value to index into _MCP_parent.index
    parent_MC_instance = parent_idx_branch[parent_begin_idx]
    if(generatorStatus_branch[parent_MC_instance] == 1): #base case
        return parent_MC_instance
    else: #recurrent case
        return find_parent(PDG_branch,parent_idx_branch,parent_begin_branch,parent_end_branch,generatorStatus_branch,parent_MC_instance)

def process_root_file(file_path,max_events = -1):
    print("began processing")
    #cellID decoding
    
    lcdd = load_geometry()
    root_file = load_root_file(file_path)
    tree = root_file.Get("events")
    z_hist = []
   
    with uproot.open(file_path) as file:
        tree_HcalBarrelHits = file["events/HcalBarrelHits"]
        tree_MCParticles = file["events/MCParticles"]
        
        
        momentum_x_MC = tree_MCParticles["MCParticles.momentum.x"].array(library="np")
        momentum_y_MC = tree_MCParticles["MCParticles.momentum.y"].array(library="np")
        momentum_z_MC = tree_MCParticles["MCParticles.momentum.z"].array(library="np")
        
        pid_branch = tree_MCParticles["MCParticles.PDG"].array(library="np")
        generatorStatus_branch = tree_MCParticles["MCParticles.generatorStatus"].array(library="np")
        parent_begin_branch = tree_MCParticles["MCParticles.parents_begin"].array(library="np")
        parent_end_branch = tree_MCParticles["MCParticles.parents_end"].array(library="np")
        parent_idx_branch = file["events/_MCParticles_parents/_MCParticles_parents.index"].array(library="np")
        
        z_pos = tree_HcalBarrelHits["HcalBarrelHits.position.z"].array(library="np")
        x_pos = tree_HcalBarrelHits["HcalBarrelHits.position.x"].array(library="np")
        energy = tree_HcalBarrelHits["HcalBarrelHits.EDep"].array(library="np")
        momentum_x = tree_HcalBarrelHits["HcalBarrelHits.momentum.x"].array(library="np")
        momentum_y = tree_HcalBarrelHits["HcalBarrelHits.momentum.y"].array(library="np")
        momentum_z = tree_HcalBarrelHits["HcalBarrelHits.momentum.z"].array(library="np")
        hit_time = tree_HcalBarrelHits["HcalBarrelHits.time"].array(library="np")
        mc_hit_idx = file["events/_HcalBarrelHits_MCParticle/_HcalBarrelHits_MCParticle.index"].array(library="np")  # Add PDG code for particle identification
        print("finished loading branches")
        
        processed_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
        for event_idx, event in enumerate(tree):
            if(len(z_pos[event_idx]) == 0):
                continue
            primary_momentum = (momentum_x_MC[event_idx][0],
                            momentum_y_MC[event_idx][0],
                            momentum_z_MC[event_idx][0])
            primary_momentum_mag = np.linalg.norm(primary_momentum)
            if(primary_momentum_mag <= 0):
                continue
            if(primary_momentum_mag > 100):
                continue
            first_hit_per_layer_particle = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
            # First pass: collect first hit data and calculate energy per layer per particle
            for hit_idx,hit in enumerate(event.HcalBarrelHits):
                
                bar_info = get_bar_info(lcdd,hit)
                stave_idx = bar_info['stave']
                layer_idx = bar_info["layer"] - 1 #gives 0 indexed layer
                slice_absolute_idx = bar_info["slice"]
                segment_idx = slice_absolute_idx // 7
                slice_idx = (slice_absolute_idx % 7) + 1 #no need for 0 indexing
                
                z = z_pos[event_idx][hit_idx]
                x = x_pos[event_idx][hit_idx]
                e = energy[event_idx][hit_idx]
                momentum = (momentum_x[event_idx][hit_idx],
                            momentum_y[event_idx][hit_idx],
                            momentum_z[event_idx][hit_idx])
                momentum_mag = np.linalg.norm(momentum)
                hittheta = theta_func(momentum_x[event_idx][hit_idx], momentum_y[event_idx][hit_idx], momentum_z[event_idx][hit_idx])
                phi = phi_func(momentum_x[event_idx][hit_idx], momentum_y[event_idx][hit_idx], momentum_z[event_idx][hit_idx])
                particle_id = mc_hit_idx[event_idx][hit_idx]
                
                hitPID = pid_branch[event_idx][particle_id]
                
                #find trueID
                trueID = -1 #need trueID to be ID of final state SIDIS particle
                if(generatorStatus_branch[event_idx][particle_id] == 1): 
                    trueID = particle_id
                else:
                    trueID = find_parent(pid_branch[event_idx],parent_idx_branch[event_idx],parent_begin_branch[event_idx],parent_end_branch[event_idx],generatorStatus_branch[event_idx],particle_id)
                truePID = pid_branch[event_idx][trueID]
                true_momentum_mag = np.linalg.norm((momentum_x_MC[event_idx][trueID],
                            momentum_y_MC[event_idx][trueID],
                            momentum_z_MC[event_idx][trueID]))
                true_theta = theta_func(momentum_x[event_idx][trueID], momentum_y[event_idx][trueID], momentum_z[event_idx][trueID])
                true_phi = phi_func(momentum_x[event_idx][trueID], momentum_y[event_idx][trueID], momentum_z[event_idx][trueID])
                
                z_hist.append(z)
                if stave_idx not in first_hit_per_layer_particle or layer_idx not in first_hit_per_layer_particle[stave_idx] or segment_idx not in first_hit_per_layer_particle[stave_idx][layer_idx] or particle_id not in first_hit_per_layer_particle[stave_idx][layer_idx][segment_idx]:
                    first_hit_per_layer_particle[stave_idx][layer_idx][segment_idx][particle_id] = {
                        "z_pos": z,
                        "x_pos": x,
                        "hitmomentum": momentum_mag,
                        "truemomentum": true_momentum_mag,
                        "truetheta": true_theta,
                        "hittheta": hittheta,
                        "time": hit_time[event_idx][hit_idx],
                        "trueID": trueID,
                        "truePID": truePID,
                        "hitID": particle_id,
                        "hitPID": hitPID,
                        "truephi": true_phi,
                        "edep" : e
                    }
                else:
                    first_hit_per_layer_particle[stave_idx][layer_idx][segment_idx][particle_id]["edep"] += e
            
            
            # Second pass: process first hit with total layer energy per particle
            for stave_idx, stave_data in first_hit_per_layer_particle.items():
                for layer_idx, particle_data in stave_data.items():
                    for segment_idx, segement_data in particle_data.items():
                        for particle_id, hit_data in segement_data.items():
                            segment_particle_energy = hit_data["edep"]
                            num_pixels_high_z = calculate_num_pixels_z_dependence(segment_particle_energy, hit_data["z_pos"])
                            num_pixels_low_z = calculate_num_pixels_z_dependence(segment_particle_energy, -1 * hit_data["z_pos"])
                            hit_data["num_pixels_high_z"] = int(np.floor(num_pixels_high_z))
                            hit_data["num_pixels_low_z"] = int(np.floor(num_pixels_low_z))
                            curr_z_pos = hit_data["z_pos"]
                            
                            hit_data["segment_energy"] = segment_particle_energy  # Store total layer energy for this particle
                            processed_data[event_idx][stave_idx][layer_idx][segment_idx][particle_id] = hit_data
            if(max_events > 0 and event_idx > max_events):
                break
    print("finished processing")
    return processed_data

def load_geometry():
    lcdd = dd4hep.Detector.getInstance()
    eic_pref = "/hpc/group/vossenlab/rck32/eic/"
    lcdd.fromXML(eic_pref + "epic_klm/epic_klmws_only.xml")
    return lcdd

def load_root_file(fileName = "/hpc/group/vossenlab/rck32/eic/full_sector_50.edm4hep.root"):
    f = ROOT.TFile(fileName)
    return f

def get_bar_info(lcdd, hit):
    cellID = hit.cellID
#     print(f"CellID: {cellID}")
    
    # Get the IDDescriptor for the HcalBarrel
    id_spec = lcdd.idSpecification("HcalBarrelHits")
    if not id_spec:
        print("Failed to get IDSpecification for HcalBarrelHits")
        return None
    
    id_dec = id_spec.decoder()
    
    # Extract individual field values
    try:
        system = id_dec.get(cellID, "system")
        barrel = id_dec.get(cellID, "barrel")
        module = id_dec.get(cellID, "module")
        layer = id_dec.get(cellID, "layer")
        slice_id = id_dec.get(cellID, "slice")
    except Exception as e:
        print(f"Error decoding cellID: {e}")
        return None
    
    return {
        "system": system,
        "barrel": barrel,
        "stave": module - 1,
        "layer": layer,
        "slice": slice_id
    }

def find_volume(world_volume, hit_info):
    target_stave = hit_info['stave']
    target_layer = hit_info['layer'] - 1
    total_slice = hit_info['slice']
    target_segment = total_slice // 7
    target_slice = (total_slice % 7) + 1

    # Get HcalBarrelVolume
    HcalBarrelVolume = world_volume.GetNodes()[0].GetVolume()  # Assuming HcalBarrelVolume is the first child of world_volume

    # Access stave directly
    stave_name = f"stave_{target_stave}"
    stave = HcalBarrelVolume.FindNode(stave_name)
    if not stave:
        print(f"Stave {stave_name} not found")
        return None

    # Access layer directly
    layer_name = f"layer{target_layer + 1}_{target_layer}" #layer1_0
    layer = stave.GetVolume().FindNode(layer_name)
    if not layer:
        print(f"Layer {layer_name} not found")
        return None
    # Access slice directly
    slice_name = f"seg{target_segment}slice{target_slice}_{total_slice}"
    slice_node = layer.GetVolume().FindNode(slice_name)
    print(f"IDs: (stave, layer, total_slice, target_segment, target_slice) ({target_stave},{target_layer},{total_slice},{target_segment},{target_slice})")
    print(f"Found slice in (stave_name,layer_name): ({stave_name},{layer_name})")
    print(f"Found slice name: {slice_name}")
    if not slice_node:
        print(f"Slice {slice_name} not found")
        return None

    return get_position(slice_node)

# Helper function to get position (unchanged)
def get_position(node):
    transformation = node.GetMatrix()
    x = transformation.GetTranslation()[0]
    y = transformation.GetTranslation()[1]
    z = transformation.GetTranslation()[2]
    return [x, y, z]

# # # Main loop (simplified)
# lcdd = load_geometry()
# file = load_root_file()
# tree = file.Get("events")
# world_volume = lcdd.worldVolume()
# x, y, z = [], [], []

# for event_idx, event in enumerate(tree):
#     for hit_idx, hit in enumerate(event.HcalBarrelHits):
#         bar_info = get_bar_info(hit)
# #         print(bar_info)
#         if bar_info:
#             result = find_volume(world_volume, bar_info)
#             if result:
#                 x.append(result[0])
#                 y.append(result[1])
#                 z.append(result[2])
#                 break
#             else:
#                 print("Skipped one event")

# # Plot histogram (unchanged)
# plot.hist(x, bins=500)
# plot.savefig("test/CellIDplot.jpeg")