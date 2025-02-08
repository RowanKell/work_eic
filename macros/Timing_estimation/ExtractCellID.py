import dd4hep
import ROOT
import ctypes
import matplotlib.pyplot as plot
import uproot
import numpy as np
import pandas as pd

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
    
# def findparentparticle(particle):
#     if(particle.getPDG() == 130 or particle.getPDG() == 13):
#         return particle
#     else:
#         for parent in particle.getParents():
#             return findparentparticle(parent)
    
'''
#Need to give any particle w/vertex inside solenoid its own trueID
'''
def find_parent_w_exclusion(PDG_branch,parent_idx_branch,parent_begin_branch,parent_end_branch,generatorStatus_branch,vx_branch,vy_branch,particle_instance_idx):
    if(parent_end_branch[particle_instance_idx] - parent_begin_branch[particle_instance_idx] > 1):
        print("hit particle has multiple parents... setting trueID to -1")
        return -1
    parent_begin_idx = parent_begin_branch[particle_instance_idx] #value to index into _MCP_parent.index
    parent_MC_instance = parent_idx_branch[parent_begin_idx]
    r_dist_from_z_axis = np.sqrt(vx_branch[parent_MC_instance] ** 2 + vy_branch[parent_MC_instance] ** 2)
    if(r_dist_from_z_axis < 1712.01): #base case - solenoid goes out to 1712mm
        return parent_MC_instance
    else: #recurrent case
        return find_parent_w_exclusion(PDG_branch,parent_idx_branch,parent_begin_branch,parent_end_branch,generatorStatus_branch,vx_branch,vy_branch,parent_MC_instance)
    
# def findparentparticle_w_exclusion(particle):
#     vertex = particle.getVertex()
#     r_dist_from_z_axs = np.sqrt(vertex[0],vertex[1])
#     if(r_dist_from_z_axis < 1712.01):
#         return particle
#     else:
#         for parent in particle.getParents():
#             return findparentparticle_w_exclusion(parent)
    
def process_root_file_old(file_path,max_events = -1,geometry_type = 1):
    print("began processing")
    #cellID decoding
    
    lcdd = load_geometry()
    world_volume = lcdd.worldVolume()
    root_file = load_root_file(file_path)
    tree = root_file.Get("events")
    z_hist = []
   
    with uproot.open(file_path) as file:
        tree_HcalBarrelHits = file["events/HcalBarrelHits"]
        tree_MCParticles = file["events/MCParticles"]
        
        
        momentum_x_MC = tree_MCParticles["MCParticles.momentum.x"].array(library="np")
        momentum_y_MC = tree_MCParticles["MCParticles.momentum.y"].array(library="np")
        momentum_z_MC = tree_MCParticles["MCParticles.momentum.z"].array(library="np")
        
        endpoint_x_MC = tree_MCParticles["MCParticles.endpoint.x"].array(library="np")
        endpoint_y_MC = tree_MCParticles["MCParticles.endpoint.y"].array(library="np")
        endpoint_z_MC = tree_MCParticles["MCParticles.endpoint.z"].array(library="np")
        
        vertex_x_MC = tree_MCParticles["MCParticles.vertex.x"].array(library="np")
        vertex_y_MC = tree_MCParticles["MCParticles.vertex.y"].array(library="np")
        
        pid_branch = tree_MCParticles["MCParticles.PDG"].array(library="np")
        generatorStatus_branch = tree_MCParticles["MCParticles.generatorStatus"].array(library="np")
        parent_begin_branch = tree_MCParticles["MCParticles.parents_begin"].array(library="np")
        parent_end_branch = tree_MCParticles["MCParticles.parents_end"].array(library="np")
        parent_idx_branch = file["events/_MCParticles_parents/_MCParticles_parents.index"].array(library="np")
        
        z_pos = tree_HcalBarrelHits["HcalBarrelHits.position.z"].array(library="np")
        y_pos = tree_HcalBarrelHits["HcalBarrelHits.position.y"].array(library="np")
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
                y = y_pos[event_idx][hit_idx]
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
#                     trueID = find_parent_w_exclusion(pid_branch[event_idx],parent_idx_branch[event_idx],parent_begin_branch[event_idx],parent_end_branch[event_idx],generatorStatus_branch[event_idx],vertex_x_MC[event_idx],vertex_y_MC[event_idx],particle_id)
                try:
                    KMU_trueID = find_parent(pid_branch[event_idx],parent_idx_branch[event_idx],parent_begin_branch[event_idx],parent_end_branch[event_idx],generatorStatus_branch[event_idx],particle_id)
                except:
                    print(f"event_idx: {event_idx}")
                truePID = pid_branch[event_idx][trueID]
                KMU_truePID = pid_branch[event_idx][KMU_trueID]
                true_momentum_mag = np.linalg.norm((momentum_x_MC[event_idx][trueID],
                            momentum_y_MC[event_idx][trueID],
                            momentum_z_MC[event_idx][trueID]))
                KMU_true_momentum_mag = np.linalg.norm((momentum_x_MC[event_idx][KMU_trueID],
                            momentum_y_MC[event_idx][KMU_trueID],
                            momentum_z_MC[event_idx][KMU_trueID]))
                true_theta = theta_func(momentum_x_MC[event_idx][trueID], momentum_y_MC[event_idx][trueID], momentum_z_MC[event_idx][trueID])
                true_phi = phi_func(momentum_x_MC[event_idx][trueID], momentum_y_MC[event_idx][trueID], momentum_z_MC[event_idx][trueID])
                KMU_true_phi = phi_func(momentum_x_MC[event_idx][KMU_trueID], momentum_y_MC[event_idx][KMU_trueID], momentum_z_MC[event_idx][KMU_trueID])
                
                KMU_true_endpointx = endpoint_x_MC[event_idx][KMU_trueID]
                KMU_true_endpointy = endpoint_y_MC[event_idx][KMU_trueID]
                KMU_true_endpointz = endpoint_z_MC[event_idx][KMU_trueID]
                
                #logic for recording strip position:
                #bar_pos = [x,y,z]
                bar_pos = find_volume(world_volume, bar_info,geometry_type)
                try:
                    strip_x = bar_pos[0]
                    strip_y = bar_pos[1]
                    strip_z = bar_pos[2]
                except TypeError:
                    print("skipping...")
                    continue
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
                        "edep" : e,
                        "strip_x" : strip_x,
                        "strip_y" : strip_y,
                        "strip_z" : strip_z,
                        "hit_x" : x,
                        "hit_y" : y,
                        "hit_z" : z,
                        "KMU_trueID" : KMU_trueID,
                        "KMU_truePID" : KMU_truePID,
                        "KMU_true_phi" : KMU_true_phi,
                        "KMU_true_momentum_mag" : KMU_true_momentum_mag,
                        "KMU_endpoint_x" : KMU_true_endpointx,
                        "KMU_endpoint_y" : KMU_true_endpointy,
                        "KMU_endpoint_z" : KMU_true_endpointz
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

def process_root_file_to_csv(file_path,max_events = -1,geometry_type = 1):
    print("began processing")
    #cellID decoding
    
    lcdd = load_geometry()
    world_volume = lcdd.worldVolume()
    root_file = load_root_file(file_path)
    tree = root_file.Get("events")
    z_hist = []
   
    with uproot.open(file_path) as file:
        tree_HcalBarrelHits = file["events/HcalBarrelHits"]
        tree_MCParticles = file["events/MCParticles"]
        
        
        momentum_x_MC = tree_MCParticles["MCParticles.momentum.x"].array(library="np")
        momentum_y_MC = tree_MCParticles["MCParticles.momentum.y"].array(library="np")
        momentum_z_MC = tree_MCParticles["MCParticles.momentum.z"].array(library="np")
        
        endpoint_x_MC = tree_MCParticles["MCParticles.endpoint.x"].array(library="np")
        endpoint_y_MC = tree_MCParticles["MCParticles.endpoint.y"].array(library="np")
        endpoint_z_MC = tree_MCParticles["MCParticles.endpoint.z"].array(library="np")
        
        vertex_x_MC = tree_MCParticles["MCParticles.vertex.x"].array(library="np")
        vertex_y_MC = tree_MCParticles["MCParticles.vertex.y"].array(library="np")
        
        pid_branch = tree_MCParticles["MCParticles.PDG"].array(library="np")
        generatorStatus_branch = tree_MCParticles["MCParticles.generatorStatus"].array(library="np")
        parent_begin_branch = tree_MCParticles["MCParticles.parents_begin"].array(library="np")
        parent_end_branch = tree_MCParticles["MCParticles.parents_end"].array(library="np")
        parent_idx_branch = file["events/_MCParticles_parents/_MCParticles_parents.index"].array(library="np")
        
        y_pos = tree_HcalBarrelHits["HcalBarrelHits.position.y"].array(library="np")
        z_pos = tree_HcalBarrelHits["HcalBarrelHits.position.z"].array(library="np")
        x_pos = tree_HcalBarrelHits["HcalBarrelHits.position.x"].array(library="np")
        energy = tree_HcalBarrelHits["HcalBarrelHits.EDep"].array(library="np")
        momentum_x = tree_HcalBarrelHits["HcalBarrelHits.momentum.x"].array(library="np")
        momentum_y = tree_HcalBarrelHits["HcalBarrelHits.momentum.y"].array(library="np")
        momentum_z = tree_HcalBarrelHits["HcalBarrelHits.momentum.z"].array(library="np")
        hit_time = tree_HcalBarrelHits["HcalBarrelHits.time"].array(library="np")
        mc_hit_idx = file["events/_HcalBarrelHits_MCParticle/_HcalBarrelHits_MCParticle.index"].array(library="np")  # Add PDG code for particle identification
        print("finished loading branches")
        
        data = []
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
                y = y_pos[event_idx][hit_idx]
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
#                     trueID = find_parent_w_exclusion(pid_branch[event_idx],parent_idx_branch[event_idx],parent_begin_branch[event_idx],parent_end_branch[event_idx],generatorStatus_branch[event_idx],vertex_x_MC[event_idx],vertex_y_MC[event_idx],particle_id)
                    trueID = find_parent(pid_branch[event_idx],parent_idx_branch[event_idx],parent_begin_branch[event_idx],parent_end_branch[event_idx],generatorStatus_branch[event_idx],particle_id)
                truePID = pid_branch[event_idx][trueID]
                true_momentum_mag = np.linalg.norm((momentum_x_MC[event_idx][trueID],
                            momentum_y_MC[event_idx][trueID],
                            momentum_z_MC[event_idx][trueID]))
                true_theta = theta_func(momentum_x_MC[event_idx][trueID], momentum_y_MC[event_idx][trueID], momentum_z_MC[event_idx][trueID])
                true_phi = phi_func(momentum_x_MC[event_idx][trueID], momentum_y_MC[event_idx][trueID], momentum_z_MC[event_idx][trueID])
                
                KMU_trueID = find_parent(pid_branch[event_idx],parent_idx_branch[event_idx],parent_begin_branch[event_idx],parent_end_branch[event_idx],generatorStatus_branch[event_idx],particle_id)
                KMU_truePID = pid_branch[event_idx][KMU_trueID]
                
                KMU_true_momentum_mag = np.linalg.norm((momentum_x_MC[event_idx][KMU_trueID],
                            momentum_y_MC[event_idx][KMU_trueID],
                            momentum_z_MC[event_idx][KMU_trueID]))
                
                KMU_true_phi = phi_func(momentum_x_MC[event_idx][KMU_trueID], momentum_y_MC[event_idx][KMU_trueID], momentum_z_MC[event_idx][KMU_trueID])
                
                KMU_true_endpointx = endpoint_x_MC[event_idx][KMU_trueID]
                KMU_true_endpointy = endpoint_y_MC[event_idx][KMU_trueID]
                KMU_true_endpointz = endpoint_z_MC[event_idx][KMU_trueID]
                
                #logic for recording strip position:
                #bar_pos = [x,y,z]
                bar_pos = find_volume(world_volume, bar_info,geometry_type)
                try:
                    strip_x = bar_pos[0]
                    strip_y = bar_pos[1]
                    strip_z = bar_pos[2]
                except TypeError:
                    print("skipping...")
                    continue
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
                        "edep" : e,
                        "strip_x" : strip_x,
                        "strip_y" : strip_y,
                        "strip_z" : strip_z,
                        "hit_x" : x,
                        "hit_y" : y,
                        "hit_z" : z,
                        "KMU_trueID" : KMU_trueID,
                        "KMU_truePID" : KMU_truePID,
                        "KMU_true_phi" : KMU_true_phi,
                        "KMU_true_momentum_mag" : KMU_true_momentum_mag,
                        "KMU_endpoint_x" : KMU_true_endpointx,
                        "KMU_endpoint_y" : KMU_true_endpointy,
                        "KMU_endpoint_z" : KMU_true_endpointz
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
                            
                            row = {
                                "event_idx": event_idx,
                                "stave_idx": stave_idx,
                                "layer_idx": layer_idx,
                                "segment_idx": segment_idx,
                                "particle_id": particle_id
                            }
                            row.update(hit_data)
                            data.append(row)
            if(max_events > 0 and event_idx > max_events):
                break
    print("finished processing")
    return pd.DataFrame(data)

def load_geometry():
    lcdd = dd4hep.Detector.getInstance()
    eic_pref = "/hpc/group/vossenlab/rck32/eic/"
    lcdd.fromXML(eic_pref + "epic_klm/epic_klmws_only.xml")
    return lcdd

def load_root_file(fileName = "/hpc/group/vossenlab/rck32/eic/work_eic/root_files/momentum_prediction/November_06/hepmc_1000events_test_file_0.edm4hep.root"):
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
#Geometry type:
#    2 if 2 bars per superlayer
#    1 if 1 bar per superlayer (updated design)
def find_volume(world_volume, hit_info,geometry_type = 1):
    target_stave = hit_info['stave']
    target_layer = hit_info['layer'] - 1
    total_slice = hit_info['slice']
    match geometry_type:
        case 1:
            num_slices_per_layer = 4
        case 2:
            num_slice_per_layer = 7
        case _:
            rum_slice_per_layer = 4
    target_segment = total_slice // num_slices_per_layer
    target_slice = (total_slice % num_slices_per_layer) + 1

    # Get HcalBarrelVolume
    HcalBarrelVolume = world_volume.GetNodes()[0].GetVolume()  # Assuming HcalBarrelVolume is the fourth child

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
#     print(f"IDs: (stave, layer, total_slice, target_segment, target_slice) ({target_stave},{target_layer},{total_slice},{target_segment},{target_slice})")
#     print(f"Found slice in (stave_name,layer_name): ({stave_name},{layer_name})")
#     print(f"Found slice name: {slice_name}")
    if not slice_node:
        print(f"Slice {slice_name} not found")
        return None
    position_in_stave = get_position(slice_node) + get_position(layer)
    x = position_in_stave[0]
    y = position_in_stave[1]
    z = position_in_stave[2] + 52.5
    z_translated = z + (1420 + 350) / 10 #add in displacement from origin
    angle = np.arctan2(x,z_translated)
    r = np.sqrt(z_translated ** 2 + x ** 2)
#     print(f"target_stave: {target_stave}")
    angle_prime = ((-target_stave - 1) * np.pi / 4) + angle
#     print(f"angle, angle_prime: {angle},{angle_prime}")
    x_prime = r * np.sin(angle_prime)# rotating x and z
#     print(f"x, x_prime: {x},{x_prime}")
    z_prime = r * np.cos(angle_prime)
    return np.array([x_prime.item(),y,z_prime.item()])

# Helper function to get position (unchanged)
def get_position(node):
    transformation = node.GetMatrix()
    x = transformation.GetTranslation()[0]
    y = transformation.GetTranslation()[1]
    z = transformation.GetTranslation()[2]
    return np.array([x, y, z])
'''
# root_file_path = "/hpc/group/vossenlab/rck32/eic/work_eic/root_files/momentum_prediction/November_06/hepmc_1000events_test_file_0.edm4hep.root"#file I sent to Greg
root_file_path = "/hpc/group/vossenlab/rck32/eic/work_eic/root_files/momentum_prediction/November_08/hepmc_100events_test_file_0.edm4hep.root"
# Main loop (simplified)
lcdd = load_geometry()
file = load_root_file(root_file_path)
tree = file.Get("events")
world_volume = lcdd.worldVolume()

up_file = uproot.open(root_file_path)
tree_HcalBarrelHits = up_file["events/HcalBarrelHits"]
tree_MCParticles = up_file["events/MCParticles"]

mc_hit_idx = up_file["events/_HcalBarrelHits_MCParticle/_HcalBarrelHits_MCParticle.index"].array(library="np")
pid_branch = tree_MCParticles["MCParticles.PDG"].array(library="np")
px_branch = tree_MCParticles["MCParticles.momentum.x"].array(library="np")
py_branch = tree_MCParticles["MCParticles.momentum.y"].array(library="np")
pz_branch = tree_MCParticles["MCParticles.momentum.z"].array(library="np")
        
vertex_x_MC = tree_MCParticles["MCParticles.vertex.x"].array(library="np")
vertex_y_MC = tree_MCParticles["MCParticles.vertex.y"].array(library="np")

generatorStatus_branch = tree_MCParticles["MCParticles.generatorStatus"].array(library="np")
parent_begin_branch = tree_MCParticles["MCParticles.parents_begin"].array(library="np")
parent_end_branch = tree_MCParticles["MCParticles.parents_end"].array(library="np")
parent_idx_branch = up_file["events/_MCParticles_parents/_MCParticles_parents.index"].array(library="np")

fig2,axs2 = plot.subplots(5,5,figsize = (15,15))

# colors = ["red","blue","green","orange","black","purple","yellow","pink"]
colors = [
    '#1f77b4',  # blue
    '#d62728',  # red
    '#2ca02c',  # green
    '#9467bd',  # purple
    '#ff7f0e',  # orange
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # yellow-green
    '#17becf',  # cyan
    '#393b79',  # dark blue
    '#b82d2d',  # dark red
    '#316931',  # dark green
    '#9e4596',  # dark purple
    '#a05d00',  # dark orange
    '#ad494a',  # dark pink
    '#5254a3',  # medium blue
    '#637939',  # olive
    '#8c6d31',  # gold
    '#843c39',  # dark red-brown
    '#7b4173',  # medium purple
    '#4a784c',  # forest green
    '#e7969c',  # light red
    '#7b4173'   # plum
]

num_plots = 0
for event_idx, event in enumerate(tree):
    x, y, z = [], [], []
    hit_x,hit_y,hit_z = [], [], []
#     parent_x,parent_y,parent_z = [],[],[]
    has_hits = False
    trueID_list = []
    trueID_theta_list = []
    color_dict = {}
    for hit_idx, hit in enumerate(event.HcalBarrelHits):
        hit_x.append(hit.position.x / 10)
        hit_y.append(hit.position.y / 10)
        hit_z.append(hit.position.z / 10)
        
        #find parent
        particle_id = mc_hit_idx[event_idx][hit_idx]

        hitPID = pid_branch[event_idx][particle_id]

        #find trueID
        trueID = -1 #need trueID to be ID of final state SIDIS particle
        if(generatorStatus_branch[event_idx][particle_id] == 1): 
            trueID = particle_id
        else:
            trueID = find_parent_w_exclusion(pid_branch[event_idx],parent_idx_branch[event_idx],parent_begin_branch[event_idx],parent_end_branch[event_idx],generatorStatus_branch[event_idx],vertex_x_MC[event_idx],vertex_y_MC[event_idx],particle_id)
#             trueID = find_parent(pid_branch[event_idx],parent_idx_branch[event_idx],parent_begin_branch[event_idx],parent_end_branch[event_idx],generatorStatus_branch[event_idx],particle_id)
#         parent_x.append()
        trueID_theta = phi_func(px_branch[event_idx][trueID],py_branch[event_idx][trueID],pz_branch[event_idx][trueID])
        if(trueID not in color_dict.keys()):
            color_dict[trueID] = colors[len(color_dict.keys())]
            trueID_theta_list.append(trueID_theta)
        
        bar_info = get_bar_info(lcdd,hit)
        result= find_volume(world_volume, bar_info)
        try:
            x.append(result[0])
            y.append(result[1])
            z.append(result[2])
            trueID_list.append(trueID)
            has_hits = True
#             print(f"Event #{event_idx}, trueID {trueID}, color {color_dict[trueID]}")
        except TypeError:
            print("skipping...")
    if(has_hits):
        row = num_plots // 5
        column = num_plots % 5
#         axs2[row,column].scatter(np.array(z),x, alpha = 0.5) #this plots the segment position - lines up with hits
        axs2[row,column].scatter(hit_x,hit_y, alpha = 0.5,color = [color_dict[trueID] for trueID in trueID_list])
        axs2[row,column].set_title(f"Event #{event_idx}")
#         for trueID_idx in range(len(trueID_list)):
#             arrow_y = []
#             for i in range(100):
#                 arrow_y.append(i * np.tan(trueID_theta_list[trueID_idx]))
#             axs2[row,column].plot(range(100),arrow_y)
        axs2[row,column].set_xlim(-360,360)
        axs2[row,column].set_ylim(-360,360)
        num_plots += 1
    if(num_plots == 25):
        break


R = (1420 + 350) / 10
color = "black"
outer_radius = R / np.cos(np.pi/8)
angles = np.linspace(-np.pi/8, 2*np.pi-np.pi/8, 9)[:-1] 
hex_x = outer_radius * np.cos(angles)
hex_y = outer_radius * np.sin(angles)
for i in range(5):
    for j in range(5):
        axs2[i,j].plot(np.append(hex_x,hex_x[0]), np.append(hex_y, hex_y[0]), color=color, linewidth=0.4)
#         circle = plot.Circle((0, 0), R, fill=False, linestyle='--', color='gray')
#         axs2[i,j].add_artist(circle)
        axs2[i,j].set_aspect('equal')
        axs2[i,j].set_xlim(-360,360)
        axs2[i,j].set_ylim(-360,360)
fig2.tight_layout()
fig2.savefig("test/scatter_inner_detectors_new_parent_func.pdf")
'''
        
# # Plot histogram (unchanged)
# fig, axs = plot.subplots(1,1,figsize = (5,5))
# # axs.hist2d(z,x,bins = 80)
# axs.scatter(z,x,np.ones(len(z)) * 0.3)
# fig.tight_layout()
# fig.savefig("test/CellID_scatter_plot.pdf")


