import numpy as np
import os
import torch
import uproot as up
from IPython.display import clear_output
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

def theta_func(px,py,pz):
    return np.arccos(pz / np.sqrt(px ** 2 + py ** 2 + pz ** 2)) * 180 / np.pi
def phi_func(px,py,pz):
    return np.arctan2(py,px) * 180 / np.pi
def Efunc(px,py,pz,m):
    return np.sqrt(px**2 + py**2 + pz**2 + m**2)

def r_func(x,y,z):
    return np.sqrt(x**2 + y**2 + z**2)
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
def inverse(x,a,b,c):
    return a  / (x + b) + c

def checkdir(path):
    if not os.path.exists(path): 
        os.makedirs(path)
        
def z_func(z_vertex, theta):
    return z_vertex + 6 * np.tan(np.pi / 2 - theta * np.pi / 180)
# return time in ns for GeV/c, GeV/c^2 and mm 
def avg_time(num,event_times):
    ten_sorted_times = sorted(event_times)[:num]
    return sum(ten_sorted_times) / len(ten_sorted_times)

def time_func(p,m,dx):
    c = 299792458 # 2.998 * 10 ^ 8 m/s
    c_n = 1 #c = 1 in natural units
    p_div_m = p / m
    vc = p_div_m * np.sqrt(1 / (1 + ((p_div_m) ** 2) * (1 / (c_n ** 2)))) # in terms of c
    v = vc * c #now in m/s
    v_mm = v * 1000 # in mm/s
    v_mmpns = v_mm / (10 ** (9)) # in mm/ns
    return dx / v_mmpns

def calculate_num_pixels_z_dependence(energy_dep, z_hit):
    efficiency = inverse(770 - z_hit,494.98,9.9733,-0.16796) * 0.5 #0.5 for QE of 50% (approx from simulation)
    return 10 * energy_dep * (1000 * 1000) * efficiency / 100

def create_unique_mapping(arr):
    # Get unique values and their inverse mapping
    unique_values, inverse_indices = np.unique(arr, return_inverse=True)
    
    # Create a dictionary mapping unique values to their indices
    value_to_index = {val: idx for idx, val in enumerate(unique_values)}
    
    # Create an array of indices
    index_array = inverse_indices
    
    return len(unique_values), value_to_index

def prepare_data_for_nn_one_segment(processed_data):
    all_features = []
    all_metadata = []
#     print(f"len of events: {len(processed_data)}")
    for event_idx, event_data in enumerate(processed_data):
        features = event_data[:4]  # Get first 4 features
        repeat_count = int(event_data[4])  # Get 5th feature as repeat count

        #cuts
        if(features[3] > 50):
            continue


        if not np.any(features == -1) and repeat_count > 0:  # Check if all features are -1 and repeat_count is valid
            # Repeat the features and metadata by repeat_count
            all_features.extend([features] * repeat_count)
            all_metadata.extend([(event_idx)] * repeat_count)
    
    # Convert to numpy arrays
    features_array = np.array(all_features)
    metadata_array = np.array(all_metadata)
    
    return features_array, metadata_array
def create_dataloader(features, metadata, batch_size=32,shuffle_bool=True):
    # Convert to PyTorch tensors
    features_tensor = torch.tensor(features, dtype=torch.float32)
    metadata_tensor = torch.tensor(metadata, dtype=torch.long)
    print(features.shape)
    # Create TensorDataset
    dataset = TensorDataset(features_tensor, metadata_tensor)
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_bool)
    
    return dataloader

def prepare_data_for_nn(processed_data):
    all_features = []
    all_metadata = []
    print(f"len of events: {len(processed_data)}")
    for event_idx, event_data in enumerate(processed_data):
        for particle_idx in range(event_data.shape[0]):
            for layer_idx in range(event_data.shape[1]):
                features = event_data[particle_idx, layer_idx, :4]  # Get first 4 features
                repeat_count = int(event_data[particle_idx, layer_idx, 4])  # Get 5th feature as repeat count
                
                #cuts
                if(features[3] > 50):
                    continue
                
                
                if not np.any(features == -1) and repeat_count > 0:  # Check if all features are -1 and repeat_count is valid
                    # Repeat the features and metadata by repeat_count
                    all_features.extend([features] * repeat_count)
                    all_metadata.extend([(event_idx, particle_idx, layer_idx)] * repeat_count)
    
    # Convert to numpy arrays
    features_array = np.array(all_features)
    metadata_array = np.array(all_metadata)
    
    return features_array, metadata_array
def process_data_one_segment(uproot_path, file_num=0, particle="pion"):
    events = up.open(uproot_path)
    
    x_pos_branch = events["HcalBarrelHits/HcalBarrelHits.position.x"].array(library='np')
    z_pos_branch = events["HcalBarrelHits/HcalBarrelHits.position.z"].array(library='np')
    EDep_branch = events["HcalBarrelHits.EDep"].array(library='np')
    PDG_branch = events["MCParticles.PDG"].array(library='np')
    x_momentum_branch = events["HcalBarrelHits/HcalBarrelHits.momentum.x"].array(library='np')
    y_momentum_branch = events["HcalBarrelHits/HcalBarrelHits.momentum.y"].array(library='np')
    z_momentum_branch = events["HcalBarrelHits/HcalBarrelHits.momentum.z"].array(library='np')
    Hits_MC_idx_branch = events["_HcalBarrelHits_MCParticle.index"].array(library='np')
    time_branch = events["HcalBarrelHits.time"].array(library='np')   
    
    num_events = len(x_pos_branch)
    num_features = 5
    
    data = np.ones((num_events,num_features),dtype=float)
    
    for event_idx in range(num_events):
        Hits_MC_idx_event = Hits_MC_idx_branch[event_idx]
        PDG_event = PDG_branch[event_idx]
        
        p = -999
        z_hit = -999
        theta = -999
        hit_time = -999
        edep_event = -999
        PDG_list = -999
        
        x_pos_event = x_pos_branch[event_idx]
        px_event = x_momentum_branch[event_idx]
        py_event = y_momentum_branch[event_idx]
        pz_event = z_momentum_branch[event_idx]
        z_event = z_pos_branch[event_idx]
        time_event = time_branch[event_idx]
        EDep_event = EDep_branch[event_idx]
        for hit_idx in range(len(x_pos_event)):
            idx = Hits_MC_idx_branch[event_idx][hit_idx]
            if(PDG_event[idx] != 13):
                continue
            if (p == -999):
                p = np.sqrt(px_event[hit_idx]**2 + py_event[hit_idx]**2 + pz_event[hit_idx]**2)
                z_hit = z_event[hit_idx]
                theta = np.arctan2(np.sqrt(px_event[hit_idx]**2 + py_event[hit_idx]**2), pz_event[hit_idx]) * 180 / 3.14159
                hit_time = time_event[hit_idx]
                edep_event = EDep_event[hit_idx]
                PDG_list = PDG_event[idx]
            else:
                edep_event += EDep_event[hit_idx]
        data[event_idx] = np.stack([z_hit,theta,p,hit_time,(np.floor(calculate_num_pixels_z_dependence(edep_event,z_hit)).astype(int))],axis = -1)


    
    return data #returns list: each entry is a diff event array; each event array has shape: (#unique particles, #layers, #features)
                #features: z hit, hit time, theta, p, energy dep

def process_data(uproot_path, file_num=0, particle="pion"):
    num_layers = 28
    data = []
    events = up.open(uproot_path)
    
    x_pos_branch = events["HcalBarrelHits/HcalBarrelHits.position.x"].array(library='np')
    z_pos_branch = events["HcalBarrelHits/HcalBarrelHits.position.z"].array(library='np')
    EDep_branch = events["HcalBarrelHits.EDep"].array(library='np')
    PDG_branch = events["MCParticles.PDG"].array(library='np')
    x_momentum_branch = events["HcalBarrelHits/HcalBarrelHits.momentum.x"].array(library='np')
    y_momentum_branch = events["HcalBarrelHits/HcalBarrelHits.momentum.y"].array(library='np')
    z_momentum_branch = events["HcalBarrelHits/HcalBarrelHits.momentum.z"].array(library='np')
    Hits_MC_idx_branch = events["_HcalBarrelHits_MCParticle.index"].array(library='np')
    time_branch = events["HcalBarrelHits.time"].array(library='np')   
    num_events = len(x_pos_branch)
    for event_idx in range(num_events):
        Hits_MC_idx_event = Hits_MC_idx_branch[event_idx]
        PDG_event = PDG_branch[event_idx]
        n_unique_parts, idx_dict = create_unique_mapping(Hits_MC_idx_event)
        
        p_layer_list = np.ones((num_layers,n_unique_parts)) * -999
        z_hit_layer_list = np.ones((num_layers,n_unique_parts)) * -999
        theta_layer_list = np.ones((num_layers,n_unique_parts)) * -999
        hit_time_layer_list = np.ones((num_layers,n_unique_parts)) * -999
        edep_event = np.ones((num_layers,n_unique_parts)) * -999
        PDG_list = np.ones((num_layers,n_unique_parts)) * -999
        
        x_pos_event = x_pos_branch[event_idx]
        px_event = x_momentum_branch[event_idx]
        py_event = y_momentum_branch[event_idx]
        pz_event = z_momentum_branch[event_idx]
        z_event = z_pos_branch[event_idx]
        time_event = time_branch[event_idx]
        EDep_event = EDep_branch[event_idx]
        for hit_idx in range(len(x_pos_event)):
            idx = Hits_MC_idx_branch[event_idx][hit_idx]
            part_idx = idx_dict[idx]
            layer_idx = get_layer(x_pos_event[hit_idx], super_layer_map)
            if layer_idx == -1: #error handling for get_layer
                continue
            elif p_layer_list[layer_idx,part_idx] == -999:
                p_layer_list[layer_idx,part_idx] = np.sqrt(px_event[hit_idx]**2 + py_event[hit_idx]**2 + pz_event[hit_idx]**2)
                z_hit_layer_list[layer_idx,part_idx] = z_event[hit_idx]
                theta_layer_list[layer_idx,part_idx] = np.arctan2(np.sqrt(px_event[hit_idx]**2 + py_event[hit_idx]**2), pz_event[hit_idx])
                hit_time_layer_list[layer_idx,part_idx] = time_event[hit_idx]
                edep_event[layer_idx,part_idx] = EDep_event[hit_idx]
                PDG_list[layer_idx,part_idx] = PDG_event[part_idx]
            else:
                edep_event[layer_idx,part_idx] += EDep_event[hit_idx]
        data.append(np.stack([z_hit_layer_list,theta_layer_list,p_layer_list,hit_time_layer_list,(np.floor(calculate_num_pixels_z_dependence(edep_event,z_hit_layer_list)).astype(int))],axis = -1))


    
    return data #returns list: each entry is a diff event array; each event array has shape: (#unique particles, #layers, #features)
                #features: z hit, hit time, theta, p, energy dep
    
def load_real_data(file_dir):
    time_branch_name = "HcalBarrelHits.time"
    hit_x_branch_name = "HcalBarrelHits.position.x"
    tree_ext = ":events"
    file_names = [(file_dir + name + tree_ext) for name in os.listdir(file_dir) if not os.path.isdir(os.path.join(file_dir, name))]
        # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_file, file_names))

    # Combine results
    event_times = np.concatenate([r[0] for r in results])
    event_x_hit = np.concatenate([r[1] for r in results])

    # Process events
    truth_times = []
    for times, x_hits in zip(event_times, event_x_hit):
        mask = times < 50
        if np.any(mask):  # Only process if there are hits passing the time condition
            try:
                layer_idx = vectorized_get_layer(x_hits[mask])
                truth_times.extend(np.column_stack((times[mask], layer_idx)))
            except Exception as e:
                print(f"Error processing event: {e}")
                print(f"x_hits[mask]: {x_hits[mask]}")
                continue

    truth_times = np.array(truth_times)

    print(f"Processed {len(truth_times)} hits")
    return truth_times

def process_times(uproot_path,threshold = 10, multipleFiles = False):
    if(multipleFiles):
        times_arrays_list = []
        cells_arrays_list = []
        x_pos_arrays_list = []
        y_pos_arrays_list = []
        z_pos_arrays_list = []
        px_arrays_list = []
        py_arrays_list = []
        pz_arrays_list = []

        # Loop through all files in the directory
        for file_name in os.listdir(uproot_path):
            if file_name.endswith(".root"):  # Ensure we're only processing ROOT files
                file_path = os.path.join(uproot_path, file_name)

                # Open the ROOT file
                with up.open(file_path) as file:
                    # Open tree
                    tree = file["events"]

                    times_arrays_list.append(tree["HcalBarrelHits/HcalBarrelHits.time"].array(library="np"))
                    cells_arrays_list.append(tree["HcalBarrelHits/HcalBarrelHits.cellID"].array(library="np"))
                    x_pos_arrays_list.append(tree["HcalBarrelHits/HcalBarrelHits.position.x"].array(library="np"))
                    y_pos_arrays_list.append(tree["HcalBarrelHits/HcalBarrelHits.position.y"].array(library="np"))
                    z_pos_arrays_list.append(tree["HcalBarrelHits/HcalBarrelHits.position.z"].array(library="np"))
                    
                    px_arrays_list.append(tree["HcalBarrelHits/HcalBarrelHits.momentum.x"].array(library="np"))
                    py_arrays_list.append(tree["HcalBarrelHits/HcalBarrelHits.momentum.y"].array(library="np"))
                    pz_arrays_list.append(tree["HcalBarrelHits/HcalBarrelHits.momentum.z"].array(library="np"))

        # Combine arrays for each branch
        times = np.concatenate(times_arrays_list)
        cells = np.concatenate(cells_arrays_list)
        x_pos_branch = np.concatenate(x_pos_arrays_list)
        y_pos_branch = np.concatenate(y_pos_arrays_list)
        z_pos_branch = np.concatenate(z_pos_arrays_list)

        # Now combined_arrays contains the concatenated arrays for each branch across all files
    else:
        events = up.open(uproot_path)

        times = events["HcalBarrelHits/HcalBarrelHits.time"].array(library='np')
        cells = events["HcalBarrelHits/HcalBarrelHits.cellID"].array(library='np')
        x_pos_branch = events["HcalBarrelHits/HcalBarrelHits.position.x"].array(library='np')
        y_pos_branch = events["HcalBarrelHits/HcalBarrelHits.position.y"].array(library='np')
        z_pos_branch = events["HcalBarrelHits/HcalBarrelHits.position.z"].array(library='np')
        
    accepted_times = []
    second_lowest_list = []
    avg_accepted_times = []
    rel_accepted_times = []

    duplicates = 0
    total = 0
    total_cells = []

    skipped = 0
    num_list = []
    #First loop over events
    for event_num in range(len(cells)):

        #skip events with less than 2 photons
    #     if(times[event_num].shape[0] < threshold): continue

        #Keep track of which cell IDs are hit
        curr_list = []
        for photon_num in range(len(cells[event_num])):
            if(cells[event_num][photon_num] in curr_list):
                duplicates += 1
            else:
                curr_list.append(cells[event_num][photon_num])
            if(cells[event_num][photon_num] not in total_cells):
                total_cells.append(cells[event_num][photon_num])
            total += 1
        num_list.append(len(curr_list))
        #check if 2 unique pixels are hit
        if(len(curr_list) < threshold): 
            skipped += 1
            continue
        curr_min = min(times[event_num])
        accepted_times.append(curr_min)
        second_lowest_list.append(min([x for x in times[event_num] if x != curr_min]))
        avg_accepted_times.append(avg_time(threshold,times[event_num]))
    #     if(len(curr_list) == 15):
    #         print(f"event #{event_num}")
    print(f"total: {total} | duplicates: {duplicates} | ratio: {duplicates / total} | num unique cells hit: {len(total_cells)} | skipped: {skipped}")
    return accepted_times, second_lowest_list, avg_accepted_times

def get_all_times(uproot_path,threshold = 10, multipleFiles = False):
    if(multipleFiles):
        times_list = []

        # Loop through all files in the directory
        for file_name in os.listdir(uproot_path):
            if file_name.endswith(".root"):  # Ensure we're only processing ROOT files
                file_path = os.path.join(uproot_path, file_name)

                # Open the ROOT file
                with up.open(file_path) as file:
                    # Open tree
                    tree = file["events"]

                    times_list.append(tree["HcalBarrelHits/HcalBarrelHits.time"].array(library="np"))

        # Combine arrays for each branch
        times = np.concatenate(times_list)

        # Now combined_arrays contains the concatenated arrays for each branch across all files
    else:
        events = up.open(uproot_path)

        times = events["HcalBarrelHits/HcalBarrelHits.time"].array(library='np')
        
    #First loop over events
    flattened_times = np.concatenate(times)
#         for i in range(len(times[event_num])):
    return flattened_times

def train_NF_timing(model,device, train_data, val_data, batch_size,num_context,num_epochs = 25, max_upticks = 5, validation_frequency = 2000, lr = 1e-4,model_path = "../models/",run_info = ""):
    max_iter = int(np.floor(train_data.shape[0] / batch_size))
    train_loss_hist = np.array([])
    val_loss_hist = np.array([])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    early_stopping_dict = {
            "lowest_loss" : -1,
            "best_model_path" : "",
            "num_upticks" : 0
    }
    for epoch in range(0,num_epochs):
        print(f"Beginning epoch #{epoch}")
        model.train()  # Set model to training mode
        train_loss = 0
        for it in tqdm(range(max_iter)):
            optimizer.zero_grad()
            # Get training samples
            begin = it * batch_size
            end = (it + 1) * batch_size
            it_data = train_data[begin:end]
            context = torch.empty(it_data.size()[0], num_context)
            context[:,0] = it_data[:,0]
            context[:,1] = it_data[:,1]
            context[:,2] = it_data[:,2]
            context = context.to(device)
            samples = (it_data[:,4] - it_data[:,3]).unsqueeze(1).to(device)
            # Compute loss
            loss = model.forward_kld(samples, context)
            # Do backprop and optimizer step
            if ~(torch.isnan(loss) | torch.isinf(loss)):
                loss.backward()
                optimizer.step()
            # Log loss
            train_loss_hist = np.append(train_loss_hist, loss.to('cpu').data.numpy())
            train_loss += loss.to('cpu').data.numpy()

        model.eval()  # Set model to evaluation mode
        val_loss = 0
        val_iter = min(100, int(np.floor(val_data.shape[0] / batch_size)))  # Limit validation to 100 batches
        with torch.no_grad():
            for val_it in range(val_iter):
                begin = val_it * batch_size
                end = (val_it + 1) * batch_size
                it_data = val_data[begin:end]
                context = torch.empty(it_data.size()[0], num_context)
                context[:,0] = it_data[:,0]
                context[:,1] = it_data[:,1]
                context[:,2] = it_data[:,2]
                context = context.to(device)
                samples = (it_data[:,4] - it_data[:,3]).unsqueeze(1).to(device)
                loss = model.forward_kld(samples, context)
                val_loss += loss.item()

        avg_val_loss = val_loss / val_iter
        print(f"train_loss: {train_loss / max_iter}\nval_loss: {avg_val_loss}")
        val_loss_hist = np.append(val_loss_hist, avg_val_loss)

        #Early stopping
        if(avg_val_loss < early_stopping_dict["lowest_loss"] or early_stopping_dict["lowest_loss"] == -1):
            early_stopping_dict["lowest_loss"] = avg_val_loss
            model.save(model_path + run_info + f"_checkpoint_e{epoch}.pth")
            early_stopping_dict["best_model_path"] = model_path + run_info + f"_checkpoint_e{epoch}.pth"
        elif(avg_val_loss > early_stopping_dict["lowest_loss"]):
            print("Validation loss increased, logging uptick...")
            early_stopping_dict["num_upticks"] += 1
        if(early_stopping_dict["num_upticks"] >= max_upticks):
            print("Exceeded max # of upticks, loading best model...")
            model.load(early_stopping_dict["best_model_path"])
            return model, train_loss_hist,val_loss_hist


#             print(f"Step {global_step} - Train Loss: {train_loss_hist[-1]:.4f}, Val Loss: {avg_val_loss:.4f}")

        model.train()  # Set model back to training mode
        print(f"Epoch {epoch} completed.")
    print(f"Exceeding max # epochs, loading best model...")
    if(early_stopping_dict["lowest_loss"] != -1):
        model.load(early_stopping_dict["best_model_path"])

    return model, train_loss_hist,val_loss_hist
