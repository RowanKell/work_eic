import numpy as np
import uproot as up
import os

def checkdir(path):
    if not os.path.exists(path): 
        os.makedirs(path)
def avg_time(num,event_times):
    ten_sorted_times = sorted(event_times)[:num]
    return sum(ten_sorted_times) / len(ten_sorted_times)

def process_times(uproot_path,threshold = 10, multipleFiles = False):
    if(multipleFiles):
        times_arrays_list = []
        cells_arrays_list = []
        x_pos_arrays_list = []
        y_pos_arrays_list = []
        z_pos_arrays_list = []

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