import uproot, numpy as np, matplotlib.pyplot as plt, awkward as ak, timeit

sector_count = 8 # number of radial sectors
superlayer_count = 14 # number of superlayers
first_sens_sublayer_pos = 182.58 * 10 # position of first sensitive sublayer
superlayer_dist = (1.06 + 6.61) * 10 # distance between superlayers
adj_sens_sublayer_dist = 1.06 * 10 # distance between adjacent sensitive sublayers within the same superlayer
sens_sublayer_thick = 10 # thickness of sensitive sublayer

# array containing the start pos of each sensitive sublayer
layer_pos = np.zeros(superlayer_count * 2) # 2 sensitive sublayers per superlayer
layer_pos[::2] = [first_sens_sublayer_pos + superlayer_dist*i for i in range(superlayer_count)] # first sublayers
layer_pos[1::2] = layer_pos[::2] + adj_sens_sublayer_dist # second sublayers


def p_func(px,py,pz):
    return np.sqrt(px**2 + py**2 + pz**2)

# returns the distance in the direction of the nearest sector
def sector_proj_dist(xpos, ypos):
    sector_angle = (np.arctan2(ypos, xpos) + np.pi / sector_count) // (2*np.pi / sector_count) * 2*np.pi / sector_count # polar angle (in radians) of the closest sector
    return xpos * np.cos(sector_angle) + ypos * np.sin(sector_angle) # scalar projection of position vector onto unit direction vector

# returns the layer number for the position of a detector hit
def layer_num(xpos, ypos):
    pos = sector_proj_dist(xpos, ypos)

    # false if hit position is before the first sensitive sublayer or after the last sensitive sublayer
    within_layer_region = np.logical_and(pos * 1.0001 > layer_pos[0], pos / 1.0001 < layer_pos[-1] + sens_sublayer_thick)

    superlayer_index = np.where(within_layer_region, ak.values_astype( (pos * 1.0001 - layer_pos[0]) // superlayer_dist, 'int64'), -1) # index of superlayer the hit may be in, returns -1 if out of region
    layer_pos_dup = ak.Array(np.broadcast_to(layer_pos, (int(ak.num(superlayer_index, axis=0)), len(layer_pos)))) # turn layer_pos into a 2d array with duplicate rows to allow indexing
    dis_from_first_sublayer = np.where(within_layer_region, pos - layer_pos_dup[superlayer_index * 2], -1) # distance of hit from the first sublayer in the superlayer, returns -1 if out of region

    # true if hit is within the first of the paired layers
    in_first_layer = np.logical_and(within_layer_region, dis_from_first_sublayer / 1.0001 <= sens_sublayer_thick)
    # true if hit is within the second of the paired layers
    in_second_layer = np.logical_and(within_layer_region, np.logical_and(dis_from_first_sublayer * 1.0001 >= adj_sens_sublayer_dist, dis_from_first_sublayer / 1.0001 <= adj_sens_sublayer_dist + sens_sublayer_thick))

    # layer number of detector hit; returns -1 if not in a layer
    hit_layer = np.where(in_first_layer, superlayer_index * 2 + 1, -1)
    hit_layer = np.where(in_second_layer, superlayer_index * 2 + 2, hit_layer)
    return hit_layer

# returns the number of pixels detected by a hit
def pixel_num(energy_dep, zpos):
    inverse = lambda x, a, b, c : a / (x + b) + c
    efficiency = inverse(770 - zpos, 494.98, 9.9733, -0.16796) # ratio of photons produced in a hit that make it to the sensor
    return 10 * energy_dep * (1000 * 1000) * efficiency

# takes in position and energy deposited for a hit as ragged 2d awkward arrays
# with each row corresponding to hits produced by a particle and its secondaries
# returns 1d array containing number of terminating tracks for each layer, starting at layer 1
def layer_counts(xpos, ypos, zpos, energy_dep):
    hit_layer = layer_num(xpos, ypos)
    hit_layer_filtered = np.where(pixel_num(energy_dep, zpos) >= 2, hit_layer, -2) # only accept layers with at least 2 pixels
    layers_traveled = ak.max(hit_layer_filtered, axis=1) # max accepted layers traveled for a track determines total layers traveled
    layer_counts = np.asarray(ak.sum(layers_traveled[:, None] == np.arange(1, superlayer_count * 2 + 1), axis=0)) # find counts for each layer, 1 through max
    return layer_counts

with uproot.open(f"/cwork/rck32/eic/work_eic/root_files/June_24/variation_full/mu_1GeV_10kevents_theta_90.edm4hep.root") as file:
    mu_hit_x = file['events/HcalBarrelHits.position.x'].array()
    mu_hit_y = file['events/HcalBarrelHits.position.y'].array()
    mu_hit_z = file['events/HcalBarrelHits.position.z'].array()
    mu_hit_edep = file['events/HcalBarrelHits.EDep'].array()

with uproot.open(f"/cwork/rck32/eic/work_eic/root_files/June_24/variation_full/pi_1GeV_10kevents_theta_90.edm4hep.root") as file:
    pi_hit_x = file['events/HcalBarrelHits.position.x'].array()
    pi_hit_y = file['events/HcalBarrelHits.position.y'].array()
    pi_hit_z = file['events/HcalBarrelHits.position.z'].array()
    pi_hit_edep = file['events/HcalBarrelHits.EDep'].array()

mu_layer_counts = layer_counts(mu_hit_x, mu_hit_y, mu_hit_z, mu_hit_edep)
pi_layer_counts = layer_counts(pi_hit_x, pi_hit_y, pi_hit_z, pi_hit_edep)

plt.figure(figsize=(16, 8))

plt.plot(np.arange(1, 29), mu_layer_counts, '-o', label='muons')
plt.plot(np.arange(1, 29), pi_layer_counts, '-o', label='pions')

plt.title('Layers traveled by track')
plt.xlabel('Layers')
plt.ylabel('Count')
plt.legend(loc='upper left')
plt.savefig("plots/test_june_24_theta_90.jpeg")
# plt.show()

tp = 0
fn = 0
fp = 0
tn = 0
total = np.sum(mu_layer_counts) + np.sum(pi_layer_counts)

for i in range(28):
    if mu_layer_counts[i] > pi_layer_counts[i]:
        tp += mu_layer_counts[i]
        fp += pi_layer_counts[i]
    elif mu_layer_counts[i] < pi_layer_counts[i]:
        tn += pi_layer_counts[i]
        fn += mu_layer_counts[i]
    else:
        if mu_layer_counts[i] == 0:
            continue
        print('equality')
        break

print("\t\t\tid'd as muon\t\t\tid'd as pion")
print(f"is muon\t\t\tTrue Positive: {tp / total:2.1%}\t\tFalse Negative: {fn / total:2.1%}")
print(f"is pion\t\t\tFalse Positive: {fp / total:2.1%}\t\tTrue Negative: {tn / total:2.1%}")