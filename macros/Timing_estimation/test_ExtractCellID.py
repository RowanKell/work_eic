
import dd4hep
import ROOT
import ctypes
import matplotlib.pyplot as plot

lcdd = dd4hep.Detector.getInstance()
eic_pref = "/hpc/group/vossenlab/rck32/eic/"
lcdd.fromXML(eic_pref + "epic_klm/epic_klmws_only.xml")
file_pref = eic_pref + "work_eic/root_files/test/"
f = ROOT.TFile(file_pref + "full_sector_50.edm4hep.root")
tree = f.Get("events")



def get_bar_info(hit):
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
    target_layer = hit_info['layer']
    total_slice = hit_info['slice']
    target_segment = total_slice // 7
    target_slice = total_slice % 7

    # Get HcalBarrelVolume
    HcalBarrelVolume = world_volume.GetNodes()[0].GetVolume()  # Assuming HcalBarrelVolume is the first child of world_volume

    # Access stave directly
    stave_name = f"stave_{target_stave}"
    stave = HcalBarrelVolume.FindNode(stave_name)
    if not stave:
        print(f"Stave {stave_name} not found")
        return None

    # Access layer directly
    layer_name = f"layer{target_layer}_{target_layer - 1}"
    layer = stave.GetVolume().FindNode(layer_name)
    if not layer:
        print(f"Layer {layer_name} not found")
        return None
#     for node in layer.GetNodes():
#         print(node.GetName())
    # Access slice directly
    slice_name = f"seg{target_segment}slice{target_slice+1}_{total_slice}"
    slice_node = layer.GetVolume().FindNode(slice_name)
    if not slice_node:
        print(f"Slice {slice_name} not found")
        return None

    return print_position(slice_node)

# Helper function to get position (unchanged)
def print_position(node):
    transformation = node.GetMatrix()
    x = transformation.GetTranslation()[0]
    y = transformation.GetTranslation()[1]
    z = transformation.GetTranslation()[2]
    return [x, y, z]

# Main loop (simplified)
world_volume = lcdd.worldVolume()
x, y, z = [], [], []

for event in tree:
    for hit in event.HcalBarrelHits:
        bar_info = get_bar_info(hit)
        if bar_info:
            result = find_volume(world_volume, bar_info)
            if result:
                x.append(result[0])
                y.append(result[1])
                z.append(result[2])
            else:
                print("Skipped one event")

# Plot histogram (unchanged)
plot.hist(x, bins=500)
plot.savefig("test/plot.jpeg")
'''
def print_position(node):
    transformation = node.GetMatrix()

    # Extract translation components (x, y, z)
    x = transformation.GetTranslation()[0]
    y = transformation.GetTranslation()[1]
    z = transformation.GetTranslation()[2]

#     print(f"  Placement Position: x={x:.2f}, y={y:.2f}, z={z:.2f}")

    # Check if transformation has rotation
    if isinstance(transformation, ROOT.TGeoRotation):
        # Extract rotation matrix
        rotation = transformation.GetRotation()

        # Get Euler angles using GetAngles method
        phi = ctypes.c_double()
        theta = ctypes.c_double()
        psi = ctypes.c_double()
        rotation.GetAngles(phi, theta, psi)

#         print(f"  Rotation (Euler angles, radians): phi={phi.value:.4f}, theta={theta.value:.4f}, psi={psi.value:.4f}") 
#     else:
#         print("  No rotation")
    return [x,y,z]
#heirarchy:
# detector
#   
def find_volume(world_volume,hit_info):
#     name = volume.GetName()
    target_stave = hit_info['stave']
    target_layer = hit_info['layer']
    total_slice = hit_info['slice']
    target_segment = total_slice // 7
    target_slice = total_slice % 7
    for node in world_volume.GetNodes():
        HcalBarrelVolume = node.GetVolume()
#         print(f"Hcal volume name: {HcalBarrelVolume.GetName()}")
        for stave in HcalBarrelVolume.GetNodes():
            stave_volume = stave.GetVolume()
#             print(f"stave volume name: {stave.GetName()}")
            target_save_name = "stave_" + str(target_stave)
#             print(f"target stave volume name: {target_save_name}")
            
            if ("stave_" + str(target_stave))!= stave.GetName():
                continue
#             print("found correct stave")
            for layer in stave_volume.GetNodes():
                layer_volume = layer.GetVolume()

#                 print(f"layer volume name: {layer.GetName()}")
                target_layer_name = "layer" + str(target_layer) + "_" + str(target_layer - 1)
#                 print(f"target layer volume name: {target_layer_name}")
                if ("layer" + str(target_layer) + "_" + str(target_layer - 1))!= layer.GetName():
                    continue
#                 print("found correct layer")
                for sli in layer_volume.GetNodes():
                    slice_volume = sli.GetVolume()
#                     print(slice_volume.GetName())
                    target_slice_name = ("seg" + str(target_segment) + "slice" + str(target_slice))
#                     print(f"target slice name: {target_slice_name}")
                    if target_slice_name!= slice_volume.GetName():
                        continue
#                     print("found correct slice")
                    return print_position(sli)

#         print("failed to find correct slice")
    
x = []
y = []
z = []
    
world_volume = lcdd.worldVolume()
# In your main loop:
for event in tree:
    hits = event.HcalBarrelHits 
    for hit in hits:
        bar_info = get_bar_info(hit)
        if bar_info:
#             print(f"Hit info: {bar_info}")
            result = find_volume(world_volume,bar_info)
            try:
                x.append(result[0])
                y.append(result[1])
                z.append(result[2])
            except:
                print("skipped one event")
            
plot.hist(x,bins = 500)
plot.savefig("test/plot.jpeg")
'''