import dd4hep
import ROOT

# Load the geometry
lcdd = dd4hep.Detector.getInstance()
eic_pref = "/hpc/group/vossenlab/rck32/eic/"
lcdd.fromXML(eic_pref + "epic_klm/epic_klmws_only.xml")

# Open the ROOT file
file_pref = eic_pref + "work_eic/root_files/test/"
f = ROOT.TFile(file_pref + "n_50_scint_sens_one_sector.edm4hep.root")
tree = f.Get("events")

def get_bar_info(hit):
    cellID = hit.cellID
    print(f"CellID: {cellID}")
    
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
        "module": module,
        "layer": layer,
        "slice": slice_id
    }


# In your main loop:
for event in tree:
    hits = event.HcalBarrelHits 
    for hit in hits:
        bar_info = get_bar_info(hit)
        if bar_info:
            print(f"Hit info: {bar_info}")
