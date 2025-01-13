from podio.root_io import Reader

reader = Reader("/hpc/group/vossenlab/rck32/eic/work_eic/root_files/test/pythia8NCDIS_10x100.edm4hep.root")
idx = 0
for event in reader.get("events"):
    if(idx != 6):
        idx+=1
        continue
    print(f"event #{idx}")
    hits = event.get("HcalBarrelHits")
    hit_idx = 0
    for hit in hits:
        print(f"hit #{hit_idx}")
        MCParticle = hit.getMCParticle()
        print(MCParticle.getParents())
        break
        hit_idx+=1
    idx+=1