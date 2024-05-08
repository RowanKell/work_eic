from DDSim.DD4hepSimulation import DD4hepSimulation

#from DDSim.Helper.Physics import Physics
from g4units import mm, GeV, MeV, deg

SIM = DD4hepSimulation()


def setupScint(kernel):
  from DDG4 import PhysicsList
  seq = kernel.physicsList()
  scint = PhysicsList(kernel, 'Geant4ScintillationPhysics/ScintillationPhys')
  scint.addParticleConstructor('G4OpticalPhoton')
  scint.VerboseLevel = 1
  scint.enableUI()
  scint.TrackSecondariesFirst = True
  # here you must configure the scintillation physics with the parameter names shown (linked) 
  # below. Same as for `scint.TrackSecondariesFirst = True`
  seq.adopt(scint)
  return None

def setupCerenkov(kernel):
  from DDG4 import PhysicsList
  seq = kernel.physicsList()
  cerenkov = PhysicsList(kernel, 'Geant4CerenkovPhysics/CerenkovPhys')
  cerenkov.MaxNumPhotonsPerStep = 10
  cerenkov.MaxBetaChangePerStep = 10.0
  cerenkov.TrackSecondariesFirst = True
  cerenkov.VerboseLevel = 1
  cerenkov.enableUI()
  seq.adopt(cerenkov)
  return None


def setupPhot(kernel):
  from DDG4 import PhysicsList
  seq = kernel.physicsList()
  ph = PhysicsList(kernel, 'Geant4OpticalPhotonPhysics/OpticalGammaPhys')
  ph.addParticleConstructor('G4OpticalPhoton')
  ph.VerboseLevel = 1
  ph.enableUI()
  seq.adopt(ph)
  return None

SIM.physics.setupUserPhysics(setupCerenkov)
SIM.physics.setupUserPhysics(setupScint)
SIM.physics.setupUserPhysics(setupPhot)

SIM.filter.tracker = 'edep0'

# Some detectors are only sensitive to optical photons
SIM.filter.filters['opticalphotons'] = dict(
  name='ParticleSelectFilter/OpticalPhotonSelector',
  parameter={"particle": "opticalphoton"},
)
# This could probably be a substring
SIM.filter.mapDetFilter['DRICH'] = 'opticalphotons'
SIM.filter.mapDetFilter['RICHEndcapN'] = 'opticalphotons'
SIM.filter.mapDetFilter['DIRC'] = 'opticalphotons'
SIM.filter.mapDetFilter['HCalBarrel'] = 'opticalphotons'

# Use the optical tracker for the DRICH
SIM.action.mapActions['DRICH'] = 'Geant4OpticalTrackerAction'
SIM.action.mapActions['RICHEndcapN'] = 'Geant4OpticalTrackerAction'
SIM.action.mapActions['DIRC'] = 'Geant4OpticalTrackerAction'

# Specify particle gun:
SIM.enableGun
SIM.gun.thetaMin = 90*deg
SIM.gun.thetaMax = 90*deg
SIM.gun.distribution = "cos(theta)"
SIM.gun.phiMin = 0*deg
SIM.gun.phiMax = 0*deg
SIM.gun.momentumMin = 5.0*GeV
SIM.gun.momentumMax = 5.0*GeV
SIM.gun.particle = "pi-"
SIM.physics.list = "FTFP_BERT"
#SIM.physics.list = "FTFP_BERT_EMZ"
SIM.physics.decays = False
#SIM.part.saveProcesses = ['conv','Decay']
SIM.printLevel = 1
SIM.output.part = 1
SIM.part.keepAllParticles = True
