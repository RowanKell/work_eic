from DDSim.DD4hepSimulation import DD4hepSimulation

#from DDSim.Helper.Physics import Physics
from g4units import mm, GeV, MeV, deg
SIM = DD4hepSimulation()

# Specify particle gun:
SIM.enableGun
SIM.gun.thetaMin = 70*deg
SIM.gun.thetaMax = 110*deg
SIM.gun.distribution = "cos(theta)"
SIM.gun.phiMin = 0*deg
SIM.gun.phiMax = 360*deg
# SIM.gun.momentumMin = 0.5*GeV
# SIM.gun.momentumMax = 4.0*GeV
SIM.gun.momentumMin = 1.0*GeV
SIM.gun.momentumMax = 3.0*GeV
# SIM.gun.particle = "pi+"
SIM.gun.particle = "neutron"
# SIM.gun.particle = "kaon0L"
SIM.physics.list = "FTFP_BERT"
SIM.physics.decays = False
#SIM.part.saveProcesses = ['conv','Decay']
#SIM.output.part = "VERBOSE"


#SIM.part.keepAllParticles = True
'''
def setupCerenkov(kernel):
  from DDG4 import PhysicsList
  seq = kernel.physicsList()

  cerenkov = PhysicsList(kernel, 'Geant4CerenkovPhysics/CerenkovPhys')
  cerenkov.MaxNumPhotonsPerStep = 10
  cerenkov.MaxBetaChangePerStep = 10.0
  cerenkov.TrackSecondariesFirst = False
  cerenkov.VerboseLevel = 0
  cerenkov.enableUI()
  seq.adopt(cerenkov)

  ph = PhysicsList(kernel, 'Geant4OpticalPhotonPhysics/OpticalGammaPhys')
  ph.addParticleConstructor('G4OpticalPhoton')
  ph.VerboseLevel = 0
  ph.enableUI()
  seq.adopt(ph)

  scint = PhysicsList(kernel, 'Geant4ScintillationPhysics/ScintillationPhys')
  scint.VerboseLevel = 0
  scint.TrackSecondariesFirst = True
  scint.enableUI()
  seq.adopt(scint)

  return None


SIM.physics.setupUserPhysics(setupCerenkov)
'''
