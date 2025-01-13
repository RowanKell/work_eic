from DDSim.DD4hepSimulation import DD4hepSimulation

#from DDSim.Helper.Physics import Physics
from g4units import mm, GeV, MeV, deg
SIM = DD4hepSimulation()


# Specify particle gun:
SIM.enableGun
SIM.gun.thetaMin = 90*deg
SIM.gun.thetaMax = 90*deg
SIM.gun.distribution = "uniform"
SIM.gun.phiMin = 90*deg
SIM.gun.phiMax = 90*deg
SIM.gun.momentumMin = 7.0*GeV
SIM.gun.momentumMax = 7.0*GeV
SIM.gun.particle = "kaon0L"
SIM.physics.list = "FTFP_BERT"
SIM.physics.decays = False
#SIM.part.saveProcesses = ['conv','Decay']
#SIM.output.part = "VERBOSE"


#SIM.part.keepAllParticles = True
