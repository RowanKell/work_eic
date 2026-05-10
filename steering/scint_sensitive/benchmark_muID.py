"""
Steering file for conventional muID benchmark.
Scintillator-sensitive mode (no optical photons).
Momentum, theta range, and particle type are set via command line args.
"""
from DDSim.DD4hepSimulation import DD4hepSimulation
from g4units import mm, GeV, MeV, deg

SIM = DD4hepSimulation()

SIM.enableGun
SIM.gun.distribution = "uniform"
SIM.gun.phiMin = 0*deg
SIM.gun.phiMax = 360*deg

SIM.physics.list = "FTFP_BERT"
SIM.physics.decays = False
SIM.part.userParticleHandler = ""
