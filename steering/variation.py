"""
This is an altered local test version of the following file:

DD4hep simulation with some argument parsing
Based on M. Frank and F. Gaede runSim.py
   @author  A.Sailer
   @version 0.1

Modified with standard EIC EPIC requirements.
"""
from __future__ import absolute_import, unicode_literals
import logging
import sys

from DDSim.DD4hepSimulation import DD4hepSimulation

from g4units import mm, GeV, MeV, deg, eV


SIM = DD4hepSimulation()

  # Specify particle gun:
SIM.enableGun
SIM.gun.position = (0, 15, 0)

SIM.gun.thetaMin = 68*deg
SIM.gun.thetaMax = 112*deg
SIM.gun.distribution = "uniform"
SIM.gun.phiMin = 67.5*deg #use 22.5 to hit corner, -45 to hit bottom right in middle
SIM.gun.phiMax = 112.5*deg
SIM.gun.momentumMin = 5.0*GeV
SIM.gun.momentumMax = 5.0*GeV
#SIM.gun.particle = "pi-"


# Ensure that Cerenkov and optical physics are always loaded
# Added Scintillation to that list
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

#If true, keeps MCParticles for all photons that don't hit sensor
#Set to false to keep file size much smaller
SIM.part.keepAllParticles = True

  # Allow energy depositions to 0 energy in trackers (which include optical detectors)
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
SIM.filter.mapDetFilter['HcalBarrel'] = 'opticalphotons'

  # Use the optical tracker for the DRICH
SIM.action.mapActions['DRICH'] = 'Geant4OpticalTrackerAction'
SIM.action.mapActions['RICHEndcapN'] = 'Geant4OpticalTrackerAction'
SIM.action.mapActions['DIRC'] = 'Geant4OpticalTrackerAction'
SIM.action.mapActions['HcalBarrel'] = 'Geant4OpticalTrackerAction'

  # Use the optical photon efficiency stacking action for hpDIRC
  #for one layer: slice2seg0

# SIM.action.stack = [
#   {
#     "name": "OpticalPhotonEfficiencyStackingAction",
#     "parameter": {
#       "LambdaMin": "286*nm",
#       "LambdaMax": "900*nm",
#       "LogicalVolume": "slice2seg0",
#       "Efficiency": [e/100. for e in [ 8.2, 11.3, 14.1, 16.7, 19. , 21.1, 23.1, 24.8, 26.4, 27.8, 29.1,
#        30.2, 31.2, 32.2, 33. , 33.7, 34.4, 35. , 35.6, 36. , 36.5, 36.9,
#        37.3, 37.6, 37.9, 38.2, 38.5, 38.7, 39. , 39.2, 39.4, 39.6, 39.9,
#        40.1, 40.3, 40.5, 40.7, 41. , 41.2, 41.4, 41.7, 41.9, 42.2, 42.4,
#        42.7, 42.9, 43.2, 43.5, 43.7, 44. , 44.3, 44.6, 44.9, 45.1, 45.4,
#        45.7, 46. , 46.2, 46.5, 46.8, 47. , 47.3, 47.5, 47.8, 48. , 48.2,
#        48.5, 48.7, 48.9, 49. , 49.2, 49.4, 49.5, 49.7, 49.8, 49.9, 50. ,
#        50.1, 50.1, 50.2, 50.2, 50.3, 50.3, 50.3, 50.2, 50.2, 50.2, 50.1,
#        50. , 49.9, 49.8, 49.7, 49.6, 49.4, 49.3, 49.1, 48.9, 48.7, 48.5,
#        48.3, 48. , 47.8, 47.5, 47.3, 47. , 46.7, 46.4, 46.1, 45.8, 45.5,
#        45.2, 44.8, 44.5, 44.1, 43.8, 43.4, 43.1, 42.7, 42.4, 42. , 41.6,
#        41.3, 40.9, 40.5, 40.1, 39.8, 39.4, 39. , 38.6, 38.2, 37.9, 37.5,
#        37.1, 36.8, 36.4, 36. , 35.7, 35.3, 35. , 34.6, 34.2, 33.9, 33.6,
#        33.2, 32.9, 32.6, 32.2, 31.9, 31.6, 31.3, 31. , 30.7, 30.4, 30.1,
#        29.8, 29.5, 29.2, 28.9, 28.6, 28.4, 28.1, 27.8, 27.6, 27.3, 27.1,
#        26.8, 26.6, 26.3, 26.1, 25.8, 25.6, 25.4, 25.1, 24.9, 24.7, 24.4,
#        24.2, 24. , 23.8, 23.5, 23.3, 23.1, 22.9, 22.7, 22.4, 22.2, 22. ,
#        21.8, 21.6, 21.4, 21.1, 20.9, 20.7, 20.5, 20.3, 20. , 19.8, 19.6,
#        19.4, 19.2, 19. , 18.7, 18.5, 18.3, 18.1, 17.9, 17.6, 17.4, 17.2,
#        17. , 16.8, 16.6, 16.3, 16.1, 15.9, 15.7, 15.5, 15.3, 15.1, 14.9,
#        14.7, 14.5, 14.3, 14.1, 13.9, 13.7, 13.5, 13.3, 13.2, 13. , 12.8,
#        12.6, 12.5, 12.3, 12.2, 12. , 11.9, 11.7, 11.6, 11.4, 11.3, 11.2,
#        11.1, 10.9, 10.8, 10.7, 10.6, 10.5, 10.4, 10.3, 10.2, 10.1, 10.1,
#        10. ,  9.9,  9.8,  9.7,  9.7,  9.6,  9.5,  9.5,  9.4,  9.3,  9.2,
#         9.2,  9.1,  9. ,  8.9,  8.9,  8.8,  8.7,  8.6,  8.5,  8.4,  8.3,
#         8.2,  8.1,  8. ,  7.8,  7.7,  7.6,  7.4,  7.3,  7.1,  6.9,  6.8,
#         6.6,  6.4,  6.2,  6.1,  5.9,  5.7,  5.5,  5.3,  5.1,  5. ,  4.8,
#         4.6,  4.5,  4.3,  4.2,  4.1,  4.1,  4. ,  4. ,  4.1,  4.1,  4.3]]
#     }
#   }
# ]


