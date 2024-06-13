Goal:

1. parameterize the photon stuff based on just the energy, and the two angles of the charged particle.
   1. Should be able to run a bunch of simulations at different energy, phi, theta, and then fit the resulting responses somehow

Guesses

1. Want to have input be the energy, phi, theta and then output time, energy 





KLM w/Ian

1. get simulation running with full layer, get output of # of layers traversed to create simple PID for muon pion
   1. Fit # of layers traversed to gaussian, binned in momentum bins
      1. Can fit the dependency for each bin

# end of day notes June 11 (tuesday)

1. confirmed that 10 photons are generated for every KeV deposited by muon/pion. 0.05% of photons reach SiPM
2. Need to check if the energy deposit (and hence photon generation, and even maybe photons hitting sensor) differ for pions vs muons, or if the difference mainly comes from the pion hadronic shower via iron
3. Prob try to generate files for pi,mu, and sensitive vol being sensor and scint for one segment, then run on multilayer to see how iron changes things

# Today

1. Running simulation with scint sensitive
   1. Can verify that pions and muons deposite same energy in the scintillator
2. Next
   1. Run sims to check how energy deposition depends on theta/phi



Next step

1. Look at the dependence of angle (theta) on photon yield (photons that hit sensor and photons generated)
   1. From there we can have # photons generated as a function of angle (and energy but I doubt that matters)





Tomorrow

1. Watch youtube video about geant4 "inserting sensitive detectors"
2. Goal
   1. Create a single layer of segments with sensor at end
      1. If possible, use one sensor per segment, but if not, try to use one for the whole layer
      2. Make sure readout is correct - accurate # of photons
   2. see how efficiency changes with position/angle of charged particle (also EDep)

## June 13

looking at these videos: https://www.youtube.com/playlist?list=PLui8F4uNCFWm3M3g3LG2cOledhI7IvTAJ

1. Goal
   1. Try to figure out how we can accurately implement the sensors