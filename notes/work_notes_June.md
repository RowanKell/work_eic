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
   1. Try to figure out how we can accurately implement the sensor
2. Progress
   1. Found solution for the error that appeared when more than 32 sensors were placed in a layer - needed to give more bits to the slice bit value in the ID
      1. This is found in the bottom of the klmws.xml file
   2. Currently working on getting ready for simulating optph production / sensing for different phi,theta, and energy values

layer_map = [1830.8000, 1841.4000, 1907.5, 1918.1,1984.1999, 1994.8000, 2060.8999, 2071.5, 2137.6001, 2148.1999, 2214.3000, 2224.8999, 2291, 2301.6001, 2367.6999, 2378.3000, 2444.3999, 2455, 2521.1001, 2531.6999, 2597.8000,2608.3999, 2674.5, 2685.1001, 2751.1999, 2761.8000, 2827.8999, 2838.5]

## June 17

1. Try to clean up code into util file so we can have a set of functions to pull from rather than a too long jupyter notebook
2. Try to get a functional form of the dependence of angle on optical photon production and % received by SiPM
3. Maybe implement magnet?
   1. **Maybe skip for now...**
   2. NOTE:
      1. 
      2. To create an xml geometry file like (epic_klmws_only.xml) one needs to only create a yml file in the configuration folder and include "features" and then the compact files you want.
      3. using this, created new configuration with solenoid included

#### Goals

1. Parameterize each charged track that hits detector
   1. Geant4 gives energy dep, pos, angle, time, etc
   2. Use energy dep, pos, angle to calculate # of photons
      1. Use this to see if we get 2 photons (or whatever threshold)
   3. Use # of photons + position to get timing + resolution
      1. See below
2. Now, we have timing, energy deposited, etc, and can use to perform PID
   1. PID can be function (NN or not) that takes in the momentum of particle, timing information for each layer, energy information for each layer, and then returns classification (pi or mu)

##### Timing parameterization

1. Tricky, as each secondary + primary will create photons/deposit energy
2. Idea
   1. Create distribution of first photon times from simulations - **use relative timing - time between particle hit and photon hit?**
      1. This will allow us to draw for each charged track a time value
   2. If multiple particles hit at similar times, we can use all of the first photons and either take the first as the hit time or take an average
      1. Need to see which gives us the best timing resolution

## Current understanding

1. Goal: take hcalbarrelhit on scintillator and get all information from that
   1. In particular use the position, angle of incidence, and energy
2. Need to know how to relate these variables to actual detector output
   1. Easy to count number of photons generated - 10 per KeV deposited
   2. Harder to count number of photons that reach detector - seems to be a function of the position it hits the bar (the further the lower the percentage) - find empirically
   3. angles can also factor into this as the predict position, so they share the same dependence
3. Timing
   1. We can get timing from # of photons and position probably
   2. Need to know the barrel length and if we will have 2 separate sensors on each side or not - this will affect the distribution
      1. if we have two separate sensors, and cannot actually build a geometry with two, we should just be able to split the sensor in half and act like the sensor is on the closer half? so take x distance to sensor to be the lower of the two distances to the two sensors

# Tuesday June 18

1. Maybe try a basic classification network?
   1. Need one_sector geometry
   2. run 10k events for both mu and pi?
   3. vary the angle and energy

### Meeting

Pre-notes

1. Goal: take hcalbarrelhit on scintillator and get all information from that
   1. In particular use the position, angle of incidence, and energy
2. Need to know how to relate these variables to actual detector output
   1. Easy to count number of photons generated - 10 per KeV deposited
   2. Harder to count number of photons that reach detector - seems to be a function of the position it hits the bar (the further the lower the percentage) - find empirically
   3. angles can also factor into this as the predict position, so they share the same dependence
3. Timing
   1. We can get timing from # of photons and position probably
   2. Need to know the barrel length and if we will have 2 separate sensors on each side or not - this will affect the distribution
      1. if we have two separate sensors, and cannot actually build a geometry with two, we should just be able to split the sensor in half and act like the sensor is on the closer half? so take x distance to sensor to be the lower of the two distances to the two sensors
4. Classification - what methods?b 
   1. Can try a NN - basic linear + ReLU activation, trying right now with 300,000 events each for mu pi
      1. input:
      2. for each layer, # of hits and total energy deposited - 56 total features
      3. 2 classes - pion or muon
      4. 10 hidden layers, h_dim = 256, output dim = 2
   2. First preliminary result - 93.15% accuracy?



##### In meeting notes

1. optimization - optimize pion muon separation, maybe cost / size of detector
2. Start Ian with basic separation
   1. Use hcalbarrelhits edep and x pos to find energy dep per layer, convert to # of photons and then convert to # of pixels
   2. Use 2 pixel threshold to count # of layers traversed
   3. Cut on # of layers traversed to separate pions and muons



Note

1. We have momentum info from tracker