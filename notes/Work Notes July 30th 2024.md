# Work Notes July 30th 2024

## Timing resolution

Method

1. Calculate resolution from std of histogram of times
2. Bin by z hit position
   1. Maybe also bin by theta? or only look at timings where the theta is close to 90
   2. going off of CLAS TOF test stand file:///home/rowan/Downloads/nim_tof12.pdf

### Work notes July 31

1. May need constant mu momentum for time res - we want the incident time to be the same for each muon, or else the arrival time of the muon is factored in 
   1. With variable p, sigma ~=500ps for learned
      1. with fixed at 5GeV, ~240ps
   2. with variable p, sigma ~=120ps for simulated



### Work Notes August 1

##### Timing res

1. Trying to check optph time resolution at higher statistics, but jobs are failing

 ##### Energy resolution

Idea:

1. Train NN
   1. Input: # pixels per layer (maybe timing but then we get variable input dimension)
   2. Output: guess at energy of particle

Plan

1. Run simulations
   1. maybe start with fixed position and angles, only vary momentum from 0.8 to 10GeV
   2. Need to train on both muons and pions, maybe others as well
2. Build NN
   1. simple classification network
   2. 28 input dim
   3. 1 output dim
   4. 