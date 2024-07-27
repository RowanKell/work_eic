# Work Notes July 21st

1. Need to preprocess July 19th slurm run 0
   1. Currently running preprocessing
2. Now training
3. Need to figure out how to get distribution correct
4. Need full sector run with optical photons on...
   1. Maybe do 20 jobs with 200 e



# Work Notes July 22nd

1. Trying to train with cuts on the input data
   1. Some datapoints are weird for the times, so we place a cut that cuts out <0.1% of data - most of these are likely geant4 errors
2. Getting weird training loss where the loss curve is split into different curves that all are being minimized



# Work Notes July 23

1. Need to run new full_sector_scint dataset for inference
   1. Limited theta and momentum just to start with more basic data - if it can't learn this it prob can't learn more complex
   2. theta 90, p = 5, phi = 0
2. Need to ensure that comparison dataset (full sector sensor) is same settings as inference data 



So far:

1. looks decent:
2. <img src="/home/rowan/Downloads/july_22_run_6.jpeg" style="zoom:25%;" />
3. constant theta and momentum

### Next steps

1. need to double check that # photons generated is consistent with truth

# Work Notes Wed July 24th

1. Testing runSim.sh with pions
2. Need to preprocess new 600 z val dataset

### Timing param

1. Overall, distribution matches when integrating over everything:
2. <img src="/home/rowan/Downloads/july_23_run_1_full_theta.jpeg" style="zoom:25%;" />

3. However, when looking at the distributions binned by layer idx, they don't match at higher layer idx
4. **FOUND SOLUTION**
   1. Had a bug where I was using the wrong metadata
      1. layer #s were being assigned to the wrong hits, making it so all of the hit times were on average the same

<img src="/home/rowan/.config/Typora/typora-user-images/image-20240724105534254.png" alt="image-20240724105534254" style="zoom:25%;" />

#### Next steps

1. try to bin by other variables to ensure this works...
2. also need to test pions





delta kT

1. Prof Vossen says
   1. seems like correction from the frames
   2. in breit frame, k = z * qT + dkt
   3. in hadron frame, k = dkt
      1. power suppressed - goes with 1/Q2 - suppressed by large scale