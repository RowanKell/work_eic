# Work Notes July 5th, 2024

#### EIC KLM Density Estimation, affinity stuff

## EIC KLM NF

Overview

1. Seems to be working well
2. Status
   1. Have figured out data preprocessing and training procedures as well as sampling (and resampling bad data)
      1. Need resampling bc it seems that the learned distribution has weird artifacts below t = 0 which are non-physical

<img src="/home/rowan/.config/Typora/typora-user-images/image-20240705105006451.png" alt="image-20240705105006451" style="zoom:33%;" />

			2. Implemented resampling and get very nice distribution (also needed to normalize distribution due to bin width differences)

<img src="/home/rowan/.config/Typora/typora-user-images/image-20240705105128371.png" alt="image-20240705105128371" style="zoom:33%;" />

3. Seems to fit very well!
4. info:
   1. Data
      1. 4 conditionals: theta, mu hit time (calculated), mu hit position (calculated), mu momentum (from MCParticle px py pz)
      2. one feature: photon hit time on sensor
      3. 80 10 10 train val test split
         1. 11 million datapoints total for this run approx
      4. data is shuffled to avoid having same context back to back
      5. Sampling:
         1. Use test context, sample timing and compare sampled timing to real test timing
         2. Note: does not make sense to look at accuracy of samples directly compared to their context hit by hit
            1. Each context set has many different times resulting from it, and naturally has a distribution (not delta peak) so we must look at the distribution of values not the individual data points when comparing truth to learned
   2. Model
      1. Autoregressive Neural Spline conditional flow
         1. 12 flows
         2. each has latent size of 1 (timing), context size of 4, 64 hidden units, 8 hidden layers
      2. Gaussian base distribution
   3. Training
      1. lr = 1e-3
      2. weight decay = 1e-5
      3. (adam optimizer)
      4. batch size = 100
5. **Issue**
   1. Missing coverage for simulations at high z values as these are taking forever to preprocess...
6. Plan
   1. Run more data, same amount per sim but more sims to have smaller increment between z pos
   2. Train new nf in meantime while data being run
      1. maybe just change # flows and hidden layers?
      2. July_5 NF is 16 flows, 16 hl, 64 hu, 100 batch size
   3. Now have more data, will take forever to train but can train larger model
      1. 12 flows, 16 hl, 128 hu, 400 batch size



## Group meeting

1. EIC KLM
   1. Need to train model on full z position range, then bin by diff z, etc to see if the timing distributions hold up with different dependencies
   2. Implement the solenoid in geometry