# Work Notes Tuesday August 6th 2024

## EIC KLM

#### Fixing parameterization

1. NF doesn't work for real data (scintillator) - need to revise how we connect the scint and sensor types of data

Current model

1. Training data
   1. Shoot 1 mm away from bar, use gun momentum and theta, calculate z hit pos time of incident
2. Real data
   1. Shoot 1 mm away from bar, use 

Problem

1. NF does well at learning how to estimate the real data using real data inputs
2. However, NF does not do nearly as well at estimating the real dating using scintillator inputs
   1. Need the scintillator inputs to match the real data inputs



Plan

1. Check real data inputs vs scintillator inputs
   1. Run scintillator + create inputs
   2. Run sensor + create inputs
   3. Run exactly the same for 1 or 10 events with exact same setup
      1. Same angle, position, momentum, etc
      2. check if the calculated position, time match up
      3. My suspicion is that the position or the time is being calculated incorrectly causing a slight diff in the distributions

Update

1. First look
2. Seems that the biggest diff is in the z hit pos
   1. Consistently get ~-0.5 for train but -3 for estimate (can't see on graph for some reason)
      1. Not a huge diff but could cause some issues?
3. others have small issues but prob less important

<img src="/home/rowan/Downloads/theta_90_pos_0_0(1).jpeg" style="zoom:25%;" />

#### Solution 

Seems like the issue is that the training data has the z hit pos calculated at the front of the scintillator, but the actual hit pos registered by the scintillator sensitive simulation is in the middle. So, to accurately get this position I needed to change the value of the x distance in the tangent formula to be 6 rather than 1 (hit is in middle at 1775.3, gun at 1769.3, so x distance is 6 mm away).

Now, we get agreement

![](/home/rowan/Pictures/Screenshots/Screenshot from 2024-08-06 16-02-32.png)

#### Next step

rerun preprocess (now) and then train with new data

look into issue with theta - we get agreement but I expected these to be theta = 90...



### Time res matching

1. Found theta error - need to make sure I give theta/phi in radians in ddsim

Matching:



<img src="/home/rowan/Downloads/real_vs_learned_first_times_run_7_e6.jpeg" style="zoom:25%;" />

<img src="/home/rowan/.config/Typora/typora-user-images/image-20240807153910268.png" alt="image-20240807153910268" style="zoom:50%;" />

Seems to match!

#### What changed to fix this

1. Before, was seeing this for 1 flow, 128 hu, 20 hl, train frac 0.08 15 epochs
   1. ![](/home/rowan/Downloads/real_vs_learned_first_times(2).jpeg)
   2. ![](/home/rowan/Pictures/Screenshots/Screenshot from 2024-08-07 15-31-22.png)
2. Ran very large model to improve the matching:
   1. aug 7 run 7
   2. 8 flows, 256 hu, 26 hl, 6 epochs, train_frac = 0.16
   3. Seems that more data probably helps more than more epochs
   4. Also may be necessary to have larger model to catch the fine details in the first, second, etc photons

## Energy reconstruction

1. Running simulation
   1. 10k neutron events 0.8GeV to 10GeV, theta = 90
2. Train classifier network on data
   1. Input:
      1. time of each layer, # photons per layer - 56 inputs
   2. Output
      1. Energy

# Thursday August 8

1. TIming resolution is now matching

<img src="/home/rowan/.config/Typora/typora-user-images/image-20240808195452769.png" alt="image-20240808195452769" style="zoom:25%;" />

1. Started on energy reconstruction, but not great so far
2. <img src="/home/rowan/.config/Typora/typora-user-images/image-20240808195533309.png" alt="image-20240808195533309" style="zoom: 25%;" />

## Faster timing

Working on checking if we can improve timing by using a better scintillator

1. In EIC KLM now, we have a time constant (decay constant) of 2.8ns which is slow
   1. Can decrease to 1.8ns to emulate fast counting scintillator
2. Can also try to increase the width of the scintillator to get more photons



Each of these changes will need a new NF model, but we can first just calculate timing with actual optph simulation



