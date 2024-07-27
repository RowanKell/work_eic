# Work Notes Thurs July 18

Plan

1. Run sector sim with scint sens
   1. See distribution of theta, hit time, z hit pos, momentum
      1. Need to make sure the training dataset has same coverage

Looking at data distribution of context vars

![image-20240718104930703](/home/rowan/.config/Typora/typora-user-images/image-20240718104930703.png)

1. hit z pos is much more distributed than in training set
   1. Maybe just run 100 jobs at diff z values
2. Need to place cut on mu hit time - can't see anything from this
3. theta doesn't tell much - shot at theta = 90 so everything is there
   1. With new theta, need to focus more on theta = 90
4. momentum
   1. Probably need to extend domain down close to 0 - maybe even 0



more refined look at actual data:

![image-20240718113421818](/home/rowan/.config/Typora/typora-user-images/image-20240718113421818.png)



Plan

1. Run new training sims
   1. 200 jobs at diff z values
   2. use cos(theta) for theta distribution
   3. use momentum range down to 0
2. Don't train model on mu hit time
   1. Just add this after, and subtract from target dataset
      1. Target should be time between hit and sensor

### Percentage

1. Need to investigate why we get 18% of photons in the full sector vs 0.5% in one segment
   1. Maybe first look at total energy dep vs # photons generated to ensure there is no funny business
   2. look at photon vertex z position - are many more just being generated near the end?
2. Idea
   1. Instead of simulating this on single bar, may be better to use full sector setup with scint sensitive, generate optph and save all
   2. Then, look at the z vertex position of the photons and also the z hit positions of their parents?
      1. If they line up, then we know the photons are starting right where the hit is
   3. Then turn sensor sensitive and look at what % of photons make it to the end as a function of z vertex position
      1. Need to relate z hit position of parent to z vertex of photon, then to probability of hitting sensor

### Other

1. Need to double check % that reach sensor for segmented layers, not just full layers
   1. Photons may die off at a high rate from internal reflection
2. using sector_epic_klm for this
3. When using segmented layers, photon yield is much much lower...

### For tomorrow

1. Maybe try some other things with models? need a reasonable loss curve...
2. Try to document the process of creating and processing data as well as training and inference
   1. Too jumbled rn, confusing myself

# Work Notes July 19

1. Loss curve looks fine for july 5th data, maybe issue with cos(theta)?
   1. Run new dataset w/uniform distribution, smaller theta window, then run another set with wide window
      1. This way we get concentrated data on theta close to 90 deg

Looking at June_18 sector scint, we see this hit time distribution

<img src="/home/rowan/.config/Typora/typora-user-images/image-20240719150401222.png" alt="image-20240719150401222" style="zoom: 25%;" />

1. Everything is between 6 and 11 ns, so the photon hit times should certainly be after 6 ns
2. Below is the sensor timings (when photons hit sensor) for 30 events of full detector

<img src="/home/rowan/.config/Typora/typora-user-images/image-20240719150602414.png" alt="image-20240719150602414" style="zoom:25%;" />

1. So we see that photons hit at 10 ns at earliest, meaning it takes at least 4 ns for them to travel
   1. We should see this distribution starting at 4ns for the samples (before adding the mu hit times)
2. Below we have the timings from the one segment simulation
   1. Assuming that the mu hit time is ~0, then this is what we add to the mu hit time (of the full detector) to get the real photon hit time on sensor for the full detector
   2. This is one segment, 0th z position (furthest away) so we get 4 nano seconds minimum of travel time

<img src="/home/rowan/.config/Typora/typora-user-images/image-20240719151152336.png" alt="image-20240719151152336" style="zoom:25%;" />

1. Now we have file 50 - this is right next to the sensor so we get some at 0, etc
2. <img src="/home/rowan/.config/Typora/typora-user-images/image-20240719153041177.png" alt="image-20240719153041177" style="zoom:25%;" />

1. Either need to somehow emphasize the z position better, or need to run full sector simulation with varying z
   1. Best option - probably run more data at higher z pos further from the sensor
      1. This way, we get a better distribution - we need a more realistics z distribution