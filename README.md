# work_eic
### Rowan Kelleher

This codebase was created to run ddsim simulations with the epic-klm geometry, and then process and analyze the resulting data.

## Workflows
### Conditional NF Timing Parameterization
I am working on a timing parameterization using a conditional normalizing flow network. The process of training and using the parameterization is unfortunately not simple.

#### Steps to use parameterization
1. Create data/run simulation
   1. First one needs to run the simulation to create training data. Data should be created using the `one_segment` branch of the epic-klm geometry repo. We use this geometry as it has a single scintillator bar and allows us to simulate a single charged particle interacting with the scintillator. This lets us run simulations where we know the position, angle, momentum, etc of the incident particle. Most importantly, it allows us to know exactly what particle created which photons and where, even if we do not have the scintillator sensitive.
   1. Run the simulation at a variety of different gun z positions to try to get a range of positions where the photons are created. We want the training data to emulate the actual detector data as much as possible
   1. The output root file should contain timing information (the sensor needs to be sensitive)
   1. Currently I use the makeJobs_full_theta_vary_z.sh script to generate these - this script uses the variation_pos.py steering file and creates many different scripts with different z positions and tunes the theta values to ensure we don't miss the bar.
1. preprocess the data
   1. To use the simulation data to train an NF model, we need to extract the relevant info and put it into torch tensors
   1. I use makeJobs_parallel.sh in the macros/Timing_estimation/data/slurm directory to send many preprocess_fast.py jobs to do this.
   1. preprocess_fast.py reads the simulation root files and outputs torch tensors with the context and data
      1. The script calculates the p magnitude, theta, hit position, and grabs the hit time and photon time. It places these in the torch tensor and saves it to a .pt file
   1. By running many of these jobs in parallel, we get a set of torch tensors that we can then concat together to get our dataset
1. Training
   1. Next, we open conditional_flow.ipynb to load our data and train a model
   1. The data is loaded via a loop and the set is randomized and split into training and test data.
   1. Then we defined the model, using these parameters
      1. # flows: number of distinct flows to append together
      1. # hidden units: number of nodes in hidden layers
      1. # hidden layers: number of hidden layers in model
      1. Batch size: how many samples to include per batch
      1. context size: how many conditional variables to train with
      	 1. We started with 4: hit z pos, hit time, theta, momentum
	 1. Now we are trying to use 3: eliminated hit time and substracted this time from the photon sensor time
	 1. This way we can learn the time between hit and sensor, then add the time of the hit to get the sensor time.
	 1. There were issues before where the training data hit times were always very small, but the evaluation dataset had very high values, and the model didn't handle this well.