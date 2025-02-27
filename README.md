# work_eic
### Rowan Kelleher

This codebase was created to run ddsim simulations with the epic-klm geometry, and then process and analyze the resulting data.

## Note
Take hepmc files from
/work/eic2/EPIC/EVGEN/SIDIS/pythia6/ep_18x275/hepmc_ip6/

## Workflows
### Basic Simulation

At the center of this repo is the eic-klm simulation. This can be run with the full detector geometry or just a single scintillator bar setup. There is a full optical photon simulation setup and a faster simulation where optical photons are not generated.

#### Running the simulation

1. First ensure you have checked out the correct branch and built that geometry in the eic shell	
   1. First checkout the branch you want with `git checkout ____`
   2. Then enter eic shell: `eic-shell`
   3. Now build the geometry:
      1. See the cmake_command.txt file for the command
   4. Finally, make sure you have sourced the geometry: `source install/setup.sh`
      1. If you are running the simulation through slurm, this is unnecessary
2. Now pick a steering file
   1. Steering files are stored in `work_eic/steering`
      1. Some steering files turn the optical photon simulation on - only use this if the geometry branch you picked has the sensor set to sensitive
      2. Others do not - use the scintillator as sensitive for these
3. Now write your output file name, particle, num events, etc
4. Run the simulation by running the simulation shell script in the eic-shell

### Timing Resolution

A crucial aspect of the simulation is producing timing resolution projections for the KLM. This is the spread of the first photon arrival times for the optical photons generated by a particle passing through the scintillator. If a particle passes through the bar at time t, the photons will arrive on the sensor at some times t_i after t. The first time would be t_0, and the standard deviation of the distribution of this time t_0 across different events provides the timing resolution

#### Steps to calculate Timing Resolution w/optical photon simulation

1. Run simulations with optical photons on
   1. For timing resolution, we want to shoot from the same place and hit the same place every time
      1. If anything changes with the way that the muon hits the scintillator, then this will alter the timings.
      2. We are interested in the distribution of timings for the same event, so we need everything to remain the same
   2. Hence, we want to use a 1 bar setup where we know exactly where the muon hit and when - fix the angle, and position
   3. Probably makes sense to shoot from a high x, 1769.3 seems good
   4. Need high statistics
2. Use `macros/time_res/time_resolution.ipynb` (or `macros/Timing_estimation/time_resolution.ipynb`)
   1. This notebook takes advantage of the process_times function from `time_res_uti.py`
      1. Here we load the root file and process the data
      2. We start by looping over events and eliminating events that don't pass the threshold
      3. Then we save the minimum time of each event, and the distribution of the first times gives us the resolution.
         1. Find sigma of distribution - that is the resolution
   2. Using the output of `process_time`, we can find the mean and sigma of these distributions using `scipy.stats.norm.fit()`

#### Steps to calculate Timing Resolution w/NF sampling

1. Run simulation with scintillator sensitive
   1. Make sure not to generate optical photons
2. Now we need to utilize the NF sampling model
3. First we need to process the data so that the it can be input into the NF model
   1. TODO: finish

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
      1. **flows: number of distinct flows to append together**
      1. **hidden units: number of nodes in hidden layers**
      1. **hidden layers: number of hidden layers in model**
      1. **Batch size: how many samples to include per batch**
      1. **context size: how many conditional variables to train with**
          1. We started with 4: hit z pos, hit time, theta, momentum
	 1. Now we are trying to use 3: eliminated hit time and substracted this time from the photon sensor time
	 1. This way we can learn the time between hit and sensor, then add the time of the hit to get the sensor time.
	 1. There were issues before where the training data hit times were always very small, but the evaluation dataset had very high values, and the model didn't handle this well.


## Notes on files / directories
1. cmake_command.txt
   1. copy and paste to build epic_klm and source
1. base
   1. Old macros written in c++
1. macros
   1. directory with more macro/analysis directories within
   1. Timing_estimation
      1. Main directory for timing parameterization
1. notes
   1. Notes folder to keep track of work notes
1. fieldmaps
   1. Has script to convert fieldmap to correct format for klm usage
1. ddsim_shells
   1. scripts to run ddsim locally
1. steering
   1. Steering files that ddsim runs
1. root_files
   1. root files output by ddsim
1. slurm
   1. slurm directory for running ddsim jobs on dcc batch system
1. OutputFiles
   1. directory to put timing information files after running timing parameterization
1. reco_slurm
   1. slurm directory for running both ddsim simulations and then timing parameterization after


Timing Parameterization file system
1. sampling_slurm
   1. runs sample.py on batch system - same as inference_data
1. sample.py
   1. loads in pre-trained NF and performs inference on real data
1. inference_Data.ipynb
   1. same as sample.py but notebook
1. inference.ipynb
   1. runs inference on test dataset saved before running training (one_segment)
1. train.py
   1. same as conditional_flow.ipynb but python script
1. conditional_flow.ipynb - 