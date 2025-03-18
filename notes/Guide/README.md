# Notes/Plots for Greg

Note: all these plots are in the github, and most are available as jpeg or pdf. I can provide pdfs of the mobo optimization stuff if needed, but I also provided the html from which I took screenshots.

## Timing parameterization

1. I think what I sent before should be enough unless you want more

## GNN for Energy Prediction

Notes

1. **Overview:**
   1. The information we get from the detector comes solely from the SiPM: **integrated charge and threshold time**
   2. a primary neutral hadron (K_L or n) is shot into the barrel
   3. the hadron will interact with the steel in the KLM to create charged secondaries
   4. Charged secondaries deposit energy in the scintillator, creating scintillation light
   5. The optical photons (scintillation light) travels through the scintillator and some of them hit the SiPM on the end (both ends)
   6. The SiPM generates a waveform from the optical photons (each optical photon generates a pulse, and these pulses are summed)
   7. The waveform is integrated to get the integrated charge, and constant fraction discrimination is used to get the time when the pulse exceeded 0.3 of its maximum amplitude
2. **KLM structure:**
   1. The KLM is broken into 8 staves that form a barrel
   2.  each stave consists of many layers
   3.  each layer consists of a sheet of steel and many scintillator strips (each of the same width)
   4. Each scintillator has an SiPM on either end
   5. The detector output consists of a dataframe where each row corresponds to a different scintillator strip
      1. each row of the dataframe contains the information about the position of the strip (xyz) and the strip ID values (stave id, layer id, strip id)
      2. Each row also contains the integrated charge and timing information for both SiPMs corresponding to that strip
3. **Data production pipeline**
   1. Detector simulation
      1. DDSIM simulates a K_L gun, shooting at a random phi and theta angle (constrained to hit within the barrel) and a random momentum between 1-3 GeV
         1. DDSIM outputs a edm4hep.root file which contains
            1. the MCParticle bank with truth information about the K_L and secondaries
            2. the HcalBarrelHits bank which contains information about energy deposited in the scintillator by charged secondaries
   2. Pre-processing
      1. I extract the energy deposition information from the HcalBarrelHits bank and sum up all the energy deposited in each strip for each particle
      2. I calculate the number of photons produced by this particle, and calculate the number of photons we expect to reach the SiPM given the position of the incident charged particle
      3. Then, I sum up the number of photons within a single strip across all particle
   3. Timing estimation
      1. I utilize a trained normalizing flow (NF) model to generate optical photon arrival times for each photon calculated in the pre-processing step
         1. The NF input is: charged secondary angle, hit position, and momentum
         2. Output: photon arrival time at SiPM
   4. SiPM modeling
      1. A simple SiPM model is used to create a waveform from photon arrival times
      2. the area under the wave form is reported as the integrated charge
      3. the time that the waveform reaches 30% of its peak is the reported timing
      4. For each SiPM, a pixel threshold of 2 pixels must be met
         1. if this threshold is not met, the SiPM is treated as producing no output
            1. If only one SIPM has output, then the other is padded with 0s in the output dataframe
         2. If a scintillator bar has no output from either SiPM, the bar is ignored altogether
4. **Energy prediction**
   1. <u>Dataset</u>
      1. Using the dataframes, we construct a dataset of graphs
      2. each event is represented by a single graph
      3. each scintillator which contains SiPM output is represented by a node within the graph
      4. each node has several features
         1. SiPM charge and timing
         2. scintillator strip position
         3. radial distance of strip from interaction point
      5. each node is connected to its 6 nearest nodes, using the distance between the scintillator strips as the distance between nodes
      6. Each graph has several graph level (event level) features
         1. Total charge (the sum of the charge for each SiPM across all SiPMs in the graph)
         2. Max charge (the maximum of the charge across all SiPMs in the graph)
         3. Number of hits (the number of nodes in the graph)
   2. <u>GNN Architecture</u>
      1. We implement a GNN using the Deep Graphs Library in python + pytorch
      2. The GNN consists of 2 graph convolutional layers
         1. The graph convolutional layers use sum aggregation and MLPs as described here https://arxiv.org/pdf/1810.00826
            1. implementation: https://www.dgl.ai/dgl_docs/generated/dgl.nn.pytorch.conv.GINConv.html
      3. An average pooling layer condenses the graph into a 1D tensor of features
      4. Several dense layers are applied to produce a single output node which corresponds to the predicted neutral hadron energy
         1. The number of dense layers and the hidden dimension of each layer is a chooseable hyperparameter
         2. Currently, we use 7 linear layers, which have hidden dimension starting at 256 and shrinking throughout the network down to 1
         3. ReLU activation is used between the graph convolutional layers, as well as between the linear layers (but not on the last layer)
   3. <u>Training</u>
      1. Data is split into a 70% train, 15% validation, 15% test split
      2. Mini batching is used with a batch size of 20rck32@duke.edu
      3. Mean squared error loss is used to train the model
      4. the adam optimizer is used



## Results and Plots

**True vs predicted energy for K_L:**

1. 10k events (split into train, test val)
2. K_L between 1-3GeV

![](/home/rowan/Downloads/March_17_predsvtruth.jpeg)

**Binned RMSE and relative RMSE:**

![](/home/rowan/Downloads/March_17_run_1_RMSE_k_6.jpeg)

**Example graph in dataset**

1. This plot shows one example of an event represented by a graph
2. The black octagons show the inside and outside borders of the klm
3. the right hand plot shows the graph up close
4. the left hand plot shows the graph in the context of the KLM
5. the orange lines show the cone with which we cluster hits together
   1. hits outside of this cone are treated as background and not included in the graph

![](/home/rowan/Downloads/March_17_graph_viz.jpeg)

**MOBO plots**

1. These plots are all 1-d slices\
2. the y axis shows an objective
3. the x axis shows a parameter
4. For RMSE, lower is better
5. For mu/pi separation, higher is better
6. **I believe that steel ratio here means what ratio of the layer is scintillator, not steel**
   1. Gonna run some new experiments with this fixed/swapped, and hopefully get better results



RMSE: K_L energy prediction root mean squared error

![image-20250318121255329](/home/rowan/.config/Typora/typora-user-images/image-20250318121255329.png)

mu/pi separation - roc curve area under the curve for correctly identifying a particle as either a muon or a pion

1. Ian wrote this code - it calculates the number of photons we would see in each layer and classifies the particle based on how far it made it through the barrel.
2. There is one objective for 5GeV particles and one for 1GeV, as the optimal design parameters change depending on energy

![image-20250318121358900](/home/rowan/.config/Typora/typora-user-images/image-20250318121358900.png)

![image-20250318121536265](/home/rowan/.config/Typora/typora-user-images/image-20250318121536265.png)