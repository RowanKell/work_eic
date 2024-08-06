# Work Notes August 2nd

1. Finished moving timing resolution preprocess function to time_res_util.py
   1. Works both with single file and multiple files
   2. getting 130ps resolution for one bar, optical photons on
      1. Can increase by increasing scintillator thickness and/or reducing the decay constant of the material

#### Update

1. Seems that the difference in time resolution between NF method and optph method comes from fact that the distributions don't match at lower times
   1. This means that the first times for the NF method are lower and more spread out than the first times for the optph method, giving a worse resolution
      1. Currently trying to train new models to fix this issue - need better matching of distributions at low t