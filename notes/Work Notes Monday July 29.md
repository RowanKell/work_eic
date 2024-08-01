# Work Notes Monday July 29

Overview

1. This week: working on timing resolution projections and ensuring that photon yield matches real data



### Timing resolution

1. Already have this for one bar in macros/time_res directory
   1. Find cellID of hit, count # unique cells
   2. save shortest time for each event that has at least 2 cells hit
   3. find mean and std of the distribution of shortest times
2. Plan with full sim/learned timing
   1. assume each photon is unique pixel
   2. set threshold - between 2->10 pixels
   3. take first time of all events that pass threshold
   4. find std/mean of this distribution