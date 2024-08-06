Folder for macros and such

Timing_estimation:
1. Used for using NF to learn timing distribution then applying this to scint sensitive sims
1. Also trying to do energy reconstruction there...

Basic_NF:
1. Has an old attempt at using photon counts to do PID (muon pion separation)
1. Can use that code to build classifier network elsewhere

time_res:
1. Older method of calculating timing resolution using 1 bar and optical photons

Variation:
1. used for figuring out dependence of photon yield on z hit position

mu_*GeV.C plots the energy deposited into each layer (by muon and secondaries) during event. 10000 refers to having 10000 events. pi_energy... is similar but wiht a pion gun.

NOTE: you cannot just change the root file being used in the .h header for one of these macros. The arrays defined in the .h file are specific to that file. If you do not update the maximum array values you may encounter buffer overflow as the tree will not be able to fit every entry into the array. Best practice is to run "root <rootfilename>" and then "events->MakeClass("InsertNameOfMacro"). Now you have a macro with a header file "InstertNameOfMacro.h" with the correct length arrays!

old files are old