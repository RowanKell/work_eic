Folder for macros and such

mu_*GeV.C plots the energy deposited into each layer (by muon and secondaries) during event. 10000 refers to having 10000 events. pi_energy... is similar but wiht a pion gun.

NOTE: you cannot just change the root file being used in the .h header for one of these macros. The arrays defined in the .h file are specific to that file. If you do not update the maximum array values you may encounter buffer overflow as the tree will not be able to fit every entry into the array. Best practice is to run "root <rootfilename>" and then "events->MakeClass("InsertNameOfMacro"). Now you have a macro with a header file "InstertNameOfMacro.h" with the correct length arrays!

old files are old