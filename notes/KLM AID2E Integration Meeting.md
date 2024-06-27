# KLM AID2E Integration Meeting

#### June 27, 2024

1. Objectives
   1. FOM
      1. Use AUC of roc_curve as metric
   2. 4 or 5 maximum, 3 would be better
2. Parameters
   1. Overall iron 
   2. Overall scintillator
   3. number of layers
   4. Barrel Length? need to somehow offset this
   5. Inner + outer radius
      1. outer radius as objective - made it as small as possible
3. May want to divide up by different momentum ranges
   1. 2 momentum ranges
   2. 0.5-3 and 3-10
4. Change # layers and ratio of iron to scintillator
5. Variable iron thickness?
6. Once we have shower shape we can do more



Framework code

1. Botorch tutorials are useful
2. parameters.config
   1. JSON for defining each parameter in xml that can be edited
   2. min, max, nominal values, units
   3. Need reasonable values for # of layers (easy) and ratio of iron to scint
3. wrapper.py
   1. loads botorch (bayesian optimization) and axe (experiment management) libraries
   2. Only needs small edits
   3. Suggests a design point
   4. edits xml, creates new geometry
4. runTestsAndObjectiveCalc.py (maybe)
   1. Run simulation jobs
      1. Run 2 jobs per design point (one per momentum range)
   2. Calculates objectives

Plan

1. setting up parameters.config
2. Manually run stuff
3. set lowerbound on objective as 10% lower than expected lower