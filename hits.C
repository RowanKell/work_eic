#define epic_klm_base_cxx
#include "epic_klm_base.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>

void epic_klm_base::Loop()
{
   if (fChain == 0) return;

   //Disable all branches
   fChain->SetBranchStatus("*",0);

   //Activate only those we need for efficiency
   fChain->SetBranchStatus("b_HcalBarrelHits_position_x",1);
   fChain->SetBranchStatus("b_HcalBarrelHits_position_y",1);
   fChain->SetBranchStatus("b_HcalBarrelHits_position_z",1);

   double hits_x;
   double hits_y;
   double hits_z;
   
   Long64_t nentries = fChain->GetEntriesFast();

   Long64_t nbytes = 0, nb = 0;
   for (Long64_t jentry=0; jentry<nentries;jentry++) {
      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      nb = fChain->GetEntry(jentry);   nbytes += nb;

      
      
   }
}
