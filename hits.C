#define hits_cxx
#include "hits.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>

void hits::Loop()
{
   if (fChain == 0) return;
   /*
   //Disable all branches
   fChain->SetBranchStatus("*",0);

   //Activate only those we need for efficiency
   fChain->SetBranchStatus("b_HcalBarrelHits_position_x",1);
   fChain->SetBranchStatus("b_HcalBarrelHits_position_y",1);
   fChain->SetBranchStatus("b_HcalBarrelHits_position_z",1);
   */
   
   
   std::vector<double> vx;
   std::vector<double> vE;
   
   Long64_t nentries = fChain->GetEntriesFast();

   Long64_t nbytes = 0, nb = 0;
   for (Long64_t jentry=0; jentry<nentries;jentry++) {
      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      nb = fChain->GetEntry(jentry);   nbytes += nb;

      for(int i = 0; i < HcalBarrelHits_; i++) {
	vx.push_back(HcalBarrelHits_position_x[i]);//WORKING
	vE.push_back(HcalBarrelHits_energy[i]);
      }
   }
   cout << "vx size " << vx.size() << "\n";
   int length = vx.size();
   /*double x[length];
   cout <<"sizeof(x): "<< sizeof(x) << "\n";
   x[0] = 1;
   cout << "x[0]: " << x[0] << "\n";
   double E[vE.size()];
   cout << "length: " << length << "\n";
   for(int i = 0; i < length; i++) {
     x[i] = vx[i];
     E[i] = vE[i];
     cout << "x[" << i << "]: " << x[i] << "\n";
     }*/
   TGraph *gr = new TGraph(length,vx,vE);
   
   TCanvas *c1 = new TCanvas();
   gr->Draw();
}
