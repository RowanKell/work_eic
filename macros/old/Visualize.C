#define Visualize_cxx
#include "Visualize.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>

void Visualize::Loop()
{

   if (fChain == 0) return;

   Long64_t nentries = fChain->GetEntriesFast();

   Long64_t nbytes = 0, nb = 0;
   int sum = 0;
   for (Long64_t jentry=0; jentry<nentries;jentry++) {
      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      nb = fChain->GetEntry(jentry);   nbytes += nb;
      cout << "MCParticles_: " << MCParticles_ << "\n";
      for(int i = 0;i < MCParticles_; i++ ){
	if(MCParticles_generatorStatus[i] != 1){
	  sum += MCParticles_momentum_x[i];
	}
      }
      
      cout << "total x momentum: " <<  sum << "\n";
      // if (Cut(ientry) < 0) continue;
   }
}
