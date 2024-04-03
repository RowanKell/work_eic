#define pi_momentum_cxx
#include "pi_momentum.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>

void pi_momentum::Loop()
{

   if (fChain == 0) return;

   Long64_t nentries = fChain->GetEntriesFast();

   Long64_t nbytes = 0, nb = 0;
   for (Long64_t jentry=0; jentry<nentries;jentry++) {
     double sum = 0;
      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      nb = fChain->GetEntry(jentry);   nbytes += nb;
      cout << "MCParticles_: " << MCParticles_ << "\n";
      for(int i = 0;i < MCParticles_; i++ ){
	//int daughter_check = 0;
	//if ((MCParticles_daughters_end[i] - MCParticles_daughters_begin[i]) == 0){ daughter_check = 1;}
	int vertex_check = MCParticles_simulatorStatus[i] & ( 0x1 << 28 );
	if((MCParticles_generatorStatus[i] != 1) && vertex_check){
	  sum += MCParticles_momentum_x[i];
	  //sum += sqrt(pow(MCParticles_momentum_z[i],2) + pow(MCParticles_momentum_y[i],2) + pow(MCParticles_momentum_x[i],2));
	}
      }
      
      cout << "total x momentum: " <<  sum << "\n";
      // if (Cut(ientry) < 0) continue;
   }
}
