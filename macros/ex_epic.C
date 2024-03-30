#define ex_epic_cxx
#include "ex_epic.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>

void ex_epic::Loop()
{
  Long64_t pdg_count = 0;
  Long64_t pdg_sum = 0;
  int num_pi = 0;
  int num_pi0 = 0;
  int num_g = 0;
  
  TCanvas *c1 = new TCanvas("c1","c1", 1200,1200);
  //c1->Divide(2,2);
  TH1I* pdg_hist = new TH1I("pdg_hist", "MCParticle PDG Values", 600, -300, 300);
  TH1I* s_hist = new TH1I("s_hist", "MCParticle Simulator status Values", 100, -10, 10);
  TH1I* g_hist = new TH1I("g_hist", "MCParticle generator status Values", 100, -10, 10);
  TH1D* h_hist = new TH1D("h_hist","hits",100,-3000,3000);
  long nhits = 0;
  double hit_val = 0;
   Long64_t nentries = fChain->GetEntriesFast();

   Long64_t nbytes = 0, nb = 0;
   for (Long64_t jentry=0; jentry<nentries;jentry++) {
      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      nb = fChain->GetEntry(jentry);   nbytes += nb;
      for(int i = 0; i < HcalBarrelHits_; i++) {
	h_hist->Fill(HcalBarrelHits_position_x[i]);
	hit_val = hit_val + HcalBarrelHits_position_x[i];
	nhits++;
      }
      for(int i = 0; i < MCParticles_; i++) {
	if( MCParticles_PDG[i] < 10000) {
	  
	  int curr_g_status = MCParticles_generatorStatus[i];
	  int curr_s_status = MCParticles_simulatorStatus[i];
	  
	  int curr_pid = MCParticles_PDG[i];
	  pdg_hist->Fill(MCParticles_PDG[i]);

	  s_hist->Fill(curr_s_status);
	  g_hist->Fill(curr_g_status);

	  pdg_sum = pdg_sum + MCParticles_PDG[i];
	  pdg_count++;
	  //cout << "PDG: " << MCParticles_PDG[i] << "\n";
	  if((curr_pid == 211) || (curr_pid == -211)){
	    num_pi++;
	  }
	  if(curr_pid == 111) {
	    num_pi0++;
	  }
	  if(curr_pid == 22) {
	    num_g++;
	  }
	}
      }
   }
   cout << "Average pdg: " << pdg_sum / pdg_count << "\n";
   cout << "#charged pi: " << num_pi << " | #pi0: " << num_pi0 << " | #gammas: " << num_g << "\n";
   cout << "pdg count: " << pdg_count << "\n";
   cout << "nhits: " << nhits << "\n";
   cout << "avg x: " << hit_val / nhits << "\n";
   //c1->cd(1);
   pdg_hist->Draw();
   /*c1->cd(2);
   s_hist->Draw();
   c1->cd(3);
   g_hist->Draw();
   c1->cd(4);*/
   //h_hist->Draw();
   c1->Print("plots/ex_steering_PDG_epic_short_steering_pi.jpeg");
   pdg_hist->Print("range");

}
