#define Pi_optical_1GeV_cxx
#include "Pi_optical_1GeV.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>

void Pi_optical_1GeV::Loop()
{
    if (fChain == 0) return;
    TFile hfile("layers_energy_1GeV_mu_100000.root","RECREATE","file with tree for muon energy by layer");
    
    
    /*DEBUG*/
    Float_t max = -999999;
    Float_t min = 9999999;
    /*ENDDEBUG*/
    
    int num_layers = 28;
    int cont = 0;
    Float_t layer_map[29] = {1820, 1830.8000, 1841.4000, 1907.5, 1918.1,1984.1999, 1994.8000, 2060.8999,2071.5,2137.6001,2148.1999,2214.3000,2224.8999,2291,2301.6001,2367.6999,2378.3000,2444.3999,2455,2521.1001,2531.6999,2597.8000,2608.3999,2674.5,2685.1001,2751.1999,2761.8000,2827.8999,2838.5}; //x value of each scintillator layer - use to assign hits to layer
    Float_t energy_arr[28] = {}; //Set equal to {} to ensure initialize to 0 in each entry
    Float_t count_map[28] = {};
    Float_t count_total = 0;
    
    Float_t mc_count_map[28] = {};
    Float_t mc_count_total = 0;
    
    long curr_layer; //use for each hit to find which layer the hit hit
    
   long nhits = 0;
   long curr_event = 0; //keep track of current event
   Long64_t nentries = fChain->GetEntriesFast();
   Long64_t nbytes = 0, nb = 0;
   
   for (Long64_t jentry=0; jentry<nentries;jentry++) {
     Float_t histo_energy_arr[28] = {};
      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      nb = fChain->GetEntry(jentry);   nbytes += nb;
      for(int i = 0; i < HcalBarrelHits_; i++) {
	Float_t curr_x = HcalBarrelHits_position_x[i];
	for(int j = 0; j < num_layers; j++) {
	  //need to find the layer of this particular hit
	  if((curr_x > layer_map[j]) && (curr_x <= layer_map[j+1])) {
	    count_map[j]++;
	    count_total++;
	    break; //can break after we found the layer
	  }
	}
	if(curr_layer == -1){continue;}

	curr_layer = -1;
      }//END OF HCALBARRELHITS LOOP
      //START OF MCPARTICLE LOOP
      for(int i = 0; i < MCParticles_; i++) {
	if(MCParticles_PDG[i] != -22){continue;}
	mc_count_total++;
	Float_t curr_x = MCParticles_vertex_x[i];
	for(int j = 0; j < num_layers; j++) {
	  //need to find the layer of this particular hit
	  if((curr_x > layer_map[j]) && (curr_x <= layer_map[j+1])) {
	    mc_count_map[j]++;
	    break; //can break after we found the layer
	  }
	}
	if(curr_layer == -1){continue;}

	curr_layer = -1;
      }
      
      curr_event++;
   }//END OF EVENT LOOP
   
   //average out the sums
   Float_t avg_count[28];
   Float_t mc_avg_count[28];

   Float_t avg_count_total = count_total / nentries;
   Float_t mc_avg_count_total = mc_count_total / nentries;
   
   for(int i = 0; i < num_layers; i++) {
     avg_count[i] = count_map[i] / nentries; //HcalBarrelHits is the number of hits given by the root tree
     mc_avg_count[i] = mc_count_map[i] / nentries; //HcalBarrelHits is the number of hits given by the root tree
   }
   TCanvas *c1 = new TCanvas("c1","c1",1250,500);
   c1->Divide(2,1);
   c1->cd(1);
   
   TGraph *gr = new TGraph(num_layers,layer_map, avg_count);
   TGraph *mc_gr = new TGraph(num_layers,layer_map, mc_avg_count);


   std::string text = "avg # of hits in each layer 100 events (1GeV pi- gun) | total: ";
   text += std::to_string(avg_count_total);
   cout << "text: " << text << "\n";
   
   std::string mc_text = "avg # of -22 PDG in each layer 100 events (1GeV pi- gun) | total: ";
   mc_text += std::to_string(mc_avg_count_total);
   cout << "mc_text: " << mc_text << "\n";
   
   gr->SetTitle(text.c_str());
   auto xaxis = gr->GetXaxis();
   xaxis->SetTitle("Barrel hit x position (mm)");
   auto yaxis = gr->GetYaxis();
   yaxis->SetTitle("# of hits per event");
   
   xaxis->SetLimits(1600,3000);
   gr->SetMarkerStyle(20);
   gr->SetMarkerSize(0.6);
   gr->SetMarkerColor(4);
   gr->Draw("AP");
   
   c1->cd(2);
   mc_gr->SetTitle(mc_text.c_str());
   auto mc_xaxis = mc_gr->GetXaxis();
   mc_xaxis->SetTitle("vertex x position (mm)");
   auto mc_yaxis = mc_gr->GetYaxis();
   mc_yaxis->SetTitle("# of optical photons in MCParticles per event");
   
   mc_xaxis->SetLimits(1600,3000);
   mc_gr->SetMarkerStyle(20);
   mc_gr->SetMarkerSize(0.6);
   mc_gr->SetMarkerColor(2);
   mc_gr->Draw("AP");

   c1->Update();
   c1->Draw();

   c1->Print("plots/May_7/pi_1GeV_energy_100_run1.svg");
   cout << "There were a total of " << avg_count_total << " optical photon hits\n";
   cout << "There were a total of " << mc_avg_count_total << " -22 PDG MCParticles\n";
}
