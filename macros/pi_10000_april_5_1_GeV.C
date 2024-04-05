#define pi_10000_april_5_1_GeV_cxx
#include "pi_10000_april_5_1_GeV.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>

void pi_10000_april_5_1_GeV::Loop()
{
    if (fChain == 0) return;
   
    int num_layers = 28;
    Float_t layer_map[28] = {1830.8000, 1841.4000, 1907.5, 1918.1,1984.1999, 1994.8000, 2060.8999,2071.5,2137.6001,2148.1999,2214.3000,2224.8999,2291,2301.6001,2367.6999,2378.3000,2444.3999,2455,2521.1001,2531.6999,2597.8000,2608.3999,2674.5,2685.1001,2751.1999,2761.8000,2827.8999,2838.5}; //x value of each scintillator layer - use to assign hits to layer
    Float_t energy_arr[28] = {}; //Set equal to {} to ensure initialize to 0 in each entry

    long curr_layer; //use for each hit to find which layer the hit hit
    
   long nhits = 0;
   long curr_event = 0; //keep track of current event
   Long64_t nentries = fChain->GetEntriesFast();
   Long64_t nbytes = 0, nb = 0;
   
   for (Long64_t jentry=0; jentry<nentries;jentry++) {
      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      nb = fChain->GetEntry(jentry);   nbytes += nb;
      for(int i = 0; i < HcalBarrelHits_; i++) {
	Float_t curr_x = HcalBarrelHits_position_x[i];
	for(int j = 0; j < num_layers; j++) {
	  //need to find the layer of this particular hit
	  if(layer_map[j] == curr_x) {
	    curr_layer = j;
	    break; //can break after we found the layer
	  }
	}
	
	if(curr_layer == -1){continue;} //skip any hits that don't correspond to a layer (assumes that hits are always at the same position)
	energy_arr[curr_layer] += HcalBarrelHits_energy[i]; //sum up all hits that correspond to 1 layer
	
	curr_layer = -1; //used to check if the hit corresponds to a layer or not. May want to bin instead
      }
      curr_event++;
   }
   
   //average out the sums
   Float_t avg_energy[28];
   for(int i = 0; i < num_layers; i++) {
     avg_energy[i] = energy_arr[i] / nentries; //HcalBarrelHits is the number of hits given by the root tree
   }

   TCanvas *c1 = new TCanvas("c1", "c1",955,600);
   c1->SetLeftMargin(0.1);
   c1->SetRightMargin(0.04);
   
   TGraph *gr = new TGraph(num_layers, layer_map, avg_energy);
   
   gr->SetTitle("Avg energy dept in each layer 10000 events (1 GeV pi- gun)");
   auto xaxis = gr->GetXaxis();
   xaxis->SetTitle("Barrel hit x position (mm)");
   auto yaxis = gr->GetYaxis();
   yaxis->SetTitle("Energy deposited per event (GeV)");
   
   xaxis->SetLimits(1600,3000);
   gr->SetMarkerStyle(22); //34 works
   gr->SetMarkerSize(0.75);
   gr->SetMarkerColor(4); //4 is blue, 2 is red
   gr->Draw("AP");
   
   TFile f("root_files/graphs/pi_10000_april_5_1_GeV.root","recreate");
   gr->Write();
   c1->Print("plots/april_4/pi_10000_1_GeV.svg");
}
