#define check_position_cxx
#include "check_position.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>

void check_position::Loop()
{
    if (fChain == 0) return;
    //    TFile hfile("layers_energy_1GeV_mu_100000.root","RECREATE","file with tree for muon energy by layer");
    
    
    /*DEBUG*/
    Float_t max = -999999;
    Float_t min = 9999999;
    /*ENDDEBUG*/
    
    int num_layers = 28;
    int cont = 0;
    Float_t layer_map[29] = {1820, 1830.8000, 1841.4000, 1907.5, 1918.1,1984.1999, 1994.8000, 2060.8999,2071.5,2137.6001,2148.1999,2214.3000,2224.8999,2291,2301.6001,2367.6999,2378.3000,2444.3999,2455,2521.1001,2531.6999,2597.8000,2608.3999,2674.5,2685.1001,2751.1999,2761.8000,2827.8999,2838.5}; //x value of each scintillator layer - use to assign hits to layer
    Float_t layer_map_28[28] = {1830.8000, 1841.4000, 1907.5, 1918.1,1984.1999, 1994.8000, 2060.8999,2071.5,2137.6001,2148.1999,2214.3000,2224.8999,2291,2301.6001,2367.6999,2378.3000,2444.3999,2455,2521.1001,2531.6999,2597.8000,2608.3999,2674.5,2685.1001,2751.1999,2761.8000,2827.8999,2838.5}; //x value of each scintillator layer - use to assign hits to layer
    Float_t energy_arr[28] = {}; //Set equal to {} to ensure initialize to 0 in each entry
    Float_t count_map[28] = {};
    Float_t count_total = 0;
    
    Float_t mc_count_map[28] = {};
    Float_t mc_count_total = 0;

    int count_total_coverage = 0;
    int num_bins = 4080;
    TH1D *x_hist = new TH1D("h1","h1",num_bins, 1820,2840);
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
	x_hist->Fill(curr_x);
	if((curr_x > layer_map[0]) && (curr_x <=layer_map[28]+5)){count_total_coverage++;}
	for(int j = 0; j < num_layers; j++) {
	  Float_t lower = layer_map_28[j] - 5;
	  Float_t upper = layer_map_28[j] + 5;
	  if((curr_x > lower) && (curr_x <= upper)) {
	    count_map[j]++;
	    count_total++;
	    break; //can break after we found the layer
	  }
	}
	if(curr_layer == -1){continue;}

	curr_layer = -1;
      }//END OF HCALBARRELHITS LOOP
      curr_event++;
   }//END OF EVENT LOOP

   //average out the sums
   Float_t avg_count[28];
   Float_t avg_count_total = count_total / nentries;
   Float_t avg_total_coverage = count_total_coverage / nentries;
   
   for(int i = 0; i < num_layers; i++) {
     avg_count[i] = count_map[i] / nentries; //HcalBarrelHits is the number of hits given by the root tree
   }
   
   TCanvas *c1 = new TCanvas("c1","c1",1250,500);
   c1->Divide(2,1);
   c1->cd(1);
   
   TGraph *gr = new TGraph(num_layers,layer_map_28, avg_count);


   std::string text = "avg # of hits in each layer 100 events (5GeV mu- gun) | total: ";
   text += std::to_string(avg_count_total);
   cout << "text: " << text << "\n";

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
   x_hist->Draw("hist");
   
   c1->Update();
   c1->Draw();
   //Graph stuff
   //TFile f("root_files/graphs/May_7/mu_1GeV_energy_100_run1.root","recreate");
   //gr->Write();
   c1->Print("plots/May_9/TEST_mu_5GeV_energy_100_run1.svg");
   cout << "There were a total of " << avg_count_total << " optical photon hits\n";
   cout << "Average total counts within coverage: " << avg_total_coverage << "\n";
   //x_hist->Print("all");
}
