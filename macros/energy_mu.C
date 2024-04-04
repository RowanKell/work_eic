#define energy_mu_cxx
#include "energy_mu.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>

void energy_mu::Loop()
{
    if (fChain == 0) return;
   
    int num_layers = 28;
    Float_t layer_map[28] = {1830.8000, 1841.4000, 1907.5, 1918.0999,1984.1999, 1994.8000, 2060.8999,2071.5,2137.6001,2148.1999,2214.3000,2224.8999,2291,2301.6001,2367.6999,2378.3000,2444.3999,2455,2521.1001,2531.6999,2597.8000,2608.3999,2674.5,2685.1001,2751.1999,2761.8000,2827.8999,2838.5}; //x value of each scintillator layer - use to assign hits to layer
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
      cout << "HcalBarrelHits_: " << HcalBarrelHits_ << "\n";
      cout << "sizeof HcalBarrelHits_position_x: " << sizeof(HcalBarrelHits_position_x) / sizeof(*HcalBarrelHits_position_x) << "\n";
      for(int i = 0; i < HcalBarrelHits_; i++) {
	//cout << "Entry: " << i << "| position: " << HcalBarrelHits_position_x[i] << "| energy: " << HcalBarrelHits_energy[i] <<"\n";
	Float_t curr_x = HcalBarrelHits_position_x[i];
	for(int j = 0; j < num_layers; j++) {
	  //need to find the layer of this particular hit
	  if(layer_map[j] == curr_x) {
	    curr_layer = j;
	    break; //can break after we found the layer
	  }
	}
	/*
	cout << "curr_layer: " << curr_layer << "\n";
	if(curr_layer == -1){continue;} //skip any hits that don't correspond to a layer (assumes that hits are always at the same position)
	if(HcalBarrelHits_energy[i] < 10 && HcalBarrelHits_energy[i] > 0 ){
	  energy_arr[curr_layer] += HcalBarrelHits_energy[i]; //sum up all hits that correspond to 1 layer
	}
	*/
	//cout << "HcalBarrelHit: " << HcalBarrelHits_energy[i] << "\n";
	
	curr_layer = -1; //used to check if the hit corresponds to a layer or not. May want to bin instead
      }
      curr_event++;
      break;
   }
   /*
   //average out the sums
   Float_t avg_energy[28];
   for(int i = 0; i < num_layers; i++) {
     avg_energy[i] = energy_arr[i] / HcalBarrelHits_; //HcalBarrelHits is the number of hits given by the root tree
   }
   
   TCanvas *c1 = new TCanvas();
   TGraph *gr = new TGraph(num_layers, layer_map, energy_arr);
   
   gr->SetTitle("Average HcalBarrelHits in each layer 1000 events (pi- gun)");
   auto xaxis = gr->GetXaxis();
   xaxis->SetTitle("Barrel hit x position (mm)");
   auto yaxis = gr->GetYaxis();
   yaxis->SetTitle("Energy deposited per event (GeV)");
   
   xaxis->SetLimits(1600,3000);
   gr->SetMarkerStyle(34);
   gr->SetMarkerSize(0.5);
   gr->SetMarkerColor(2);
   gr->Draw("AP");
   
   //TFile f("root_files/graphs/pi_100_april_4.root","recreate");
   gr->Write();
   //c1->Print("plots/april_4/pi_100.pdf");
   
   cout << "working\n";*/
   //DEBUG INFO
   
   /*cout << "Energy at each layer:\n";
   for(int i = 0; i < num_layers; i++) {
     if(i + 1 == num_layers) {cout << avg_energy[i] << "\n";}
     cout << energy_arr[i] << " | ";
   }
   cout << "Total energy at each layer:\n";
   for(int i = 0; i < num_layers; i++) {
     if(i + 1 == num_layers) {cout << energy_arr[i] << "\n";}
     cout << energy_arr[i] << " | ";
     }*/
}
