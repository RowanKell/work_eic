#define mu_1GeV_energy_cxx
#include "mu_1GeV_energy.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>

void mu_1GeV_energy::Loop()

{
    if (fChain == 0) return;
    TFile hfile("layers_energy_1GeV_muon.root","RECREATE","file with tree for muon energy by layer");
    TTree tree("t_layers_energy","tree with energy deposited by 1GeV muon by layer");
    Float_t tree_energy;
    
    TBranch *b_layer_0 = tree->Branch("layer_0",&tree_energy,);
    TBranch *b_layer_1 = tree->Branch("layer_1",&tree_energy,);
    TBranch *b_layer_2 = tree->Branch("layer_2",&tree_energy,);
    TBranch *b_layer_3 = tree->Branch("layer_3",&tree_energy,);
    TBranch *b_layer_4 = tree->Branch("layer_4",&tree_energy,);
    TBranch *b_layer_5 = tree->Branch("layer_5",&tree_energy,);
    TBranch *b_layer_6 = tree->Branch("layer_6",&tree_energy,);
    TBranch *b_layer_7 = tree->Branch("layer_7",&tree_energy,);
    TBranch *b_layer_8 = tree->Branch("layer_8",&tree_energy,);
    TBranch *b_layer_9 = tree->Branch("layer_8",&tree_energy,);
    TBranch *b_layer_10 = tree->Branch("layer_9",&tree_energy,);
    TBranch *b_layer_11 = tree->Branch("layer_10",&tree_energy,);
    TBranch *b_layer_12 = tree->Branch("layer_11",&tree_energy,);
    TBranch *b_layer_13 = tree->Branch("layer_12",&tree_energy,);
    TBranch *b_layer_14 = tree->Branch("layer_13",&tree_energy,);
    TBranch *b_layer_15 = tree->Branch("layer_14",&tree_energy,);
    TBranch *b_layer_16 = tree->Branch("layer_15",&tree_energy,);
    TBranch *b_layer_17 = tree->Branch("layer_16",&tree_energy,);
    TBranch *b_layer_18 = tree->Branch("layer_17",&tree_energy,);
    TBranch *b_layer_19 = tree->Branch("layer_18",&tree_energy,);
    TBranch *b_layer_20 = tree->Branch("layer_19",&tree_energy,);
    TBranch *b_layer_21 = tree->Branch("layer_20",&tree_energy,);
    TBranch *b_layer_22 = tree->Branch("layer_21",&tree_energy,);
    TBranch *b_layer_23 = tree->Branch("layer_22",&tree_energy,);
    TBranch *b_layer_24 = tree->Branch("layer_23",&tree_energy,);
    TBranch *b_layer_25 = tree->Branch("layer_24",&tree_energy,);
    TBranch *b_layer_26 = tree->Branch("layer_25",&tree_energy,);
    TBranch *b_layer_27 tree->Branch("layer_26",&tree_energy,);
    TBranch *b_layer_1 tree->Branch("layer_27",&tree_energy,);

    /*DEBUG*/
    Float_t max = -999999;
    Float_t min = 9999999;
    /*ENDDEBUG*/

    
    int num_layers = 28;
    int cont = 0;
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
	if(curr_layer == -1){continue;}
	//if(curr_layer == -1){cont++; cout << "continuing #" << cont << "\n"; continue;} //skip any hits that don't correspond to a layer (assumes that hits are always at the same position)
	energy_arr[curr_layer] += HcalBarrelHits_energy[i]; //sum up all hits that correspond to 1 layer

	
	/*DEBUG*/
	if(HcalBarrelHits_energy[i] > max) {max = HcalBarrelHits_energy[i];}
	if(HcalBarrelHits_energy[i] < min) {min = HcalBarrelHits_energy[i];}

	/*ENDDEBUG*/

	   
	curr_layer = -1; //used to check if the hit corresponds to a layer or not. May want to bin instead
      }
      curr_event++;
   }
   
   //average out the sums
   Float_t avg_energy[28];
   for(int i = 0; i < num_layers; i++) {
     avg_energy[i] = energy_arr[i] / nentries; //HcalBarrelHits is the number of hits given by the root tree
     //cout << "sum energy in bin #" << i << ": " << energy_arr[i] << "\n";
   }

   // Calculate Standard deviation
   
   Float_t y_err[28] = {};
   Float_t x_err[28] = {};

   Float_t sq_residuals[28] = {};

   //Loop over events
   for(int entry = 0; entry < nentries; entry++) {
     Float_t error_energy_arr[28] = {};
     b_HcalBarrelHits_->GetEntry(entry);
     b_HcalBarrelHits_energy->GetEntry(entry);
     //Loop over hits
     for(int i = 0; i < HcalBarrelHits_; i++) {
       Float_t curr_x = HcalBarrelHits_position_x[i];
       for(int j = 0; j < num_layers; j++) {
	 //need to find the layer of this particular hit
	 if(layer_map[j] == curr_x) {
	   curr_layer = j;
	   break; //can break after we found the layer
	 }
       }
       if(curr_layer == -1){continue;}
       error_energy_arr[curr_layer] += HcalBarrelHits_energy[i];
       //Calculate residual
       
       curr_layer = -1; //used to check if the hit corresponds to a layer or not. May want to bin instead
      }
     for(int layer = 0; layer < num_layers; layer++) {
       sq_residuals[layer] += pow(error_energy_arr[layer] - avg_energy[layer],2);
     }
   }
   for(int i = 0; i < num_layers; i++) {
     y_err[i] = pow(sq_residuals[i] / nentries,0.5);
   }
   TCanvas *c1 = new TCanvas();
   TGraph *gr = new TGraphErrors(num_layers,layer_map, avg_energy,x_err, y_err);
   
   gr->SetTitle("avg energy deposition in each layer 10000 events (1GeV mu- gun)");
   auto xaxis = gr->GetXaxis();
   xaxis->SetTitle("Barrel hit x position (mm)");
   auto yaxis = gr->GetYaxis();
   yaxis->SetTitle("Energy deposited per event (GeV)");
   
   xaxis->SetLimits(1600,3000);
   gr->SetMarkerStyle(34);
   gr->SetMarkerSize(0.5);
   gr->SetMarkerColor(2);
   gr->Draw("AP");
   
   TFile f("root_files/graphs/april_6/mu_1GeV_energy_run1.root","recreate");
   gr->Write();
   c1->Print("plots/april_6/mu_1GeV_energy_run1.pdf");
}
