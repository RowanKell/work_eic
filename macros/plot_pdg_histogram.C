#include <TFile.h>
#include <TTree.h>
#include <TH1I.h>

void plot_pdg_histogram(const char* filename = "root_files/EPIC_KLM_1GeV_K_1000.edm4hep.root") {
  // Open the ROOT file
  TFile* file = TFile::Open(filename);
  if (!file) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
  }

  // Get the MCParticle tree
  TTree* tree = (TTree*)file->Get("events");
  if (!tree) {
    std::cerr << "Error getting MCParticles tree" << std::endl;
    return;
  }

  // Create a histogram for PDG values
  TH1I* pdg_hist = new TH1I("pdg_hist", "MCParticle PDG Values", 100, -300, 300);

  // Get the PDG branch
  int pdg[10];
  int MCParticles = 0;
  tree->SetBranchAddress("MCParticles",&MCParticles);
  tree->SetBranchAddress("MCParticles.PDG", &pdg);

  // Loop over all events and particles
  for (Long64_t i = 0; i < tree->GetEntries(); ++i) {
    tree->GetEntry(i);
    for(int j = 0; j < MCParticles; j++){
      pdg_hist->Fill(pdg[j]);
    }
  }

  // Draw the histogram
  pdg_hist->Draw();

  // Keep the canvas open
  gPad->WaitPrimitive();

  // Clean up
  delete pdg_hist;
  delete tree;
  delete file;
}
