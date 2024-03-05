#define pi_endpoint_cxx
#include "pi_endpoint.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>

void pi_endpoint::Loop()
{
  if (fChain == 0) return;
  Long64_t nentries = fChain->GetEntriesFast();
  Long64_t nbytes = 0, nb = 0;

  double x;
  double y;
  TH1F *h1 = new TH1F("hpi","radial position;r;counts",100,1500,3000);
  for (Long64_t jentry=0; jentry<nentries;jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    nb = fChain->GetEntry(jentry);   nbytes += nb; //throws error
    // if (Cut(ientry) < 0) continue;
    for(int x_iter = 0; x_iter < MCParticles_; x_iter++){
      x = MCParticles_endpoint_x[x_iter];
      y = MCParticles_endpoint_y[x_iter];
      h1->Fill(pow(pow(x,2) + pow(y,2),0.5));
    }
  }
  h1->SetLineColor(kRed);
  h1->SetFillColor(kRed);
  TCanvas *c1 = new TCanvas();
  h1->Draw();

  TFile f("root_files/histos/pi_hist.root","recreate");
  h1->Write();
}
