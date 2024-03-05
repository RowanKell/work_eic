#define plot_2_hist_cxx
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>

int plot_2_hist() {
  
  TFile *fpi = new TFile("root_files/histos/pi_hist.root");
  TFile *fmu = new TFile("root_files/histos/mu_hist.root");
  
  TH1D *hpi = (TH1D*)fpi->Get("hpi");
  TH1D *hmu = (TH1D*)fmu->Get("hmu");

  TCanvas *c1 = new TCanvas();
  
  hmu->Draw();
  hpi->Draw("same");
  c1->Print("plots/pimu50000.jpg");
  return 0;
}
