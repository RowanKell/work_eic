#define plot_2_hist_cxx
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>

int plot_2_hist() {

  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  TFile *fpi = new TFile("root_files/histos/pi_hist.root");
  TFile *fmu = new TFile("root_files/histos/mu_hist.root");
  
  TH1D *hpi = (TH1D*)fpi->Get("hpi");
  TH1D *hmu = (TH1D*)fmu->Get("hmu");

  TCanvas *c1 = new TCanvas("c1","c1",3200,2000);
  
  hmu->Draw();
  hpi->Draw("same");
  c1->SaveAs("plots/pimu50000_nostats.pdf");
  return 0;
}
