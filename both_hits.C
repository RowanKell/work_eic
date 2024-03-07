#define both_hits_cxx
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>

int both_hits() {
  
  TFile *fpi = new TFile("root_files/histos/pi_hits.root");
  TFile *fmu = new TFile("root_files/histos/mu_hits.root");

  TGraph *grp = (TGraph*)fpi->Get("Graph");
  TGraph *grm = (TGraph*)fmu->Get("Graph");

  TCanvas *c1 = new TCanvas();

  grp->SetMarkerColor(2);
  grp->SetMarkerStyle(21);
  grp->Draw("AP");
  grp->SetMarkerSize(0.2);
  grm->SetMarkerColor(4);
  grm->SetMarkerStyle(21);
  grm->Draw("P");
  grm->SetMarkerSize(0.2);

  auto legend = new TLegend(0.1,0.9,0.3,.7);
  TGraph *pmarker = new TGraph();
  TGraph *mmarker = new TGraph();
  pmarker->SetMarkerStyle(21);
  mmarker->SetMarkerStyle(21);
  pmarker->SetMarkerSize(1);
  mmarker->SetMarkerSize(1);
  pmarker->SetMarkerColor(2);
  mmarker->SetMarkerColor(4);
  legend->AddEntry(pmarker, "pi- hits","p");
  legend->AddEntry(mmarker, "mu- hits","p");
  legend->SetTextSize(0.025);
  legend->Draw();
  c1->Print("plots/pimuhits1000.jpg");
  return 0;
}
