#ifdef __CLING__
#pragma cling optimize(0)
#endif
void c1()
{
//=========Macro generated from canvas: c1/c1
//=========  (Wed May  8 16:34:36 2024) by ROOT version 6.30/02
   TCanvas *c1 = new TCanvas("c1", "c1",10,64,1246,500);
   c1->Range(0,0,1,1);
   c1->SetFillColor(0);
   c1->SetBorderMode(0);
   c1->SetBorderSize(2);
   c1->SetFrameBorderMode(0);
  
// ------------>Primitives in pad: c1_1
   TPad *c1_1__0 = new TPad("c1_1", "c1_1",0.01,0.01,0.49,0.99);
   c1_1__0->Draw();
   c1_1__0->cd();
   c1_1__0->Range(1425,50.6675,3175,144.4025);
   c1_1__0->SetFillColor(0);
   c1_1__0->SetBorderMode(0);
   c1_1__0->SetBorderSize(2);
   c1_1__0->SetFrameBorderMode(0);
   c1_1__0->SetFrameBorderMode(0);
   
   Double_t Graph0_fx1[28] = { 1820, 1830.8, 1841.4, 1907.5, 1918.1, 1984.2, 1994.8, 2060.9, 2071.5, 2137.6, 2148.2, 2214.3, 2224.9, 2291, 2301.6, 2367.7, 2378.3,
   2444.4, 2455, 2521.1, 2531.7, 2597.8, 2608.4, 2674.5, 2685.1, 2751.2, 2761.8, 2827.9 };
   Double_t Graph0_fy1[28] = { 66.29, 128.78, 119.92, 115.98, 120.33, 122.69, 118.32, 117.95, 118.07, 114.51, 112.22, 108.96, 111.82, 113.87, 108.98, 104.22, 102.69,
   100.26, 99.5, 98.06, 93.93, 90.36, 92.91, 91.86, 94.35, 93.7, 85.3, 82.56 };
   TGraph *graph = new TGraph(28,Graph0_fx1,Graph0_fy1);
   graph->SetName("Graph0");
   graph->SetTitle("avg # of hits in each layer 100 events (5GeV mu- gun)");
   graph->SetFillStyle(1000);
   graph->SetMarkerColor(2);
   graph->SetMarkerStyle(34);
   graph->SetMarkerSize(0.5);
   
   TH1F *Graph_Graph01 = new TH1F("Graph_Graph01","avg # of hits in each layer 100 events (5GeV mu- gun)",100,1600,3000);
   Graph_Graph01->SetMinimum(60.041);
   Graph_Graph01->SetMaximum(135.029);
   Graph_Graph01->SetDirectory(nullptr);
   Graph_Graph01->SetStats(0);

   Int_t ci;      // for color index setting
   TColor *color; // for color definition with alpha
   ci = TColor::GetColor("#000099");
   Graph_Graph01->SetLineColor(ci);
   Graph_Graph01->GetXaxis()->SetTitle("Barrel hit x position (mm)");
   Graph_Graph01->GetXaxis()->SetLabelFont(42);
   Graph_Graph01->GetXaxis()->SetTitleOffset(1);
   Graph_Graph01->GetXaxis()->SetTitleFont(42);
   Graph_Graph01->GetYaxis()->SetTitle("# of hits per event");
   Graph_Graph01->GetYaxis()->SetLabelFont(42);
   Graph_Graph01->GetYaxis()->SetTitleFont(42);
   Graph_Graph01->GetZaxis()->SetLabelFont(42);
   Graph_Graph01->GetZaxis()->SetTitleOffset(1);
   Graph_Graph01->GetZaxis()->SetTitleFont(42);
   graph->SetHistogram(Graph_Graph01);
   
   graph->Draw("ap");
   
   TPaveText *pt = new TPaveText(0.15,0.94,0.85,0.995,"blNDC");
   pt->SetName("title");
   pt->SetBorderSize(0);
   pt->SetFillColor(0);
   pt->SetFillStyle(0);
   pt->SetTextFont(42);
   TText *pt_LaTex = pt->AddText("avg # of hits in each layer 100 events (5GeV mu- gun)");
   pt->Draw();
   c1_1__0->Modified();
   c1->cd();
  
// ------------>Primitives in pad: c1_2
   TPad *c1_2__1 = new TPad("c1_2", "c1_2",0.51,0.01,0.99,0.99);
   c1_2__1->Draw();
   c1_2__1->cd();
   c1_2__1->Range(1425,326.975,3175,1070.045);
   c1_2__1->SetFillColor(0);
   c1_2__1->SetBorderMode(0);
   c1_2__1->SetBorderSize(2);
   c1_2__1->SetFrameBorderMode(0);
   c1_2__1->SetFrameBorderMode(0);
   
   Double_t Graph0_fx2[28] = { 1820, 1830.8, 1841.4, 1907.5, 1918.1, 1984.2, 1994.8, 2060.9, 2071.5, 2137.6, 2148.2, 2214.3, 2224.9, 2291, 2301.6, 2367.7, 2378.3,
   2444.4, 2455, 2521.1, 2531.7, 2597.8, 2608.4, 2674.5, 2685.1, 2751.2, 2761.8, 2827.9 };
   Double_t Graph0_fy2[28] = { 450.82, 902.14, 850.01, 868.63, 910.47, 887.33, 895.31, 910.14, 929.77, 916.4, 920.89, 932.16, 932.75, 946.2, 885.36, 886.57, 875.99,
   839.71, 870.95, 873.68, 838.1, 819.38, 820.42, 870.35, 876.62, 874.26, 842.84, 825.86 };
   graph = new TGraph(28,Graph0_fx2,Graph0_fy2);
   graph->SetName("Graph0");
   graph->SetTitle("avg # of -22 PDG in each layer 100 events (5GeV mu- gun)");
   graph->SetFillStyle(1000);
   graph->SetMarkerColor(2);
   graph->SetMarkerStyle(34);
   graph->SetMarkerSize(0.5);
   
   TH1F *Graph_Graph02 = new TH1F("Graph_Graph02","avg # of -22 PDG in each layer 100 events (5GeV mu- gun)",100,1600,3000);
   Graph_Graph02->SetMinimum(401.282);
   Graph_Graph02->SetMaximum(995.738);
   Graph_Graph02->SetDirectory(nullptr);
   Graph_Graph02->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_Graph02->SetLineColor(ci);
   Graph_Graph02->GetXaxis()->SetTitle("vertex x position (mm)");
   Graph_Graph02->GetXaxis()->SetLabelFont(42);
   Graph_Graph02->GetXaxis()->SetTitleOffset(1);
   Graph_Graph02->GetXaxis()->SetTitleFont(42);
   Graph_Graph02->GetYaxis()->SetTitle("# of optical photons in MCParticles per event");
   Graph_Graph02->GetYaxis()->SetLabelFont(42);
   Graph_Graph02->GetYaxis()->SetTitleFont(42);
   Graph_Graph02->GetZaxis()->SetLabelFont(42);
   Graph_Graph02->GetZaxis()->SetTitleOffset(1);
   Graph_Graph02->GetZaxis()->SetTitleFont(42);
   graph->SetHistogram(Graph_Graph02);
   
   graph->Draw("ap");
   
   pt = new TPaveText(0.15,0.94,0.85,0.995,"blNDC");
   pt->SetName("title");
   pt->SetBorderSize(0);
   pt->SetFillColor(0);
   pt->SetFillStyle(0);
   pt->SetTextFont(42);
   pt_LaTex = pt->AddText("avg # of -22 PDG in each layer 100 events (5GeV mu- gun)");
   pt->Draw();
   c1_2__1->Modified();
   c1->cd();
   c1->Modified();
   c1->SetSelected(c1);
}
