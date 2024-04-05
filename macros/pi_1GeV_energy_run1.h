//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Fri Apr  5 15:05:10 2024 by ROOT version 6.28/10
// from TTree events/events data tree
// found on file: root_files/pi_10000_april_5_run1_1_GeV.edm4hep.root
//////////////////////////////////////////////////////////

#ifndef pi_1GeV_energy_run1_h
#define pi_1GeV_energy_run1_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

// Header file for the classes stored in the TTree if any.

class pi_1GeV_energy_run1 {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

// Fixed size dimensions of array or collections stored in the TTree if any.
   static constexpr Int_t kMaxEventHeader = 1;
   static constexpr Int_t kMaxHcalBarrelHits = 227;
   static constexpr Int_t kMax_HcalBarrelHits_contributions = 774;
   static constexpr Int_t kMaxHcalBarrelHitsContributions = 774;
   static constexpr Int_t kMax_HcalBarrelHitsContributions_particle = 774;
   static constexpr Int_t kMaxMCParticles = 263;
   static constexpr Int_t kMax_MCParticles_parents = 262;
   static constexpr Int_t kMax_MCParticles_daughters = 262;
   static constexpr Int_t kMax_intMap = 1;
   static constexpr Int_t kMax_floatMap = 1;
   static constexpr Int_t kMax_stringMap = 1;
   static constexpr Int_t kMax_doubleMap = 1;

   // Declaration of leaf types
   Int_t           EventHeader_;
   Int_t           EventHeader_eventNumber[kMaxEventHeader];   //[EventHeader_]
   Int_t           EventHeader_runNumber[kMaxEventHeader];   //[EventHeader_]
   ULong_t         EventHeader_timeStamp[kMaxEventHeader];   //[EventHeader_]
   Float_t         EventHeader_weight[kMaxEventHeader];   //[EventHeader_]
   Int_t           HcalBarrelHits_;
   ULong_t         HcalBarrelHits_cellID[kMaxHcalBarrelHits];   //[HcalBarrelHits_]
   Float_t         HcalBarrelHits_energy[kMaxHcalBarrelHits];   //[HcalBarrelHits_]
   Float_t         HcalBarrelHits_position_x[kMaxHcalBarrelHits];   //[HcalBarrelHits_]
   Float_t         HcalBarrelHits_position_y[kMaxHcalBarrelHits];   //[HcalBarrelHits_]
   Float_t         HcalBarrelHits_position_z[kMaxHcalBarrelHits];   //[HcalBarrelHits_]
   UInt_t          HcalBarrelHits_contributions_begin[kMaxHcalBarrelHits];   //[HcalBarrelHits_]
   UInt_t          HcalBarrelHits_contributions_end[kMaxHcalBarrelHits];   //[HcalBarrelHits_]
   Int_t           _HcalBarrelHits_contributions_;
   Int_t           _HcalBarrelHits_contributions_index[kMax_HcalBarrelHits_contributions];   //[_HcalBarrelHits_contributions_]
   UInt_t          _HcalBarrelHits_contributions_collectionID[kMax_HcalBarrelHits_contributions];   //[_HcalBarrelHits_contributions_]
   Int_t           HcalBarrelHitsContributions_;
   Int_t           HcalBarrelHitsContributions_PDG[kMaxHcalBarrelHitsContributions];   //[HcalBarrelHitsContributions_]
   Float_t         HcalBarrelHitsContributions_energy[kMaxHcalBarrelHitsContributions];   //[HcalBarrelHitsContributions_]
   Float_t         HcalBarrelHitsContributions_time[kMaxHcalBarrelHitsContributions];   //[HcalBarrelHitsContributions_]
   Float_t         HcalBarrelHitsContributions_stepPosition_x[kMaxHcalBarrelHitsContributions];   //[HcalBarrelHitsContributions_]
   Float_t         HcalBarrelHitsContributions_stepPosition_y[kMaxHcalBarrelHitsContributions];   //[HcalBarrelHitsContributions_]
   Float_t         HcalBarrelHitsContributions_stepPosition_z[kMaxHcalBarrelHitsContributions];   //[HcalBarrelHitsContributions_]
   Int_t           _HcalBarrelHitsContributions_particle_;
   Int_t           _HcalBarrelHitsContributions_particle_index[kMax_HcalBarrelHitsContributions_particle];   //[_HcalBarrelHitsContributions_particle_]
   UInt_t          _HcalBarrelHitsContributions_particle_collectionID[kMax_HcalBarrelHitsContributions_particle];   //[_HcalBarrelHitsContributions_particle_]
   Int_t           MCParticles_;
   Int_t           MCParticles_PDG[kMaxMCParticles];   //[MCParticles_]
   Int_t           MCParticles_generatorStatus[kMaxMCParticles];   //[MCParticles_]
   Int_t           MCParticles_simulatorStatus[kMaxMCParticles];   //[MCParticles_]
   Float_t         MCParticles_charge[kMaxMCParticles];   //[MCParticles_]
   Float_t         MCParticles_time[kMaxMCParticles];   //[MCParticles_]
   Double_t        MCParticles_mass[kMaxMCParticles];   //[MCParticles_]
   Double_t        MCParticles_vertex_x[kMaxMCParticles];   //[MCParticles_]
   Double_t        MCParticles_vertex_y[kMaxMCParticles];   //[MCParticles_]
   Double_t        MCParticles_vertex_z[kMaxMCParticles];   //[MCParticles_]
   Double_t        MCParticles_endpoint_x[kMaxMCParticles];   //[MCParticles_]
   Double_t        MCParticles_endpoint_y[kMaxMCParticles];   //[MCParticles_]
   Double_t        MCParticles_endpoint_z[kMaxMCParticles];   //[MCParticles_]
   Float_t         MCParticles_momentum_x[kMaxMCParticles];   //[MCParticles_]
   Float_t         MCParticles_momentum_y[kMaxMCParticles];   //[MCParticles_]
   Float_t         MCParticles_momentum_z[kMaxMCParticles];   //[MCParticles_]
   Float_t         MCParticles_momentumAtEndpoint_x[kMaxMCParticles];   //[MCParticles_]
   Float_t         MCParticles_momentumAtEndpoint_y[kMaxMCParticles];   //[MCParticles_]
   Float_t         MCParticles_momentumAtEndpoint_z[kMaxMCParticles];   //[MCParticles_]
   Float_t         MCParticles_spin_x[kMaxMCParticles];   //[MCParticles_]
   Float_t         MCParticles_spin_y[kMaxMCParticles];   //[MCParticles_]
   Float_t         MCParticles_spin_z[kMaxMCParticles];   //[MCParticles_]
   Int_t           MCParticles_colorFlow_a[kMaxMCParticles];   //[MCParticles_]
   Int_t           MCParticles_colorFlow_b[kMaxMCParticles];   //[MCParticles_]
   UInt_t          MCParticles_parents_begin[kMaxMCParticles];   //[MCParticles_]
   UInt_t          MCParticles_parents_end[kMaxMCParticles];   //[MCParticles_]
   UInt_t          MCParticles_daughters_begin[kMaxMCParticles];   //[MCParticles_]
   UInt_t          MCParticles_daughters_end[kMaxMCParticles];   //[MCParticles_]
   Int_t           _MCParticles_parents_;
   Int_t           _MCParticles_parents_index[kMax_MCParticles_parents];   //[_MCParticles_parents_]
   UInt_t          _MCParticles_parents_collectionID[kMax_MCParticles_parents];   //[_MCParticles_parents_]
   Int_t           _MCParticles_daughters_;
   Int_t           _MCParticles_daughters_index[kMax_MCParticles_daughters];   //[_MCParticles_daughters_]
   UInt_t          _MCParticles_daughters_collectionID[kMax_MCParticles_daughters];   //[_MCParticles_daughters_]
 //podio::GenericParameters *PARAMETERS;
   Int_t           _intMap_;
   string          _intMap_first[kMax_intMap];
   vector<int>     _intMap_second[kMax_intMap];
   Int_t           _floatMap_;
   string          _floatMap_first[kMax_floatMap];
   vector<float>   _floatMap_second[kMax_floatMap];
   Int_t           _stringMap_;
   string          _stringMap_first[kMax_stringMap];
   vector<string>  _stringMap_second[kMax_stringMap];
   Int_t           _doubleMap_;
   string          _doubleMap_first[kMax_doubleMap];
   vector<double>  _doubleMap_second[kMax_doubleMap];

   // List of branches
   TBranch        *b_EventHeader_;   //!
   TBranch        *b_EventHeader_eventNumber;   //!
   TBranch        *b_EventHeader_runNumber;   //!
   TBranch        *b_EventHeader_timeStamp;   //!
   TBranch        *b_EventHeader_weight;   //!
   TBranch        *b_HcalBarrelHits_;   //!
   TBranch        *b_HcalBarrelHits_cellID;   //!
   TBranch        *b_HcalBarrelHits_energy;   //!
   TBranch        *b_HcalBarrelHits_position_x;   //!
   TBranch        *b_HcalBarrelHits_position_y;   //!
   TBranch        *b_HcalBarrelHits_position_z;   //!
   TBranch        *b_HcalBarrelHits_contributions_begin;   //!
   TBranch        *b_HcalBarrelHits_contributions_end;   //!
   TBranch        *b__HcalBarrelHits_contributions_;   //!
   TBranch        *b__HcalBarrelHits_contributions_index;   //!
   TBranch        *b__HcalBarrelHits_contributions_collectionID;   //!
   TBranch        *b_HcalBarrelHitsContributions_;   //!
   TBranch        *b_HcalBarrelHitsContributions_PDG;   //!
   TBranch        *b_HcalBarrelHitsContributions_energy;   //!
   TBranch        *b_HcalBarrelHitsContributions_time;   //!
   TBranch        *b_HcalBarrelHitsContributions_stepPosition_x;   //!
   TBranch        *b_HcalBarrelHitsContributions_stepPosition_y;   //!
   TBranch        *b_HcalBarrelHitsContributions_stepPosition_z;   //!
   TBranch        *b__HcalBarrelHitsContributions_particle_;   //!
   TBranch        *b__HcalBarrelHitsContributions_particle_index;   //!
   TBranch        *b__HcalBarrelHitsContributions_particle_collectionID;   //!
   TBranch        *b_MCParticles_;   //!
   TBranch        *b_MCParticles_PDG;   //!
   TBranch        *b_MCParticles_generatorStatus;   //!
   TBranch        *b_MCParticles_simulatorStatus;   //!
   TBranch        *b_MCParticles_charge;   //!
   TBranch        *b_MCParticles_time;   //!
   TBranch        *b_MCParticles_mass;   //!
   TBranch        *b_MCParticles_vertex_x;   //!
   TBranch        *b_MCParticles_vertex_y;   //!
   TBranch        *b_MCParticles_vertex_z;   //!
   TBranch        *b_MCParticles_endpoint_x;   //!
   TBranch        *b_MCParticles_endpoint_y;   //!
   TBranch        *b_MCParticles_endpoint_z;   //!
   TBranch        *b_MCParticles_momentum_x;   //!
   TBranch        *b_MCParticles_momentum_y;   //!
   TBranch        *b_MCParticles_momentum_z;   //!
   TBranch        *b_MCParticles_momentumAtEndpoint_x;   //!
   TBranch        *b_MCParticles_momentumAtEndpoint_y;   //!
   TBranch        *b_MCParticles_momentumAtEndpoint_z;   //!
   TBranch        *b_MCParticles_spin_x;   //!
   TBranch        *b_MCParticles_spin_y;   //!
   TBranch        *b_MCParticles_spin_z;   //!
   TBranch        *b_MCParticles_colorFlow_a;   //!
   TBranch        *b_MCParticles_colorFlow_b;   //!
   TBranch        *b_MCParticles_parents_begin;   //!
   TBranch        *b_MCParticles_parents_end;   //!
   TBranch        *b_MCParticles_daughters_begin;   //!
   TBranch        *b_MCParticles_daughters_end;   //!
   TBranch        *b__MCParticles_parents_;   //!
   TBranch        *b__MCParticles_parents_index;   //!
   TBranch        *b__MCParticles_parents_collectionID;   //!
   TBranch        *b__MCParticles_daughters_;   //!
   TBranch        *b__MCParticles_daughters_index;   //!
   TBranch        *b__MCParticles_daughters_collectionID;   //!
   TBranch        *b_PARAMETERS__intMap_;   //!
   TBranch        *b__intMap_first;   //!
   TBranch        *b__intMap_second;   //!
   TBranch        *b_PARAMETERS__floatMap_;   //!
   TBranch        *b__floatMap_first;   //!
   TBranch        *b__floatMap_second;   //!
   TBranch        *b_PARAMETERS__stringMap_;   //!
   TBranch        *b__stringMap_first;   //!
   TBranch        *b__stringMap_second;   //!
   TBranch        *b_PARAMETERS__doubleMap_;   //!
   TBranch        *b__doubleMap_first;   //!
   TBranch        *b__doubleMap_second;   //!

   pi_1GeV_energy_run1(TTree *tree=0);
   virtual ~pi_1GeV_energy_run1();
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   virtual void     Loop();
   virtual Bool_t   Notify();
   virtual void     Show(Long64_t entry = -1);
};

#endif

#ifdef pi_1GeV_energy_run1_cxx
pi_1GeV_energy_run1::pi_1GeV_energy_run1(TTree *tree) : fChain(0) 
{
// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
   if (tree == 0) {
      TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("root_files/pi_10000_april_5_run1_1_GeV.edm4hep.root");
      if (!f || !f->IsOpen()) {
         f = new TFile("root_files/pi_10000_april_5_run1_1_GeV.edm4hep.root");
      }
      f->GetObject("events",tree);

   }
   Init(tree);
   Loop();
}

pi_1GeV_energy_run1::~pi_1GeV_energy_run1()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t pi_1GeV_energy_run1::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t pi_1GeV_energy_run1::LoadTree(Long64_t entry)
{
// Set the environment to read one entry
   if (!fChain) return -5;
   Long64_t centry = fChain->LoadTree(entry);
   if (centry < 0) return centry;
   if (fChain->GetTreeNumber() != fCurrent) {
      fCurrent = fChain->GetTreeNumber();
      Notify();
   }
   return centry;
}

void pi_1GeV_energy_run1::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("EventHeader", &EventHeader_, &b_EventHeader_);
   fChain->SetBranchAddress("EventHeader.eventNumber", EventHeader_eventNumber, &b_EventHeader_eventNumber);
   fChain->SetBranchAddress("EventHeader.runNumber", EventHeader_runNumber, &b_EventHeader_runNumber);
   fChain->SetBranchAddress("EventHeader.timeStamp", EventHeader_timeStamp, &b_EventHeader_timeStamp);
   fChain->SetBranchAddress("EventHeader.weight", EventHeader_weight, &b_EventHeader_weight);
   fChain->SetBranchAddress("HcalBarrelHits", &HcalBarrelHits_, &b_HcalBarrelHits_);
   fChain->SetBranchAddress("HcalBarrelHits.cellID", HcalBarrelHits_cellID, &b_HcalBarrelHits_cellID);
   fChain->SetBranchAddress("HcalBarrelHits.energy", HcalBarrelHits_energy, &b_HcalBarrelHits_energy);
   fChain->SetBranchAddress("HcalBarrelHits.position.x", HcalBarrelHits_position_x, &b_HcalBarrelHits_position_x);
   fChain->SetBranchAddress("HcalBarrelHits.position.y", HcalBarrelHits_position_y, &b_HcalBarrelHits_position_y);
   fChain->SetBranchAddress("HcalBarrelHits.position.z", HcalBarrelHits_position_z, &b_HcalBarrelHits_position_z);
   fChain->SetBranchAddress("HcalBarrelHits.contributions_begin", HcalBarrelHits_contributions_begin, &b_HcalBarrelHits_contributions_begin);
   fChain->SetBranchAddress("HcalBarrelHits.contributions_end", HcalBarrelHits_contributions_end, &b_HcalBarrelHits_contributions_end);
   fChain->SetBranchAddress("_HcalBarrelHits_contributions", &_HcalBarrelHits_contributions_, &b__HcalBarrelHits_contributions_);
   fChain->SetBranchAddress("_HcalBarrelHits_contributions.index", _HcalBarrelHits_contributions_index, &b__HcalBarrelHits_contributions_index);
   fChain->SetBranchAddress("_HcalBarrelHits_contributions.collectionID", _HcalBarrelHits_contributions_collectionID, &b__HcalBarrelHits_contributions_collectionID);
   fChain->SetBranchAddress("HcalBarrelHitsContributions", &HcalBarrelHitsContributions_, &b_HcalBarrelHitsContributions_);
   fChain->SetBranchAddress("HcalBarrelHitsContributions.PDG", HcalBarrelHitsContributions_PDG, &b_HcalBarrelHitsContributions_PDG);
   fChain->SetBranchAddress("HcalBarrelHitsContributions.energy", HcalBarrelHitsContributions_energy, &b_HcalBarrelHitsContributions_energy);
   fChain->SetBranchAddress("HcalBarrelHitsContributions.time", HcalBarrelHitsContributions_time, &b_HcalBarrelHitsContributions_time);
   fChain->SetBranchAddress("HcalBarrelHitsContributions.stepPosition.x", HcalBarrelHitsContributions_stepPosition_x, &b_HcalBarrelHitsContributions_stepPosition_x);
   fChain->SetBranchAddress("HcalBarrelHitsContributions.stepPosition.y", HcalBarrelHitsContributions_stepPosition_y, &b_HcalBarrelHitsContributions_stepPosition_y);
   fChain->SetBranchAddress("HcalBarrelHitsContributions.stepPosition.z", HcalBarrelHitsContributions_stepPosition_z, &b_HcalBarrelHitsContributions_stepPosition_z);
   fChain->SetBranchAddress("_HcalBarrelHitsContributions_particle", &_HcalBarrelHitsContributions_particle_, &b__HcalBarrelHitsContributions_particle_);
   fChain->SetBranchAddress("_HcalBarrelHitsContributions_particle.index", _HcalBarrelHitsContributions_particle_index, &b__HcalBarrelHitsContributions_particle_index);
   fChain->SetBranchAddress("_HcalBarrelHitsContributions_particle.collectionID", _HcalBarrelHitsContributions_particle_collectionID, &b__HcalBarrelHitsContributions_particle_collectionID);
   fChain->SetBranchAddress("MCParticles", &MCParticles_, &b_MCParticles_);
   fChain->SetBranchAddress("MCParticles.PDG", MCParticles_PDG, &b_MCParticles_PDG);
   fChain->SetBranchAddress("MCParticles.generatorStatus", MCParticles_generatorStatus, &b_MCParticles_generatorStatus);
   fChain->SetBranchAddress("MCParticles.simulatorStatus", MCParticles_simulatorStatus, &b_MCParticles_simulatorStatus);
   fChain->SetBranchAddress("MCParticles.charge", MCParticles_charge, &b_MCParticles_charge);
   fChain->SetBranchAddress("MCParticles.time", MCParticles_time, &b_MCParticles_time);
   fChain->SetBranchAddress("MCParticles.mass", MCParticles_mass, &b_MCParticles_mass);
   fChain->SetBranchAddress("MCParticles.vertex.x", MCParticles_vertex_x, &b_MCParticles_vertex_x);
   fChain->SetBranchAddress("MCParticles.vertex.y", MCParticles_vertex_y, &b_MCParticles_vertex_y);
   fChain->SetBranchAddress("MCParticles.vertex.z", MCParticles_vertex_z, &b_MCParticles_vertex_z);
   fChain->SetBranchAddress("MCParticles.endpoint.x", MCParticles_endpoint_x, &b_MCParticles_endpoint_x);
   fChain->SetBranchAddress("MCParticles.endpoint.y", MCParticles_endpoint_y, &b_MCParticles_endpoint_y);
   fChain->SetBranchAddress("MCParticles.endpoint.z", MCParticles_endpoint_z, &b_MCParticles_endpoint_z);
   fChain->SetBranchAddress("MCParticles.momentum.x", MCParticles_momentum_x, &b_MCParticles_momentum_x);
   fChain->SetBranchAddress("MCParticles.momentum.y", MCParticles_momentum_y, &b_MCParticles_momentum_y);
   fChain->SetBranchAddress("MCParticles.momentum.z", MCParticles_momentum_z, &b_MCParticles_momentum_z);
   fChain->SetBranchAddress("MCParticles.momentumAtEndpoint.x", MCParticles_momentumAtEndpoint_x, &b_MCParticles_momentumAtEndpoint_x);
   fChain->SetBranchAddress("MCParticles.momentumAtEndpoint.y", MCParticles_momentumAtEndpoint_y, &b_MCParticles_momentumAtEndpoint_y);
   fChain->SetBranchAddress("MCParticles.momentumAtEndpoint.z", MCParticles_momentumAtEndpoint_z, &b_MCParticles_momentumAtEndpoint_z);
   fChain->SetBranchAddress("MCParticles.spin.x", MCParticles_spin_x, &b_MCParticles_spin_x);
   fChain->SetBranchAddress("MCParticles.spin.y", MCParticles_spin_y, &b_MCParticles_spin_y);
   fChain->SetBranchAddress("MCParticles.spin.z", MCParticles_spin_z, &b_MCParticles_spin_z);
   fChain->SetBranchAddress("MCParticles.colorFlow.a", MCParticles_colorFlow_a, &b_MCParticles_colorFlow_a);
   fChain->SetBranchAddress("MCParticles.colorFlow.b", MCParticles_colorFlow_b, &b_MCParticles_colorFlow_b);
   fChain->SetBranchAddress("MCParticles.parents_begin", MCParticles_parents_begin, &b_MCParticles_parents_begin);
   fChain->SetBranchAddress("MCParticles.parents_end", MCParticles_parents_end, &b_MCParticles_parents_end);
   fChain->SetBranchAddress("MCParticles.daughters_begin", MCParticles_daughters_begin, &b_MCParticles_daughters_begin);
   fChain->SetBranchAddress("MCParticles.daughters_end", MCParticles_daughters_end, &b_MCParticles_daughters_end);
   fChain->SetBranchAddress("_MCParticles_parents", &_MCParticles_parents_, &b__MCParticles_parents_);
   fChain->SetBranchAddress("_MCParticles_parents.index", _MCParticles_parents_index, &b__MCParticles_parents_index);
   fChain->SetBranchAddress("_MCParticles_parents.collectionID", _MCParticles_parents_collectionID, &b__MCParticles_parents_collectionID);
   fChain->SetBranchAddress("_MCParticles_daughters", &_MCParticles_daughters_, &b__MCParticles_daughters_);
   fChain->SetBranchAddress("_MCParticles_daughters.index", _MCParticles_daughters_index, &b__MCParticles_daughters_index);
   fChain->SetBranchAddress("_MCParticles_daughters.collectionID", _MCParticles_daughters_collectionID, &b__MCParticles_daughters_collectionID);
   fChain->SetBranchAddress("_intMap", &_intMap_, &b_PARAMETERS__intMap_);
   fChain->SetBranchAddress("_intMap.first", &_intMap_first, &b__intMap_first);
   fChain->SetBranchAddress("_intMap.second", &_intMap_second, &b__intMap_second);
   fChain->SetBranchAddress("_floatMap", &_floatMap_, &b_PARAMETERS__floatMap_);
   fChain->SetBranchAddress("_floatMap.first", &_floatMap_first, &b__floatMap_first);
   fChain->SetBranchAddress("_floatMap.second", &_floatMap_second, &b__floatMap_second);
   fChain->SetBranchAddress("_stringMap", &_stringMap_, &b_PARAMETERS__stringMap_);
   fChain->SetBranchAddress("_stringMap.first", &_stringMap_first, &b__stringMap_first);
   fChain->SetBranchAddress("_stringMap.second", &_stringMap_second, &b__stringMap_second);
   fChain->SetBranchAddress("_doubleMap", &_doubleMap_, &b_PARAMETERS__doubleMap_);
   fChain->SetBranchAddress("_doubleMap.first", &_doubleMap_first, &b__doubleMap_first);
   fChain->SetBranchAddress("_doubleMap.second", &_doubleMap_second, &b__doubleMap_second);
   Notify();
}

Bool_t pi_1GeV_energy_run1::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void pi_1GeV_energy_run1::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t pi_1GeV_energy_run1::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef pi_1GeV_energy_run1_cxx
