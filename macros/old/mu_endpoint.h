//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Thu Feb 29 15:20:47 2024 by ROOT version 6.28/10
// from TTree events/events data tree
// found on file: test_100_pion_10GeV.edm4hep.root
//////////////////////////////////////////////////////////

#ifndef mu_endpoint_h
#define mu_endpoint_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

// Header file for the classes stored in the TTree if any.

class mu_endpoint {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

// Fixed size dimensions of array or collections stored in the TTree if any.
   static constexpr Int_t kMaxB0ECalHits = 1;
   static constexpr Int_t kMax_B0ECalHits_contributions = 1;
   static constexpr Int_t kMaxB0ECalHitsContributions = 1;
   static constexpr Int_t kMax_B0ECalHitsContributions_particle = 1;
   static constexpr Int_t kMaxB0TrackerHits = 1;
   static constexpr Int_t kMax_B0TrackerHits_MCParticle = 1;
   static constexpr Int_t kMaxDIRCBarHits = 1;
   static constexpr Int_t kMax_DIRCBarHits_MCParticle = 1;
   static constexpr Int_t kMaxDRICHHits = 1;
   static constexpr Int_t kMax_DRICHHits_MCParticle = 1;
   static constexpr Int_t kMaxEcalBarrelImagingHits = 260;
   static constexpr Int_t kMax_EcalBarrelImagingHits_contributions = 635;
   static constexpr Int_t kMaxEcalBarrelImagingHitsContributions = 635;
   static constexpr Int_t kMax_EcalBarrelImagingHitsContributions_particle = 635;
   static constexpr Int_t kMaxEcalBarrelScFiHits = 4487;
   static constexpr Int_t kMax_EcalBarrelScFiHits_contributions = 16734;
   static constexpr Int_t kMaxEcalBarrelScFiHitsContributions = 16734;
   static constexpr Int_t kMax_EcalBarrelScFiHitsContributions_particle = 16734;
   static constexpr Int_t kMaxEcalEndcapNHits = 85;
   static constexpr Int_t kMax_EcalEndcapNHits_contributions = 596;
   static constexpr Int_t kMaxEcalEndcapNHitsContributions = 596;
   static constexpr Int_t kMax_EcalEndcapNHitsContributions_particle = 596;
   static constexpr Int_t kMaxEcalEndcapPHits = 26;
   static constexpr Int_t kMax_EcalEndcapPHits_contributions = 269;
   static constexpr Int_t kMaxEcalEndcapPHitsContributions = 269;
   static constexpr Int_t kMax_EcalEndcapPHitsContributions_particle = 269;
   static constexpr Int_t kMaxEcalEndcapPInsertHits = 1;
   static constexpr Int_t kMax_EcalEndcapPInsertHits_contributions = 2;
   static constexpr Int_t kMaxEcalEndcapPInsertHitsContributions = 2;
   static constexpr Int_t kMax_EcalEndcapPInsertHitsContributions_particle = 2;
   static constexpr Int_t kMaxEcalFarForwardZDCHits = 1;
   static constexpr Int_t kMax_EcalFarForwardZDCHits_contributions = 1;
   static constexpr Int_t kMaxEcalFarForwardZDCHitsContributions = 1;
   static constexpr Int_t kMax_EcalFarForwardZDCHitsContributions_particle = 1;
   static constexpr Int_t kMaxEcalLumiSpecHits = 1;
   static constexpr Int_t kMax_EcalLumiSpecHits_contributions = 1;
   static constexpr Int_t kMaxEcalLumiSpecHitsContributions = 1;
   static constexpr Int_t kMax_EcalLumiSpecHitsContributions_particle = 1;
   static constexpr Int_t kMaxEventHeader = 1;
   static constexpr Int_t kMaxForwardOffMTrackerHits = 1;
   static constexpr Int_t kMax_ForwardOffMTrackerHits_MCParticle = 1;
   static constexpr Int_t kMaxForwardRomanPotHits = 1;
   static constexpr Int_t kMax_ForwardRomanPotHits_MCParticle = 1;
   static constexpr Int_t kMaxHcalBarrelHits = 244;
   static constexpr Int_t kMax_HcalBarrelHits_contributions = 2565;
   static constexpr Int_t kMaxHcalBarrelHitsContributions = 2565;
   static constexpr Int_t kMax_HcalBarrelHitsContributions_particle = 2565;
   static constexpr Int_t kMaxHcalEndcapNHits = 31;
   static constexpr Int_t kMax_HcalEndcapNHits_contributions = 95;
   static constexpr Int_t kMaxHcalEndcapNHitsContributions = 95;
   static constexpr Int_t kMax_HcalEndcapNHitsContributions_particle = 95;
   static constexpr Int_t kMaxHcalEndcapPInsertHits = 1;
   static constexpr Int_t kMax_HcalEndcapPInsertHits_contributions = 1;
   static constexpr Int_t kMaxHcalEndcapPInsertHitsContributions = 1;
   static constexpr Int_t kMax_HcalEndcapPInsertHitsContributions_particle = 1;
   static constexpr Int_t kMaxHcalFarForwardZDCHits = 1;
   static constexpr Int_t kMax_HcalFarForwardZDCHits_contributions = 1;
   static constexpr Int_t kMaxHcalFarForwardZDCHitsContributions = 1;
   static constexpr Int_t kMax_HcalFarForwardZDCHitsContributions_particle = 1;
   static constexpr Int_t kMaxLFHCALHits = 23;
   static constexpr Int_t kMax_LFHCALHits_contributions = 42;
   static constexpr Int_t kMaxLFHCALHitsContributions = 42;
   static constexpr Int_t kMax_LFHCALHitsContributions_particle = 42;
   static constexpr Int_t kMaxLumiDirectPCALHits = 1;
   static constexpr Int_t kMax_LumiDirectPCALHits_contributions = 1;
   static constexpr Int_t kMaxLumiDirectPCALHitsContributions = 1;
   static constexpr Int_t kMax_LumiDirectPCALHitsContributions_particle = 1;
   static constexpr Int_t kMaxLumiSpecTrackerHits = 1;
   static constexpr Int_t kMax_LumiSpecTrackerHits_MCParticle = 1;
   static constexpr Int_t kMaxMCParticles = 63;
   static constexpr Int_t kMax_MCParticles_parents = 62;
   static constexpr Int_t kMax_MCParticles_daughters = 62;
   static constexpr Int_t kMaxMPGDBarrelHits = 5;
   static constexpr Int_t kMax_MPGDBarrelHits_MCParticle = 5;
   static constexpr Int_t kMaxMPGDDIRCHits = 16;
   static constexpr Int_t kMax_MPGDDIRCHits_MCParticle = 16;
   static constexpr Int_t kMaxPFRICHHits = 1;
   static constexpr Int_t kMax_PFRICHHits_MCParticle = 1;
   static constexpr Int_t kMaxSiBarrelHits = 20;
   static constexpr Int_t kMax_SiBarrelHits_MCParticle = 20;
   static constexpr Int_t kMaxTaggerTrackerHits = 1;
   static constexpr Int_t kMax_TaggerTrackerHits_MCParticle = 1;
   static constexpr Int_t kMaxTOFBarrelHits = 8;
   static constexpr Int_t kMax_TOFBarrelHits_MCParticle = 8;
   static constexpr Int_t kMaxTOFEndcapHits = 1;
   static constexpr Int_t kMax_TOFEndcapHits_MCParticle = 1;
   static constexpr Int_t kMaxTrackerEndcapHits = 4;
   static constexpr Int_t kMax_TrackerEndcapHits_MCParticle = 4;
   static constexpr Int_t kMaxVertexBarrelHits = 26;
   static constexpr Int_t kMax_VertexBarrelHits_MCParticle = 26;
   static constexpr Int_t kMax_intMap = 1;
   static constexpr Int_t kMax_floatMap = 1;
   static constexpr Int_t kMax_stringMap = 1;
   static constexpr Int_t kMax_doubleMap = 1;

   // Declaration of leaf types
   Int_t           B0ECalHits_;
   ULong_t         B0ECalHits_cellID[kMaxB0ECalHits];   //[B0ECalHits_]
   Float_t         B0ECalHits_energy[kMaxB0ECalHits];   //[B0ECalHits_]
   Float_t         B0ECalHits_position_x[kMaxB0ECalHits];   //[B0ECalHits_]
   Float_t         B0ECalHits_position_y[kMaxB0ECalHits];   //[B0ECalHits_]
   Float_t         B0ECalHits_position_z[kMaxB0ECalHits];   //[B0ECalHits_]
   UInt_t          B0ECalHits_contributions_begin[kMaxB0ECalHits];   //[B0ECalHits_]
   UInt_t          B0ECalHits_contributions_end[kMaxB0ECalHits];   //[B0ECalHits_]
   Int_t           _B0ECalHits_contributions_;
   Int_t           _B0ECalHits_contributions_index[kMax_B0ECalHits_contributions];   //[_B0ECalHits_contributions_]
   UInt_t          _B0ECalHits_contributions_collectionID[kMax_B0ECalHits_contributions];   //[_B0ECalHits_contributions_]
   Int_t           B0ECalHitsContributions_;
   Int_t           B0ECalHitsContributions_PDG[kMaxB0ECalHitsContributions];   //[B0ECalHitsContributions_]
   Float_t         B0ECalHitsContributions_energy[kMaxB0ECalHitsContributions];   //[B0ECalHitsContributions_]
   Float_t         B0ECalHitsContributions_time[kMaxB0ECalHitsContributions];   //[B0ECalHitsContributions_]
   Float_t         B0ECalHitsContributions_stepPosition_x[kMaxB0ECalHitsContributions];   //[B0ECalHitsContributions_]
   Float_t         B0ECalHitsContributions_stepPosition_y[kMaxB0ECalHitsContributions];   //[B0ECalHitsContributions_]
   Float_t         B0ECalHitsContributions_stepPosition_z[kMaxB0ECalHitsContributions];   //[B0ECalHitsContributions_]
   Int_t           _B0ECalHitsContributions_particle_;
   Int_t           _B0ECalHitsContributions_particle_index[kMax_B0ECalHitsContributions_particle];   //[_B0ECalHitsContributions_particle_]
   UInt_t          _B0ECalHitsContributions_particle_collectionID[kMax_B0ECalHitsContributions_particle];   //[_B0ECalHitsContributions_particle_]
   Int_t           B0TrackerHits_;
   ULong_t         B0TrackerHits_cellID[kMaxB0TrackerHits];   //[B0TrackerHits_]
   Float_t         B0TrackerHits_EDep[kMaxB0TrackerHits];   //[B0TrackerHits_]
   Float_t         B0TrackerHits_time[kMaxB0TrackerHits];   //[B0TrackerHits_]
   Float_t         B0TrackerHits_pathLength[kMaxB0TrackerHits];   //[B0TrackerHits_]
   Int_t           B0TrackerHits_quality[kMaxB0TrackerHits];   //[B0TrackerHits_]
   Double_t        B0TrackerHits_position_x[kMaxB0TrackerHits];   //[B0TrackerHits_]
   Double_t        B0TrackerHits_position_y[kMaxB0TrackerHits];   //[B0TrackerHits_]
   Double_t        B0TrackerHits_position_z[kMaxB0TrackerHits];   //[B0TrackerHits_]
   Float_t         B0TrackerHits_momentum_x[kMaxB0TrackerHits];   //[B0TrackerHits_]
   Float_t         B0TrackerHits_momentum_y[kMaxB0TrackerHits];   //[B0TrackerHits_]
   Float_t         B0TrackerHits_momentum_z[kMaxB0TrackerHits];   //[B0TrackerHits_]
   Int_t           _B0TrackerHits_MCParticle_;
   Int_t           _B0TrackerHits_MCParticle_index[kMax_B0TrackerHits_MCParticle];   //[_B0TrackerHits_MCParticle_]
   UInt_t          _B0TrackerHits_MCParticle_collectionID[kMax_B0TrackerHits_MCParticle];   //[_B0TrackerHits_MCParticle_]
   Int_t           DIRCBarHits_;
   ULong_t         DIRCBarHits_cellID[kMaxDIRCBarHits];   //[DIRCBarHits_]
   Float_t         DIRCBarHits_EDep[kMaxDIRCBarHits];   //[DIRCBarHits_]
   Float_t         DIRCBarHits_time[kMaxDIRCBarHits];   //[DIRCBarHits_]
   Float_t         DIRCBarHits_pathLength[kMaxDIRCBarHits];   //[DIRCBarHits_]
   Int_t           DIRCBarHits_quality[kMaxDIRCBarHits];   //[DIRCBarHits_]
   Double_t        DIRCBarHits_position_x[kMaxDIRCBarHits];   //[DIRCBarHits_]
   Double_t        DIRCBarHits_position_y[kMaxDIRCBarHits];   //[DIRCBarHits_]
   Double_t        DIRCBarHits_position_z[kMaxDIRCBarHits];   //[DIRCBarHits_]
   Float_t         DIRCBarHits_momentum_x[kMaxDIRCBarHits];   //[DIRCBarHits_]
   Float_t         DIRCBarHits_momentum_y[kMaxDIRCBarHits];   //[DIRCBarHits_]
   Float_t         DIRCBarHits_momentum_z[kMaxDIRCBarHits];   //[DIRCBarHits_]
   Int_t           _DIRCBarHits_MCParticle_;
   Int_t           _DIRCBarHits_MCParticle_index[kMax_DIRCBarHits_MCParticle];   //[_DIRCBarHits_MCParticle_]
   UInt_t          _DIRCBarHits_MCParticle_collectionID[kMax_DIRCBarHits_MCParticle];   //[_DIRCBarHits_MCParticle_]
   Int_t           DRICHHits_;
   ULong_t         DRICHHits_cellID[kMaxDRICHHits];   //[DRICHHits_]
   Float_t         DRICHHits_EDep[kMaxDRICHHits];   //[DRICHHits_]
   Float_t         DRICHHits_time[kMaxDRICHHits];   //[DRICHHits_]
   Float_t         DRICHHits_pathLength[kMaxDRICHHits];   //[DRICHHits_]
   Int_t           DRICHHits_quality[kMaxDRICHHits];   //[DRICHHits_]
   Double_t        DRICHHits_position_x[kMaxDRICHHits];   //[DRICHHits_]
   Double_t        DRICHHits_position_y[kMaxDRICHHits];   //[DRICHHits_]
   Double_t        DRICHHits_position_z[kMaxDRICHHits];   //[DRICHHits_]
   Float_t         DRICHHits_momentum_x[kMaxDRICHHits];   //[DRICHHits_]
   Float_t         DRICHHits_momentum_y[kMaxDRICHHits];   //[DRICHHits_]
   Float_t         DRICHHits_momentum_z[kMaxDRICHHits];   //[DRICHHits_]
   Int_t           _DRICHHits_MCParticle_;
   Int_t           _DRICHHits_MCParticle_index[kMax_DRICHHits_MCParticle];   //[_DRICHHits_MCParticle_]
   UInt_t          _DRICHHits_MCParticle_collectionID[kMax_DRICHHits_MCParticle];   //[_DRICHHits_MCParticle_]
   Int_t           EcalBarrelImagingHits_;
   ULong_t         EcalBarrelImagingHits_cellID[kMaxEcalBarrelImagingHits];   //[EcalBarrelImagingHits_]
   Float_t         EcalBarrelImagingHits_energy[kMaxEcalBarrelImagingHits];   //[EcalBarrelImagingHits_]
   Float_t         EcalBarrelImagingHits_position_x[kMaxEcalBarrelImagingHits];   //[EcalBarrelImagingHits_]
   Float_t         EcalBarrelImagingHits_position_y[kMaxEcalBarrelImagingHits];   //[EcalBarrelImagingHits_]
   Float_t         EcalBarrelImagingHits_position_z[kMaxEcalBarrelImagingHits];   //[EcalBarrelImagingHits_]
   UInt_t          EcalBarrelImagingHits_contributions_begin[kMaxEcalBarrelImagingHits];   //[EcalBarrelImagingHits_]
   UInt_t          EcalBarrelImagingHits_contributions_end[kMaxEcalBarrelImagingHits];   //[EcalBarrelImagingHits_]
   Int_t           _EcalBarrelImagingHits_contributions_;
   Int_t           _EcalBarrelImagingHits_contributions_index[kMax_EcalBarrelImagingHits_contributions];   //[_EcalBarrelImagingHits_contributions_]
   UInt_t          _EcalBarrelImagingHits_contributions_collectionID[kMax_EcalBarrelImagingHits_contributions];   //[_EcalBarrelImagingHits_contributions_]
   Int_t           EcalBarrelImagingHitsContributions_;
   Int_t           EcalBarrelImagingHitsContributions_PDG[kMaxEcalBarrelImagingHitsContributions];   //[EcalBarrelImagingHitsContributions_]
   Float_t         EcalBarrelImagingHitsContributions_energy[kMaxEcalBarrelImagingHitsContributions];   //[EcalBarrelImagingHitsContributions_]
   Float_t         EcalBarrelImagingHitsContributions_time[kMaxEcalBarrelImagingHitsContributions];   //[EcalBarrelImagingHitsContributions_]
   Float_t         EcalBarrelImagingHitsContributions_stepPosition_x[kMaxEcalBarrelImagingHitsContributions];   //[EcalBarrelImagingHitsContributions_]
   Float_t         EcalBarrelImagingHitsContributions_stepPosition_y[kMaxEcalBarrelImagingHitsContributions];   //[EcalBarrelImagingHitsContributions_]
   Float_t         EcalBarrelImagingHitsContributions_stepPosition_z[kMaxEcalBarrelImagingHitsContributions];   //[EcalBarrelImagingHitsContributions_]
   Int_t           _EcalBarrelImagingHitsContributions_particle_;
   Int_t           _EcalBarrelImagingHitsContributions_particle_index[kMax_EcalBarrelImagingHitsContributions_particle];   //[_EcalBarrelImagingHitsContributions_particle_]
   UInt_t          _EcalBarrelImagingHitsContributions_particle_collectionID[kMax_EcalBarrelImagingHitsContributions_particle];   //[_EcalBarrelImagingHitsContributions_particle_]
   Int_t           EcalBarrelScFiHits_;
   ULong_t         EcalBarrelScFiHits_cellID[kMaxEcalBarrelScFiHits];   //[EcalBarrelScFiHits_]
   Float_t         EcalBarrelScFiHits_energy[kMaxEcalBarrelScFiHits];   //[EcalBarrelScFiHits_]
   Float_t         EcalBarrelScFiHits_position_x[kMaxEcalBarrelScFiHits];   //[EcalBarrelScFiHits_]
   Float_t         EcalBarrelScFiHits_position_y[kMaxEcalBarrelScFiHits];   //[EcalBarrelScFiHits_]
   Float_t         EcalBarrelScFiHits_position_z[kMaxEcalBarrelScFiHits];   //[EcalBarrelScFiHits_]
   UInt_t          EcalBarrelScFiHits_contributions_begin[kMaxEcalBarrelScFiHits];   //[EcalBarrelScFiHits_]
   UInt_t          EcalBarrelScFiHits_contributions_end[kMaxEcalBarrelScFiHits];   //[EcalBarrelScFiHits_]
   Int_t           _EcalBarrelScFiHits_contributions_;
   Int_t           _EcalBarrelScFiHits_contributions_index[kMax_EcalBarrelScFiHits_contributions];   //[_EcalBarrelScFiHits_contributions_]
   UInt_t          _EcalBarrelScFiHits_contributions_collectionID[kMax_EcalBarrelScFiHits_contributions];   //[_EcalBarrelScFiHits_contributions_]
   Int_t           EcalBarrelScFiHitsContributions_;
   Int_t           EcalBarrelScFiHitsContributions_PDG[kMaxEcalBarrelScFiHitsContributions];   //[EcalBarrelScFiHitsContributions_]
   Float_t         EcalBarrelScFiHitsContributions_energy[kMaxEcalBarrelScFiHitsContributions];   //[EcalBarrelScFiHitsContributions_]
   Float_t         EcalBarrelScFiHitsContributions_time[kMaxEcalBarrelScFiHitsContributions];   //[EcalBarrelScFiHitsContributions_]
   Float_t         EcalBarrelScFiHitsContributions_stepPosition_x[kMaxEcalBarrelScFiHitsContributions];   //[EcalBarrelScFiHitsContributions_]
   Float_t         EcalBarrelScFiHitsContributions_stepPosition_y[kMaxEcalBarrelScFiHitsContributions];   //[EcalBarrelScFiHitsContributions_]
   Float_t         EcalBarrelScFiHitsContributions_stepPosition_z[kMaxEcalBarrelScFiHitsContributions];   //[EcalBarrelScFiHitsContributions_]
   Int_t           _EcalBarrelScFiHitsContributions_particle_;
   Int_t           _EcalBarrelScFiHitsContributions_particle_index[kMax_EcalBarrelScFiHitsContributions_particle];   //[_EcalBarrelScFiHitsContributions_particle_]
   UInt_t          _EcalBarrelScFiHitsContributions_particle_collectionID[kMax_EcalBarrelScFiHitsContributions_particle];   //[_EcalBarrelScFiHitsContributions_particle_]
   Int_t           EcalEndcapNHits_;
   ULong_t         EcalEndcapNHits_cellID[kMaxEcalEndcapNHits];   //[EcalEndcapNHits_]
   Float_t         EcalEndcapNHits_energy[kMaxEcalEndcapNHits];   //[EcalEndcapNHits_]
   Float_t         EcalEndcapNHits_position_x[kMaxEcalEndcapNHits];   //[EcalEndcapNHits_]
   Float_t         EcalEndcapNHits_position_y[kMaxEcalEndcapNHits];   //[EcalEndcapNHits_]
   Float_t         EcalEndcapNHits_position_z[kMaxEcalEndcapNHits];   //[EcalEndcapNHits_]
   UInt_t          EcalEndcapNHits_contributions_begin[kMaxEcalEndcapNHits];   //[EcalEndcapNHits_]
   UInt_t          EcalEndcapNHits_contributions_end[kMaxEcalEndcapNHits];   //[EcalEndcapNHits_]
   Int_t           _EcalEndcapNHits_contributions_;
   Int_t           _EcalEndcapNHits_contributions_index[kMax_EcalEndcapNHits_contributions];   //[_EcalEndcapNHits_contributions_]
   UInt_t          _EcalEndcapNHits_contributions_collectionID[kMax_EcalEndcapNHits_contributions];   //[_EcalEndcapNHits_contributions_]
   Int_t           EcalEndcapNHitsContributions_;
   Int_t           EcalEndcapNHitsContributions_PDG[kMaxEcalEndcapNHitsContributions];   //[EcalEndcapNHitsContributions_]
   Float_t         EcalEndcapNHitsContributions_energy[kMaxEcalEndcapNHitsContributions];   //[EcalEndcapNHitsContributions_]
   Float_t         EcalEndcapNHitsContributions_time[kMaxEcalEndcapNHitsContributions];   //[EcalEndcapNHitsContributions_]
   Float_t         EcalEndcapNHitsContributions_stepPosition_x[kMaxEcalEndcapNHitsContributions];   //[EcalEndcapNHitsContributions_]
   Float_t         EcalEndcapNHitsContributions_stepPosition_y[kMaxEcalEndcapNHitsContributions];   //[EcalEndcapNHitsContributions_]
   Float_t         EcalEndcapNHitsContributions_stepPosition_z[kMaxEcalEndcapNHitsContributions];   //[EcalEndcapNHitsContributions_]
   Int_t           _EcalEndcapNHitsContributions_particle_;
   Int_t           _EcalEndcapNHitsContributions_particle_index[kMax_EcalEndcapNHitsContributions_particle];   //[_EcalEndcapNHitsContributions_particle_]
   UInt_t          _EcalEndcapNHitsContributions_particle_collectionID[kMax_EcalEndcapNHitsContributions_particle];   //[_EcalEndcapNHitsContributions_particle_]
   Int_t           EcalEndcapPHits_;
   ULong_t         EcalEndcapPHits_cellID[kMaxEcalEndcapPHits];   //[EcalEndcapPHits_]
   Float_t         EcalEndcapPHits_energy[kMaxEcalEndcapPHits];   //[EcalEndcapPHits_]
   Float_t         EcalEndcapPHits_position_x[kMaxEcalEndcapPHits];   //[EcalEndcapPHits_]
   Float_t         EcalEndcapPHits_position_y[kMaxEcalEndcapPHits];   //[EcalEndcapPHits_]
   Float_t         EcalEndcapPHits_position_z[kMaxEcalEndcapPHits];   //[EcalEndcapPHits_]
   UInt_t          EcalEndcapPHits_contributions_begin[kMaxEcalEndcapPHits];   //[EcalEndcapPHits_]
   UInt_t          EcalEndcapPHits_contributions_end[kMaxEcalEndcapPHits];   //[EcalEndcapPHits_]
   Int_t           _EcalEndcapPHits_contributions_;
   Int_t           _EcalEndcapPHits_contributions_index[kMax_EcalEndcapPHits_contributions];   //[_EcalEndcapPHits_contributions_]
   UInt_t          _EcalEndcapPHits_contributions_collectionID[kMax_EcalEndcapPHits_contributions];   //[_EcalEndcapPHits_contributions_]
   Int_t           EcalEndcapPHitsContributions_;
   Int_t           EcalEndcapPHitsContributions_PDG[kMaxEcalEndcapPHitsContributions];   //[EcalEndcapPHitsContributions_]
   Float_t         EcalEndcapPHitsContributions_energy[kMaxEcalEndcapPHitsContributions];   //[EcalEndcapPHitsContributions_]
   Float_t         EcalEndcapPHitsContributions_time[kMaxEcalEndcapPHitsContributions];   //[EcalEndcapPHitsContributions_]
   Float_t         EcalEndcapPHitsContributions_stepPosition_x[kMaxEcalEndcapPHitsContributions];   //[EcalEndcapPHitsContributions_]
   Float_t         EcalEndcapPHitsContributions_stepPosition_y[kMaxEcalEndcapPHitsContributions];   //[EcalEndcapPHitsContributions_]
   Float_t         EcalEndcapPHitsContributions_stepPosition_z[kMaxEcalEndcapPHitsContributions];   //[EcalEndcapPHitsContributions_]
   Int_t           _EcalEndcapPHitsContributions_particle_;
   Int_t           _EcalEndcapPHitsContributions_particle_index[kMax_EcalEndcapPHitsContributions_particle];   //[_EcalEndcapPHitsContributions_particle_]
   UInt_t          _EcalEndcapPHitsContributions_particle_collectionID[kMax_EcalEndcapPHitsContributions_particle];   //[_EcalEndcapPHitsContributions_particle_]
   Int_t           EcalEndcapPInsertHits_;
   ULong_t         EcalEndcapPInsertHits_cellID[kMaxEcalEndcapPInsertHits];   //[EcalEndcapPInsertHits_]
   Float_t         EcalEndcapPInsertHits_energy[kMaxEcalEndcapPInsertHits];   //[EcalEndcapPInsertHits_]
   Float_t         EcalEndcapPInsertHits_position_x[kMaxEcalEndcapPInsertHits];   //[EcalEndcapPInsertHits_]
   Float_t         EcalEndcapPInsertHits_position_y[kMaxEcalEndcapPInsertHits];   //[EcalEndcapPInsertHits_]
   Float_t         EcalEndcapPInsertHits_position_z[kMaxEcalEndcapPInsertHits];   //[EcalEndcapPInsertHits_]
   UInt_t          EcalEndcapPInsertHits_contributions_begin[kMaxEcalEndcapPInsertHits];   //[EcalEndcapPInsertHits_]
   UInt_t          EcalEndcapPInsertHits_contributions_end[kMaxEcalEndcapPInsertHits];   //[EcalEndcapPInsertHits_]
   Int_t           _EcalEndcapPInsertHits_contributions_;
   Int_t           _EcalEndcapPInsertHits_contributions_index[kMax_EcalEndcapPInsertHits_contributions];   //[_EcalEndcapPInsertHits_contributions_]
   UInt_t          _EcalEndcapPInsertHits_contributions_collectionID[kMax_EcalEndcapPInsertHits_contributions];   //[_EcalEndcapPInsertHits_contributions_]
   Int_t           EcalEndcapPInsertHitsContributions_;
   Int_t           EcalEndcapPInsertHitsContributions_PDG[kMaxEcalEndcapPInsertHitsContributions];   //[EcalEndcapPInsertHitsContributions_]
   Float_t         EcalEndcapPInsertHitsContributions_energy[kMaxEcalEndcapPInsertHitsContributions];   //[EcalEndcapPInsertHitsContributions_]
   Float_t         EcalEndcapPInsertHitsContributions_time[kMaxEcalEndcapPInsertHitsContributions];   //[EcalEndcapPInsertHitsContributions_]
   Float_t         EcalEndcapPInsertHitsContributions_stepPosition_x[kMaxEcalEndcapPInsertHitsContributions];   //[EcalEndcapPInsertHitsContributions_]
   Float_t         EcalEndcapPInsertHitsContributions_stepPosition_y[kMaxEcalEndcapPInsertHitsContributions];   //[EcalEndcapPInsertHitsContributions_]
   Float_t         EcalEndcapPInsertHitsContributions_stepPosition_z[kMaxEcalEndcapPInsertHitsContributions];   //[EcalEndcapPInsertHitsContributions_]
   Int_t           _EcalEndcapPInsertHitsContributions_particle_;
   Int_t           _EcalEndcapPInsertHitsContributions_particle_index[kMax_EcalEndcapPInsertHitsContributions_particle];   //[_EcalEndcapPInsertHitsContributions_particle_]
   UInt_t          _EcalEndcapPInsertHitsContributions_particle_collectionID[kMax_EcalEndcapPInsertHitsContributions_particle];   //[_EcalEndcapPInsertHitsContributions_particle_]
   Int_t           EcalFarForwardZDCHits_;
   ULong_t         EcalFarForwardZDCHits_cellID[kMaxEcalFarForwardZDCHits];   //[EcalFarForwardZDCHits_]
   Float_t         EcalFarForwardZDCHits_energy[kMaxEcalFarForwardZDCHits];   //[EcalFarForwardZDCHits_]
   Float_t         EcalFarForwardZDCHits_position_x[kMaxEcalFarForwardZDCHits];   //[EcalFarForwardZDCHits_]
   Float_t         EcalFarForwardZDCHits_position_y[kMaxEcalFarForwardZDCHits];   //[EcalFarForwardZDCHits_]
   Float_t         EcalFarForwardZDCHits_position_z[kMaxEcalFarForwardZDCHits];   //[EcalFarForwardZDCHits_]
   UInt_t          EcalFarForwardZDCHits_contributions_begin[kMaxEcalFarForwardZDCHits];   //[EcalFarForwardZDCHits_]
   UInt_t          EcalFarForwardZDCHits_contributions_end[kMaxEcalFarForwardZDCHits];   //[EcalFarForwardZDCHits_]
   Int_t           _EcalFarForwardZDCHits_contributions_;
   Int_t           _EcalFarForwardZDCHits_contributions_index[kMax_EcalFarForwardZDCHits_contributions];   //[_EcalFarForwardZDCHits_contributions_]
   UInt_t          _EcalFarForwardZDCHits_contributions_collectionID[kMax_EcalFarForwardZDCHits_contributions];   //[_EcalFarForwardZDCHits_contributions_]
   Int_t           EcalFarForwardZDCHitsContributions_;
   Int_t           EcalFarForwardZDCHitsContributions_PDG[kMaxEcalFarForwardZDCHitsContributions];   //[EcalFarForwardZDCHitsContributions_]
   Float_t         EcalFarForwardZDCHitsContributions_energy[kMaxEcalFarForwardZDCHitsContributions];   //[EcalFarForwardZDCHitsContributions_]
   Float_t         EcalFarForwardZDCHitsContributions_time[kMaxEcalFarForwardZDCHitsContributions];   //[EcalFarForwardZDCHitsContributions_]
   Float_t         EcalFarForwardZDCHitsContributions_stepPosition_x[kMaxEcalFarForwardZDCHitsContributions];   //[EcalFarForwardZDCHitsContributions_]
   Float_t         EcalFarForwardZDCHitsContributions_stepPosition_y[kMaxEcalFarForwardZDCHitsContributions];   //[EcalFarForwardZDCHitsContributions_]
   Float_t         EcalFarForwardZDCHitsContributions_stepPosition_z[kMaxEcalFarForwardZDCHitsContributions];   //[EcalFarForwardZDCHitsContributions_]
   Int_t           _EcalFarForwardZDCHitsContributions_particle_;
   Int_t           _EcalFarForwardZDCHitsContributions_particle_index[kMax_EcalFarForwardZDCHitsContributions_particle];   //[_EcalFarForwardZDCHitsContributions_particle_]
   UInt_t          _EcalFarForwardZDCHitsContributions_particle_collectionID[kMax_EcalFarForwardZDCHitsContributions_particle];   //[_EcalFarForwardZDCHitsContributions_particle_]
   Int_t           EcalLumiSpecHits_;
   ULong_t         EcalLumiSpecHits_cellID[kMaxEcalLumiSpecHits];   //[EcalLumiSpecHits_]
   Float_t         EcalLumiSpecHits_energy[kMaxEcalLumiSpecHits];   //[EcalLumiSpecHits_]
   Float_t         EcalLumiSpecHits_position_x[kMaxEcalLumiSpecHits];   //[EcalLumiSpecHits_]
   Float_t         EcalLumiSpecHits_position_y[kMaxEcalLumiSpecHits];   //[EcalLumiSpecHits_]
   Float_t         EcalLumiSpecHits_position_z[kMaxEcalLumiSpecHits];   //[EcalLumiSpecHits_]
   UInt_t          EcalLumiSpecHits_contributions_begin[kMaxEcalLumiSpecHits];   //[EcalLumiSpecHits_]
   UInt_t          EcalLumiSpecHits_contributions_end[kMaxEcalLumiSpecHits];   //[EcalLumiSpecHits_]
   Int_t           _EcalLumiSpecHits_contributions_;
   Int_t           _EcalLumiSpecHits_contributions_index[kMax_EcalLumiSpecHits_contributions];   //[_EcalLumiSpecHits_contributions_]
   UInt_t          _EcalLumiSpecHits_contributions_collectionID[kMax_EcalLumiSpecHits_contributions];   //[_EcalLumiSpecHits_contributions_]
   Int_t           EcalLumiSpecHitsContributions_;
   Int_t           EcalLumiSpecHitsContributions_PDG[kMaxEcalLumiSpecHitsContributions];   //[EcalLumiSpecHitsContributions_]
   Float_t         EcalLumiSpecHitsContributions_energy[kMaxEcalLumiSpecHitsContributions];   //[EcalLumiSpecHitsContributions_]
   Float_t         EcalLumiSpecHitsContributions_time[kMaxEcalLumiSpecHitsContributions];   //[EcalLumiSpecHitsContributions_]
   Float_t         EcalLumiSpecHitsContributions_stepPosition_x[kMaxEcalLumiSpecHitsContributions];   //[EcalLumiSpecHitsContributions_]
   Float_t         EcalLumiSpecHitsContributions_stepPosition_y[kMaxEcalLumiSpecHitsContributions];   //[EcalLumiSpecHitsContributions_]
   Float_t         EcalLumiSpecHitsContributions_stepPosition_z[kMaxEcalLumiSpecHitsContributions];   //[EcalLumiSpecHitsContributions_]
   Int_t           _EcalLumiSpecHitsContributions_particle_;
   Int_t           _EcalLumiSpecHitsContributions_particle_index[kMax_EcalLumiSpecHitsContributions_particle];   //[_EcalLumiSpecHitsContributions_particle_]
   UInt_t          _EcalLumiSpecHitsContributions_particle_collectionID[kMax_EcalLumiSpecHitsContributions_particle];   //[_EcalLumiSpecHitsContributions_particle_]
   Int_t           EventHeader_;
   Int_t           EventHeader_eventNumber[kMaxEventHeader];   //[EventHeader_]
   Int_t           EventHeader_runNumber[kMaxEventHeader];   //[EventHeader_]
   ULong_t         EventHeader_timeStamp[kMaxEventHeader];   //[EventHeader_]
   Float_t         EventHeader_weight[kMaxEventHeader];   //[EventHeader_]
   Int_t           ForwardOffMTrackerHits_;
   ULong_t         ForwardOffMTrackerHits_cellID[kMaxForwardOffMTrackerHits];   //[ForwardOffMTrackerHits_]
   Float_t         ForwardOffMTrackerHits_EDep[kMaxForwardOffMTrackerHits];   //[ForwardOffMTrackerHits_]
   Float_t         ForwardOffMTrackerHits_time[kMaxForwardOffMTrackerHits];   //[ForwardOffMTrackerHits_]
   Float_t         ForwardOffMTrackerHits_pathLength[kMaxForwardOffMTrackerHits];   //[ForwardOffMTrackerHits_]
   Int_t           ForwardOffMTrackerHits_quality[kMaxForwardOffMTrackerHits];   //[ForwardOffMTrackerHits_]
   Double_t        ForwardOffMTrackerHits_position_x[kMaxForwardOffMTrackerHits];   //[ForwardOffMTrackerHits_]
   Double_t        ForwardOffMTrackerHits_position_y[kMaxForwardOffMTrackerHits];   //[ForwardOffMTrackerHits_]
   Double_t        ForwardOffMTrackerHits_position_z[kMaxForwardOffMTrackerHits];   //[ForwardOffMTrackerHits_]
   Float_t         ForwardOffMTrackerHits_momentum_x[kMaxForwardOffMTrackerHits];   //[ForwardOffMTrackerHits_]
   Float_t         ForwardOffMTrackerHits_momentum_y[kMaxForwardOffMTrackerHits];   //[ForwardOffMTrackerHits_]
   Float_t         ForwardOffMTrackerHits_momentum_z[kMaxForwardOffMTrackerHits];   //[ForwardOffMTrackerHits_]
   Int_t           _ForwardOffMTrackerHits_MCParticle_;
   Int_t           _ForwardOffMTrackerHits_MCParticle_index[kMax_ForwardOffMTrackerHits_MCParticle];   //[_ForwardOffMTrackerHits_MCParticle_]
   UInt_t          _ForwardOffMTrackerHits_MCParticle_collectionID[kMax_ForwardOffMTrackerHits_MCParticle];   //[_ForwardOffMTrackerHits_MCParticle_]
   Int_t           ForwardRomanPotHits_;
   ULong_t         ForwardRomanPotHits_cellID[kMaxForwardRomanPotHits];   //[ForwardRomanPotHits_]
   Float_t         ForwardRomanPotHits_EDep[kMaxForwardRomanPotHits];   //[ForwardRomanPotHits_]
   Float_t         ForwardRomanPotHits_time[kMaxForwardRomanPotHits];   //[ForwardRomanPotHits_]
   Float_t         ForwardRomanPotHits_pathLength[kMaxForwardRomanPotHits];   //[ForwardRomanPotHits_]
   Int_t           ForwardRomanPotHits_quality[kMaxForwardRomanPotHits];   //[ForwardRomanPotHits_]
   Double_t        ForwardRomanPotHits_position_x[kMaxForwardRomanPotHits];   //[ForwardRomanPotHits_]
   Double_t        ForwardRomanPotHits_position_y[kMaxForwardRomanPotHits];   //[ForwardRomanPotHits_]
   Double_t        ForwardRomanPotHits_position_z[kMaxForwardRomanPotHits];   //[ForwardRomanPotHits_]
   Float_t         ForwardRomanPotHits_momentum_x[kMaxForwardRomanPotHits];   //[ForwardRomanPotHits_]
   Float_t         ForwardRomanPotHits_momentum_y[kMaxForwardRomanPotHits];   //[ForwardRomanPotHits_]
   Float_t         ForwardRomanPotHits_momentum_z[kMaxForwardRomanPotHits];   //[ForwardRomanPotHits_]
   Int_t           _ForwardRomanPotHits_MCParticle_;
   Int_t           _ForwardRomanPotHits_MCParticle_index[kMax_ForwardRomanPotHits_MCParticle];   //[_ForwardRomanPotHits_MCParticle_]
   UInt_t          _ForwardRomanPotHits_MCParticle_collectionID[kMax_ForwardRomanPotHits_MCParticle];   //[_ForwardRomanPotHits_MCParticle_]
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
   Int_t           HcalEndcapNHits_;
   ULong_t         HcalEndcapNHits_cellID[kMaxHcalEndcapNHits];   //[HcalEndcapNHits_]
   Float_t         HcalEndcapNHits_energy[kMaxHcalEndcapNHits];   //[HcalEndcapNHits_]
   Float_t         HcalEndcapNHits_position_x[kMaxHcalEndcapNHits];   //[HcalEndcapNHits_]
   Float_t         HcalEndcapNHits_position_y[kMaxHcalEndcapNHits];   //[HcalEndcapNHits_]
   Float_t         HcalEndcapNHits_position_z[kMaxHcalEndcapNHits];   //[HcalEndcapNHits_]
   UInt_t          HcalEndcapNHits_contributions_begin[kMaxHcalEndcapNHits];   //[HcalEndcapNHits_]
   UInt_t          HcalEndcapNHits_contributions_end[kMaxHcalEndcapNHits];   //[HcalEndcapNHits_]
   Int_t           _HcalEndcapNHits_contributions_;
   Int_t           _HcalEndcapNHits_contributions_index[kMax_HcalEndcapNHits_contributions];   //[_HcalEndcapNHits_contributions_]
   UInt_t          _HcalEndcapNHits_contributions_collectionID[kMax_HcalEndcapNHits_contributions];   //[_HcalEndcapNHits_contributions_]
   Int_t           HcalEndcapNHitsContributions_;
   Int_t           HcalEndcapNHitsContributions_PDG[kMaxHcalEndcapNHitsContributions];   //[HcalEndcapNHitsContributions_]
   Float_t         HcalEndcapNHitsContributions_energy[kMaxHcalEndcapNHitsContributions];   //[HcalEndcapNHitsContributions_]
   Float_t         HcalEndcapNHitsContributions_time[kMaxHcalEndcapNHitsContributions];   //[HcalEndcapNHitsContributions_]
   Float_t         HcalEndcapNHitsContributions_stepPosition_x[kMaxHcalEndcapNHitsContributions];   //[HcalEndcapNHitsContributions_]
   Float_t         HcalEndcapNHitsContributions_stepPosition_y[kMaxHcalEndcapNHitsContributions];   //[HcalEndcapNHitsContributions_]
   Float_t         HcalEndcapNHitsContributions_stepPosition_z[kMaxHcalEndcapNHitsContributions];   //[HcalEndcapNHitsContributions_]
   Int_t           _HcalEndcapNHitsContributions_particle_;
   Int_t           _HcalEndcapNHitsContributions_particle_index[kMax_HcalEndcapNHitsContributions_particle];   //[_HcalEndcapNHitsContributions_particle_]
   UInt_t          _HcalEndcapNHitsContributions_particle_collectionID[kMax_HcalEndcapNHitsContributions_particle];   //[_HcalEndcapNHitsContributions_particle_]
   Int_t           HcalEndcapPInsertHits_;
   ULong_t         HcalEndcapPInsertHits_cellID[kMaxHcalEndcapPInsertHits];   //[HcalEndcapPInsertHits_]
   Float_t         HcalEndcapPInsertHits_energy[kMaxHcalEndcapPInsertHits];   //[HcalEndcapPInsertHits_]
   Float_t         HcalEndcapPInsertHits_position_x[kMaxHcalEndcapPInsertHits];   //[HcalEndcapPInsertHits_]
   Float_t         HcalEndcapPInsertHits_position_y[kMaxHcalEndcapPInsertHits];   //[HcalEndcapPInsertHits_]
   Float_t         HcalEndcapPInsertHits_position_z[kMaxHcalEndcapPInsertHits];   //[HcalEndcapPInsertHits_]
   UInt_t          HcalEndcapPInsertHits_contributions_begin[kMaxHcalEndcapPInsertHits];   //[HcalEndcapPInsertHits_]
   UInt_t          HcalEndcapPInsertHits_contributions_end[kMaxHcalEndcapPInsertHits];   //[HcalEndcapPInsertHits_]
   Int_t           _HcalEndcapPInsertHits_contributions_;
   Int_t           _HcalEndcapPInsertHits_contributions_index[kMax_HcalEndcapPInsertHits_contributions];   //[_HcalEndcapPInsertHits_contributions_]
   UInt_t          _HcalEndcapPInsertHits_contributions_collectionID[kMax_HcalEndcapPInsertHits_contributions];   //[_HcalEndcapPInsertHits_contributions_]
   Int_t           HcalEndcapPInsertHitsContributions_;
   Int_t           HcalEndcapPInsertHitsContributions_PDG[kMaxHcalEndcapPInsertHitsContributions];   //[HcalEndcapPInsertHitsContributions_]
   Float_t         HcalEndcapPInsertHitsContributions_energy[kMaxHcalEndcapPInsertHitsContributions];   //[HcalEndcapPInsertHitsContributions_]
   Float_t         HcalEndcapPInsertHitsContributions_time[kMaxHcalEndcapPInsertHitsContributions];   //[HcalEndcapPInsertHitsContributions_]
   Float_t         HcalEndcapPInsertHitsContributions_stepPosition_x[kMaxHcalEndcapPInsertHitsContributions];   //[HcalEndcapPInsertHitsContributions_]
   Float_t         HcalEndcapPInsertHitsContributions_stepPosition_y[kMaxHcalEndcapPInsertHitsContributions];   //[HcalEndcapPInsertHitsContributions_]
   Float_t         HcalEndcapPInsertHitsContributions_stepPosition_z[kMaxHcalEndcapPInsertHitsContributions];   //[HcalEndcapPInsertHitsContributions_]
   Int_t           _HcalEndcapPInsertHitsContributions_particle_;
   Int_t           _HcalEndcapPInsertHitsContributions_particle_index[kMax_HcalEndcapPInsertHitsContributions_particle];   //[_HcalEndcapPInsertHitsContributions_particle_]
   UInt_t          _HcalEndcapPInsertHitsContributions_particle_collectionID[kMax_HcalEndcapPInsertHitsContributions_particle];   //[_HcalEndcapPInsertHitsContributions_particle_]
   Int_t           HcalFarForwardZDCHits_;
   ULong_t         HcalFarForwardZDCHits_cellID[kMaxHcalFarForwardZDCHits];   //[HcalFarForwardZDCHits_]
   Float_t         HcalFarForwardZDCHits_energy[kMaxHcalFarForwardZDCHits];   //[HcalFarForwardZDCHits_]
   Float_t         HcalFarForwardZDCHits_position_x[kMaxHcalFarForwardZDCHits];   //[HcalFarForwardZDCHits_]
   Float_t         HcalFarForwardZDCHits_position_y[kMaxHcalFarForwardZDCHits];   //[HcalFarForwardZDCHits_]
   Float_t         HcalFarForwardZDCHits_position_z[kMaxHcalFarForwardZDCHits];   //[HcalFarForwardZDCHits_]
   UInt_t          HcalFarForwardZDCHits_contributions_begin[kMaxHcalFarForwardZDCHits];   //[HcalFarForwardZDCHits_]
   UInt_t          HcalFarForwardZDCHits_contributions_end[kMaxHcalFarForwardZDCHits];   //[HcalFarForwardZDCHits_]
   Int_t           _HcalFarForwardZDCHits_contributions_;
   Int_t           _HcalFarForwardZDCHits_contributions_index[kMax_HcalFarForwardZDCHits_contributions];   //[_HcalFarForwardZDCHits_contributions_]
   UInt_t          _HcalFarForwardZDCHits_contributions_collectionID[kMax_HcalFarForwardZDCHits_contributions];   //[_HcalFarForwardZDCHits_contributions_]
   Int_t           HcalFarForwardZDCHitsContributions_;
   Int_t           HcalFarForwardZDCHitsContributions_PDG[kMaxHcalFarForwardZDCHitsContributions];   //[HcalFarForwardZDCHitsContributions_]
   Float_t         HcalFarForwardZDCHitsContributions_energy[kMaxHcalFarForwardZDCHitsContributions];   //[HcalFarForwardZDCHitsContributions_]
   Float_t         HcalFarForwardZDCHitsContributions_time[kMaxHcalFarForwardZDCHitsContributions];   //[HcalFarForwardZDCHitsContributions_]
   Float_t         HcalFarForwardZDCHitsContributions_stepPosition_x[kMaxHcalFarForwardZDCHitsContributions];   //[HcalFarForwardZDCHitsContributions_]
   Float_t         HcalFarForwardZDCHitsContributions_stepPosition_y[kMaxHcalFarForwardZDCHitsContributions];   //[HcalFarForwardZDCHitsContributions_]
   Float_t         HcalFarForwardZDCHitsContributions_stepPosition_z[kMaxHcalFarForwardZDCHitsContributions];   //[HcalFarForwardZDCHitsContributions_]
   Int_t           _HcalFarForwardZDCHitsContributions_particle_;
   Int_t           _HcalFarForwardZDCHitsContributions_particle_index[kMax_HcalFarForwardZDCHitsContributions_particle];   //[_HcalFarForwardZDCHitsContributions_particle_]
   UInt_t          _HcalFarForwardZDCHitsContributions_particle_collectionID[kMax_HcalFarForwardZDCHitsContributions_particle];   //[_HcalFarForwardZDCHitsContributions_particle_]
   Int_t           LFHCALHits_;
   ULong_t         LFHCALHits_cellID[kMaxLFHCALHits];   //[LFHCALHits_]
   Float_t         LFHCALHits_energy[kMaxLFHCALHits];   //[LFHCALHits_]
   Float_t         LFHCALHits_position_x[kMaxLFHCALHits];   //[LFHCALHits_]
   Float_t         LFHCALHits_position_y[kMaxLFHCALHits];   //[LFHCALHits_]
   Float_t         LFHCALHits_position_z[kMaxLFHCALHits];   //[LFHCALHits_]
   UInt_t          LFHCALHits_contributions_begin[kMaxLFHCALHits];   //[LFHCALHits_]
   UInt_t          LFHCALHits_contributions_end[kMaxLFHCALHits];   //[LFHCALHits_]
   Int_t           _LFHCALHits_contributions_;
   Int_t           _LFHCALHits_contributions_index[kMax_LFHCALHits_contributions];   //[_LFHCALHits_contributions_]
   UInt_t          _LFHCALHits_contributions_collectionID[kMax_LFHCALHits_contributions];   //[_LFHCALHits_contributions_]
   Int_t           LFHCALHitsContributions_;
   Int_t           LFHCALHitsContributions_PDG[kMaxLFHCALHitsContributions];   //[LFHCALHitsContributions_]
   Float_t         LFHCALHitsContributions_energy[kMaxLFHCALHitsContributions];   //[LFHCALHitsContributions_]
   Float_t         LFHCALHitsContributions_time[kMaxLFHCALHitsContributions];   //[LFHCALHitsContributions_]
   Float_t         LFHCALHitsContributions_stepPosition_x[kMaxLFHCALHitsContributions];   //[LFHCALHitsContributions_]
   Float_t         LFHCALHitsContributions_stepPosition_y[kMaxLFHCALHitsContributions];   //[LFHCALHitsContributions_]
   Float_t         LFHCALHitsContributions_stepPosition_z[kMaxLFHCALHitsContributions];   //[LFHCALHitsContributions_]
   Int_t           _LFHCALHitsContributions_particle_;
   Int_t           _LFHCALHitsContributions_particle_index[kMax_LFHCALHitsContributions_particle];   //[_LFHCALHitsContributions_particle_]
   UInt_t          _LFHCALHitsContributions_particle_collectionID[kMax_LFHCALHitsContributions_particle];   //[_LFHCALHitsContributions_particle_]
   Int_t           LumiDirectPCALHits_;
   ULong_t         LumiDirectPCALHits_cellID[kMaxLumiDirectPCALHits];   //[LumiDirectPCALHits_]
   Float_t         LumiDirectPCALHits_energy[kMaxLumiDirectPCALHits];   //[LumiDirectPCALHits_]
   Float_t         LumiDirectPCALHits_position_x[kMaxLumiDirectPCALHits];   //[LumiDirectPCALHits_]
   Float_t         LumiDirectPCALHits_position_y[kMaxLumiDirectPCALHits];   //[LumiDirectPCALHits_]
   Float_t         LumiDirectPCALHits_position_z[kMaxLumiDirectPCALHits];   //[LumiDirectPCALHits_]
   UInt_t          LumiDirectPCALHits_contributions_begin[kMaxLumiDirectPCALHits];   //[LumiDirectPCALHits_]
   UInt_t          LumiDirectPCALHits_contributions_end[kMaxLumiDirectPCALHits];   //[LumiDirectPCALHits_]
   Int_t           _LumiDirectPCALHits_contributions_;
   Int_t           _LumiDirectPCALHits_contributions_index[kMax_LumiDirectPCALHits_contributions];   //[_LumiDirectPCALHits_contributions_]
   UInt_t          _LumiDirectPCALHits_contributions_collectionID[kMax_LumiDirectPCALHits_contributions];   //[_LumiDirectPCALHits_contributions_]
   Int_t           LumiDirectPCALHitsContributions_;
   Int_t           LumiDirectPCALHitsContributions_PDG[kMaxLumiDirectPCALHitsContributions];   //[LumiDirectPCALHitsContributions_]
   Float_t         LumiDirectPCALHitsContributions_energy[kMaxLumiDirectPCALHitsContributions];   //[LumiDirectPCALHitsContributions_]
   Float_t         LumiDirectPCALHitsContributions_time[kMaxLumiDirectPCALHitsContributions];   //[LumiDirectPCALHitsContributions_]
   Float_t         LumiDirectPCALHitsContributions_stepPosition_x[kMaxLumiDirectPCALHitsContributions];   //[LumiDirectPCALHitsContributions_]
   Float_t         LumiDirectPCALHitsContributions_stepPosition_y[kMaxLumiDirectPCALHitsContributions];   //[LumiDirectPCALHitsContributions_]
   Float_t         LumiDirectPCALHitsContributions_stepPosition_z[kMaxLumiDirectPCALHitsContributions];   //[LumiDirectPCALHitsContributions_]
   Int_t           _LumiDirectPCALHitsContributions_particle_;
   Int_t           _LumiDirectPCALHitsContributions_particle_index[kMax_LumiDirectPCALHitsContributions_particle];   //[_LumiDirectPCALHitsContributions_particle_]
   UInt_t          _LumiDirectPCALHitsContributions_particle_collectionID[kMax_LumiDirectPCALHitsContributions_particle];   //[_LumiDirectPCALHitsContributions_particle_]
   Int_t           LumiSpecTrackerHits_;
   ULong_t         LumiSpecTrackerHits_cellID[kMaxLumiSpecTrackerHits];   //[LumiSpecTrackerHits_]
   Float_t         LumiSpecTrackerHits_EDep[kMaxLumiSpecTrackerHits];   //[LumiSpecTrackerHits_]
   Float_t         LumiSpecTrackerHits_time[kMaxLumiSpecTrackerHits];   //[LumiSpecTrackerHits_]
   Float_t         LumiSpecTrackerHits_pathLength[kMaxLumiSpecTrackerHits];   //[LumiSpecTrackerHits_]
   Int_t           LumiSpecTrackerHits_quality[kMaxLumiSpecTrackerHits];   //[LumiSpecTrackerHits_]
   Double_t        LumiSpecTrackerHits_position_x[kMaxLumiSpecTrackerHits];   //[LumiSpecTrackerHits_]
   Double_t        LumiSpecTrackerHits_position_y[kMaxLumiSpecTrackerHits];   //[LumiSpecTrackerHits_]
   Double_t        LumiSpecTrackerHits_position_z[kMaxLumiSpecTrackerHits];   //[LumiSpecTrackerHits_]
   Float_t         LumiSpecTrackerHits_momentum_x[kMaxLumiSpecTrackerHits];   //[LumiSpecTrackerHits_]
   Float_t         LumiSpecTrackerHits_momentum_y[kMaxLumiSpecTrackerHits];   //[LumiSpecTrackerHits_]
   Float_t         LumiSpecTrackerHits_momentum_z[kMaxLumiSpecTrackerHits];   //[LumiSpecTrackerHits_]
   Int_t           _LumiSpecTrackerHits_MCParticle_;
   Int_t           _LumiSpecTrackerHits_MCParticle_index[kMax_LumiSpecTrackerHits_MCParticle];   //[_LumiSpecTrackerHits_MCParticle_]
   UInt_t          _LumiSpecTrackerHits_MCParticle_collectionID[kMax_LumiSpecTrackerHits_MCParticle];   //[_LumiSpecTrackerHits_MCParticle_]
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
   Int_t           MPGDBarrelHits_;
   ULong_t         MPGDBarrelHits_cellID[kMaxMPGDBarrelHits];   //[MPGDBarrelHits_]
   Float_t         MPGDBarrelHits_EDep[kMaxMPGDBarrelHits];   //[MPGDBarrelHits_]
   Float_t         MPGDBarrelHits_time[kMaxMPGDBarrelHits];   //[MPGDBarrelHits_]
   Float_t         MPGDBarrelHits_pathLength[kMaxMPGDBarrelHits];   //[MPGDBarrelHits_]
   Int_t           MPGDBarrelHits_quality[kMaxMPGDBarrelHits];   //[MPGDBarrelHits_]
   Double_t        MPGDBarrelHits_position_x[kMaxMPGDBarrelHits];   //[MPGDBarrelHits_]
   Double_t        MPGDBarrelHits_position_y[kMaxMPGDBarrelHits];   //[MPGDBarrelHits_]
   Double_t        MPGDBarrelHits_position_z[kMaxMPGDBarrelHits];   //[MPGDBarrelHits_]
   Float_t         MPGDBarrelHits_momentum_x[kMaxMPGDBarrelHits];   //[MPGDBarrelHits_]
   Float_t         MPGDBarrelHits_momentum_y[kMaxMPGDBarrelHits];   //[MPGDBarrelHits_]
   Float_t         MPGDBarrelHits_momentum_z[kMaxMPGDBarrelHits];   //[MPGDBarrelHits_]
   Int_t           _MPGDBarrelHits_MCParticle_;
   Int_t           _MPGDBarrelHits_MCParticle_index[kMax_MPGDBarrelHits_MCParticle];   //[_MPGDBarrelHits_MCParticle_]
   UInt_t          _MPGDBarrelHits_MCParticle_collectionID[kMax_MPGDBarrelHits_MCParticle];   //[_MPGDBarrelHits_MCParticle_]
   Int_t           MPGDDIRCHits_;
   ULong_t         MPGDDIRCHits_cellID[kMaxMPGDDIRCHits];   //[MPGDDIRCHits_]
   Float_t         MPGDDIRCHits_EDep[kMaxMPGDDIRCHits];   //[MPGDDIRCHits_]
   Float_t         MPGDDIRCHits_time[kMaxMPGDDIRCHits];   //[MPGDDIRCHits_]
   Float_t         MPGDDIRCHits_pathLength[kMaxMPGDDIRCHits];   //[MPGDDIRCHits_]
   Int_t           MPGDDIRCHits_quality[kMaxMPGDDIRCHits];   //[MPGDDIRCHits_]
   Double_t        MPGDDIRCHits_position_x[kMaxMPGDDIRCHits];   //[MPGDDIRCHits_]
   Double_t        MPGDDIRCHits_position_y[kMaxMPGDDIRCHits];   //[MPGDDIRCHits_]
   Double_t        MPGDDIRCHits_position_z[kMaxMPGDDIRCHits];   //[MPGDDIRCHits_]
   Float_t         MPGDDIRCHits_momentum_x[kMaxMPGDDIRCHits];   //[MPGDDIRCHits_]
   Float_t         MPGDDIRCHits_momentum_y[kMaxMPGDDIRCHits];   //[MPGDDIRCHits_]
   Float_t         MPGDDIRCHits_momentum_z[kMaxMPGDDIRCHits];   //[MPGDDIRCHits_]
   Int_t           _MPGDDIRCHits_MCParticle_;
   Int_t           _MPGDDIRCHits_MCParticle_index[kMax_MPGDDIRCHits_MCParticle];   //[_MPGDDIRCHits_MCParticle_]
   UInt_t          _MPGDDIRCHits_MCParticle_collectionID[kMax_MPGDDIRCHits_MCParticle];   //[_MPGDDIRCHits_MCParticle_]
   Int_t           PFRICHHits_;
   ULong_t         PFRICHHits_cellID[kMaxPFRICHHits];   //[PFRICHHits_]
   Float_t         PFRICHHits_EDep[kMaxPFRICHHits];   //[PFRICHHits_]
   Float_t         PFRICHHits_time[kMaxPFRICHHits];   //[PFRICHHits_]
   Float_t         PFRICHHits_pathLength[kMaxPFRICHHits];   //[PFRICHHits_]
   Int_t           PFRICHHits_quality[kMaxPFRICHHits];   //[PFRICHHits_]
   Double_t        PFRICHHits_position_x[kMaxPFRICHHits];   //[PFRICHHits_]
   Double_t        PFRICHHits_position_y[kMaxPFRICHHits];   //[PFRICHHits_]
   Double_t        PFRICHHits_position_z[kMaxPFRICHHits];   //[PFRICHHits_]
   Float_t         PFRICHHits_momentum_x[kMaxPFRICHHits];   //[PFRICHHits_]
   Float_t         PFRICHHits_momentum_y[kMaxPFRICHHits];   //[PFRICHHits_]
   Float_t         PFRICHHits_momentum_z[kMaxPFRICHHits];   //[PFRICHHits_]
   Int_t           _PFRICHHits_MCParticle_;
   Int_t           _PFRICHHits_MCParticle_index[kMax_PFRICHHits_MCParticle];   //[_PFRICHHits_MCParticle_]
   UInt_t          _PFRICHHits_MCParticle_collectionID[kMax_PFRICHHits_MCParticle];   //[_PFRICHHits_MCParticle_]
   Int_t           SiBarrelHits_;
   ULong_t         SiBarrelHits_cellID[kMaxSiBarrelHits];   //[SiBarrelHits_]
   Float_t         SiBarrelHits_EDep[kMaxSiBarrelHits];   //[SiBarrelHits_]
   Float_t         SiBarrelHits_time[kMaxSiBarrelHits];   //[SiBarrelHits_]
   Float_t         SiBarrelHits_pathLength[kMaxSiBarrelHits];   //[SiBarrelHits_]
   Int_t           SiBarrelHits_quality[kMaxSiBarrelHits];   //[SiBarrelHits_]
   Double_t        SiBarrelHits_position_x[kMaxSiBarrelHits];   //[SiBarrelHits_]
   Double_t        SiBarrelHits_position_y[kMaxSiBarrelHits];   //[SiBarrelHits_]
   Double_t        SiBarrelHits_position_z[kMaxSiBarrelHits];   //[SiBarrelHits_]
   Float_t         SiBarrelHits_momentum_x[kMaxSiBarrelHits];   //[SiBarrelHits_]
   Float_t         SiBarrelHits_momentum_y[kMaxSiBarrelHits];   //[SiBarrelHits_]
   Float_t         SiBarrelHits_momentum_z[kMaxSiBarrelHits];   //[SiBarrelHits_]
   Int_t           _SiBarrelHits_MCParticle_;
   Int_t           _SiBarrelHits_MCParticle_index[kMax_SiBarrelHits_MCParticle];   //[_SiBarrelHits_MCParticle_]
   UInt_t          _SiBarrelHits_MCParticle_collectionID[kMax_SiBarrelHits_MCParticle];   //[_SiBarrelHits_MCParticle_]
   Int_t           TaggerTrackerHits_;
   ULong_t         TaggerTrackerHits_cellID[kMaxTaggerTrackerHits];   //[TaggerTrackerHits_]
   Float_t         TaggerTrackerHits_EDep[kMaxTaggerTrackerHits];   //[TaggerTrackerHits_]
   Float_t         TaggerTrackerHits_time[kMaxTaggerTrackerHits];   //[TaggerTrackerHits_]
   Float_t         TaggerTrackerHits_pathLength[kMaxTaggerTrackerHits];   //[TaggerTrackerHits_]
   Int_t           TaggerTrackerHits_quality[kMaxTaggerTrackerHits];   //[TaggerTrackerHits_]
   Double_t        TaggerTrackerHits_position_x[kMaxTaggerTrackerHits];   //[TaggerTrackerHits_]
   Double_t        TaggerTrackerHits_position_y[kMaxTaggerTrackerHits];   //[TaggerTrackerHits_]
   Double_t        TaggerTrackerHits_position_z[kMaxTaggerTrackerHits];   //[TaggerTrackerHits_]
   Float_t         TaggerTrackerHits_momentum_x[kMaxTaggerTrackerHits];   //[TaggerTrackerHits_]
   Float_t         TaggerTrackerHits_momentum_y[kMaxTaggerTrackerHits];   //[TaggerTrackerHits_]
   Float_t         TaggerTrackerHits_momentum_z[kMaxTaggerTrackerHits];   //[TaggerTrackerHits_]
   Int_t           _TaggerTrackerHits_MCParticle_;
   Int_t           _TaggerTrackerHits_MCParticle_index[kMax_TaggerTrackerHits_MCParticle];   //[_TaggerTrackerHits_MCParticle_]
   UInt_t          _TaggerTrackerHits_MCParticle_collectionID[kMax_TaggerTrackerHits_MCParticle];   //[_TaggerTrackerHits_MCParticle_]
   Int_t           TOFBarrelHits_;
   ULong_t         TOFBarrelHits_cellID[kMaxTOFBarrelHits];   //[TOFBarrelHits_]
   Float_t         TOFBarrelHits_EDep[kMaxTOFBarrelHits];   //[TOFBarrelHits_]
   Float_t         TOFBarrelHits_time[kMaxTOFBarrelHits];   //[TOFBarrelHits_]
   Float_t         TOFBarrelHits_pathLength[kMaxTOFBarrelHits];   //[TOFBarrelHits_]
   Int_t           TOFBarrelHits_quality[kMaxTOFBarrelHits];   //[TOFBarrelHits_]
   Double_t        TOFBarrelHits_position_x[kMaxTOFBarrelHits];   //[TOFBarrelHits_]
   Double_t        TOFBarrelHits_position_y[kMaxTOFBarrelHits];   //[TOFBarrelHits_]
   Double_t        TOFBarrelHits_position_z[kMaxTOFBarrelHits];   //[TOFBarrelHits_]
   Float_t         TOFBarrelHits_momentum_x[kMaxTOFBarrelHits];   //[TOFBarrelHits_]
   Float_t         TOFBarrelHits_momentum_y[kMaxTOFBarrelHits];   //[TOFBarrelHits_]
   Float_t         TOFBarrelHits_momentum_z[kMaxTOFBarrelHits];   //[TOFBarrelHits_]
   Int_t           _TOFBarrelHits_MCParticle_;
   Int_t           _TOFBarrelHits_MCParticle_index[kMax_TOFBarrelHits_MCParticle];   //[_TOFBarrelHits_MCParticle_]
   UInt_t          _TOFBarrelHits_MCParticle_collectionID[kMax_TOFBarrelHits_MCParticle];   //[_TOFBarrelHits_MCParticle_]
   Int_t           TOFEndcapHits_;
   ULong_t         TOFEndcapHits_cellID[kMaxTOFEndcapHits];   //[TOFEndcapHits_]
   Float_t         TOFEndcapHits_EDep[kMaxTOFEndcapHits];   //[TOFEndcapHits_]
   Float_t         TOFEndcapHits_time[kMaxTOFEndcapHits];   //[TOFEndcapHits_]
   Float_t         TOFEndcapHits_pathLength[kMaxTOFEndcapHits];   //[TOFEndcapHits_]
   Int_t           TOFEndcapHits_quality[kMaxTOFEndcapHits];   //[TOFEndcapHits_]
   Double_t        TOFEndcapHits_position_x[kMaxTOFEndcapHits];   //[TOFEndcapHits_]
   Double_t        TOFEndcapHits_position_y[kMaxTOFEndcapHits];   //[TOFEndcapHits_]
   Double_t        TOFEndcapHits_position_z[kMaxTOFEndcapHits];   //[TOFEndcapHits_]
   Float_t         TOFEndcapHits_momentum_x[kMaxTOFEndcapHits];   //[TOFEndcapHits_]
   Float_t         TOFEndcapHits_momentum_y[kMaxTOFEndcapHits];   //[TOFEndcapHits_]
   Float_t         TOFEndcapHits_momentum_z[kMaxTOFEndcapHits];   //[TOFEndcapHits_]
   Int_t           _TOFEndcapHits_MCParticle_;
   Int_t           _TOFEndcapHits_MCParticle_index[kMax_TOFEndcapHits_MCParticle];   //[_TOFEndcapHits_MCParticle_]
   UInt_t          _TOFEndcapHits_MCParticle_collectionID[kMax_TOFEndcapHits_MCParticle];   //[_TOFEndcapHits_MCParticle_]
   Int_t           TrackerEndcapHits_;
   ULong_t         TrackerEndcapHits_cellID[kMaxTrackerEndcapHits];   //[TrackerEndcapHits_]
   Float_t         TrackerEndcapHits_EDep[kMaxTrackerEndcapHits];   //[TrackerEndcapHits_]
   Float_t         TrackerEndcapHits_time[kMaxTrackerEndcapHits];   //[TrackerEndcapHits_]
   Float_t         TrackerEndcapHits_pathLength[kMaxTrackerEndcapHits];   //[TrackerEndcapHits_]
   Int_t           TrackerEndcapHits_quality[kMaxTrackerEndcapHits];   //[TrackerEndcapHits_]
   Double_t        TrackerEndcapHits_position_x[kMaxTrackerEndcapHits];   //[TrackerEndcapHits_]
   Double_t        TrackerEndcapHits_position_y[kMaxTrackerEndcapHits];   //[TrackerEndcapHits_]
   Double_t        TrackerEndcapHits_position_z[kMaxTrackerEndcapHits];   //[TrackerEndcapHits_]
   Float_t         TrackerEndcapHits_momentum_x[kMaxTrackerEndcapHits];   //[TrackerEndcapHits_]
   Float_t         TrackerEndcapHits_momentum_y[kMaxTrackerEndcapHits];   //[TrackerEndcapHits_]
   Float_t         TrackerEndcapHits_momentum_z[kMaxTrackerEndcapHits];   //[TrackerEndcapHits_]
   Int_t           _TrackerEndcapHits_MCParticle_;
   Int_t           _TrackerEndcapHits_MCParticle_index[kMax_TrackerEndcapHits_MCParticle];   //[_TrackerEndcapHits_MCParticle_]
   UInt_t          _TrackerEndcapHits_MCParticle_collectionID[kMax_TrackerEndcapHits_MCParticle];   //[_TrackerEndcapHits_MCParticle_]
   Int_t           VertexBarrelHits_;
   ULong_t         VertexBarrelHits_cellID[kMaxVertexBarrelHits];   //[VertexBarrelHits_]
   Float_t         VertexBarrelHits_EDep[kMaxVertexBarrelHits];   //[VertexBarrelHits_]
   Float_t         VertexBarrelHits_time[kMaxVertexBarrelHits];   //[VertexBarrelHits_]
   Float_t         VertexBarrelHits_pathLength[kMaxVertexBarrelHits];   //[VertexBarrelHits_]
   Int_t           VertexBarrelHits_quality[kMaxVertexBarrelHits];   //[VertexBarrelHits_]
   Double_t        VertexBarrelHits_position_x[kMaxVertexBarrelHits];   //[VertexBarrelHits_]
   Double_t        VertexBarrelHits_position_y[kMaxVertexBarrelHits];   //[VertexBarrelHits_]
   Double_t        VertexBarrelHits_position_z[kMaxVertexBarrelHits];   //[VertexBarrelHits_]
   Float_t         VertexBarrelHits_momentum_x[kMaxVertexBarrelHits];   //[VertexBarrelHits_]
   Float_t         VertexBarrelHits_momentum_y[kMaxVertexBarrelHits];   //[VertexBarrelHits_]
   Float_t         VertexBarrelHits_momentum_z[kMaxVertexBarrelHits];   //[VertexBarrelHits_]
   Int_t           _VertexBarrelHits_MCParticle_;
   Int_t           _VertexBarrelHits_MCParticle_index[kMax_VertexBarrelHits_MCParticle];   //[_VertexBarrelHits_MCParticle_]
   UInt_t          _VertexBarrelHits_MCParticle_collectionID[kMax_VertexBarrelHits_MCParticle];   //[_VertexBarrelHits_MCParticle_]
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
   TBranch        *b_B0ECalHits_;   //!
   TBranch        *b_B0ECalHits_cellID;   //!
   TBranch        *b_B0ECalHits_energy;   //!
   TBranch        *b_B0ECalHits_position_x;   //!
   TBranch        *b_B0ECalHits_position_y;   //!
   TBranch        *b_B0ECalHits_position_z;   //!
   TBranch        *b_B0ECalHits_contributions_begin;   //!
   TBranch        *b_B0ECalHits_contributions_end;   //!
   TBranch        *b__B0ECalHits_contributions_;   //!
   TBranch        *b__B0ECalHits_contributions_index;   //!
   TBranch        *b__B0ECalHits_contributions_collectionID;   //!
   TBranch        *b_B0ECalHitsContributions_;   //!
   TBranch        *b_B0ECalHitsContributions_PDG;   //!
   TBranch        *b_B0ECalHitsContributions_energy;   //!
   TBranch        *b_B0ECalHitsContributions_time;   //!
   TBranch        *b_B0ECalHitsContributions_stepPosition_x;   //!
   TBranch        *b_B0ECalHitsContributions_stepPosition_y;   //!
   TBranch        *b_B0ECalHitsContributions_stepPosition_z;   //!
   TBranch        *b__B0ECalHitsContributions_particle_;   //!
   TBranch        *b__B0ECalHitsContributions_particle_index;   //!
   TBranch        *b__B0ECalHitsContributions_particle_collectionID;   //!
   TBranch        *b_B0TrackerHits_;   //!
   TBranch        *b_B0TrackerHits_cellID;   //!
   TBranch        *b_B0TrackerHits_EDep;   //!
   TBranch        *b_B0TrackerHits_time;   //!
   TBranch        *b_B0TrackerHits_pathLength;   //!
   TBranch        *b_B0TrackerHits_quality;   //!
   TBranch        *b_B0TrackerHits_position_x;   //!
   TBranch        *b_B0TrackerHits_position_y;   //!
   TBranch        *b_B0TrackerHits_position_z;   //!
   TBranch        *b_B0TrackerHits_momentum_x;   //!
   TBranch        *b_B0TrackerHits_momentum_y;   //!
   TBranch        *b_B0TrackerHits_momentum_z;   //!
   TBranch        *b__B0TrackerHits_MCParticle_;   //!
   TBranch        *b__B0TrackerHits_MCParticle_index;   //!
   TBranch        *b__B0TrackerHits_MCParticle_collectionID;   //!
   TBranch        *b_DIRCBarHits_;   //!
   TBranch        *b_DIRCBarHits_cellID;   //!
   TBranch        *b_DIRCBarHits_EDep;   //!
   TBranch        *b_DIRCBarHits_time;   //!
   TBranch        *b_DIRCBarHits_pathLength;   //!
   TBranch        *b_DIRCBarHits_quality;   //!
   TBranch        *b_DIRCBarHits_position_x;   //!
   TBranch        *b_DIRCBarHits_position_y;   //!
   TBranch        *b_DIRCBarHits_position_z;   //!
   TBranch        *b_DIRCBarHits_momentum_x;   //!
   TBranch        *b_DIRCBarHits_momentum_y;   //!
   TBranch        *b_DIRCBarHits_momentum_z;   //!
   TBranch        *b__DIRCBarHits_MCParticle_;   //!
   TBranch        *b__DIRCBarHits_MCParticle_index;   //!
   TBranch        *b__DIRCBarHits_MCParticle_collectionID;   //!
   TBranch        *b_DRICHHits_;   //!
   TBranch        *b_DRICHHits_cellID;   //!
   TBranch        *b_DRICHHits_EDep;   //!
   TBranch        *b_DRICHHits_time;   //!
   TBranch        *b_DRICHHits_pathLength;   //!
   TBranch        *b_DRICHHits_quality;   //!
   TBranch        *b_DRICHHits_position_x;   //!
   TBranch        *b_DRICHHits_position_y;   //!
   TBranch        *b_DRICHHits_position_z;   //!
   TBranch        *b_DRICHHits_momentum_x;   //!
   TBranch        *b_DRICHHits_momentum_y;   //!
   TBranch        *b_DRICHHits_momentum_z;   //!
   TBranch        *b__DRICHHits_MCParticle_;   //!
   TBranch        *b__DRICHHits_MCParticle_index;   //!
   TBranch        *b__DRICHHits_MCParticle_collectionID;   //!
   TBranch        *b_EcalBarrelImagingHits_;   //!
   TBranch        *b_EcalBarrelImagingHits_cellID;   //!
   TBranch        *b_EcalBarrelImagingHits_energy;   //!
   TBranch        *b_EcalBarrelImagingHits_position_x;   //!
   TBranch        *b_EcalBarrelImagingHits_position_y;   //!
   TBranch        *b_EcalBarrelImagingHits_position_z;   //!
   TBranch        *b_EcalBarrelImagingHits_contributions_begin;   //!
   TBranch        *b_EcalBarrelImagingHits_contributions_end;   //!
   TBranch        *b__EcalBarrelImagingHits_contributions_;   //!
   TBranch        *b__EcalBarrelImagingHits_contributions_index;   //!
   TBranch        *b__EcalBarrelImagingHits_contributions_collectionID;   //!
   TBranch        *b_EcalBarrelImagingHitsContributions_;   //!
   TBranch        *b_EcalBarrelImagingHitsContributions_PDG;   //!
   TBranch        *b_EcalBarrelImagingHitsContributions_energy;   //!
   TBranch        *b_EcalBarrelImagingHitsContributions_time;   //!
   TBranch        *b_EcalBarrelImagingHitsContributions_stepPosition_x;   //!
   TBranch        *b_EcalBarrelImagingHitsContributions_stepPosition_y;   //!
   TBranch        *b_EcalBarrelImagingHitsContributions_stepPosition_z;   //!
   TBranch        *b__EcalBarrelImagingHitsContributions_particle_;   //!
   TBranch        *b__EcalBarrelImagingHitsContributions_particle_index;   //!
   TBranch        *b__EcalBarrelImagingHitsContributions_particle_collectionID;   //!
   TBranch        *b_EcalBarrelScFiHits_;   //!
   TBranch        *b_EcalBarrelScFiHits_cellID;   //!
   TBranch        *b_EcalBarrelScFiHits_energy;   //!
   TBranch        *b_EcalBarrelScFiHits_position_x;   //!
   TBranch        *b_EcalBarrelScFiHits_position_y;   //!
   TBranch        *b_EcalBarrelScFiHits_position_z;   //!
   TBranch        *b_EcalBarrelScFiHits_contributions_begin;   //!
   TBranch        *b_EcalBarrelScFiHits_contributions_end;   //!
   TBranch        *b__EcalBarrelScFiHits_contributions_;   //!
   TBranch        *b__EcalBarrelScFiHits_contributions_index;   //!
   TBranch        *b__EcalBarrelScFiHits_contributions_collectionID;   //!
   TBranch        *b_EcalBarrelScFiHitsContributions_;   //!
   TBranch        *b_EcalBarrelScFiHitsContributions_PDG;   //!
   TBranch        *b_EcalBarrelScFiHitsContributions_energy;   //!
   TBranch        *b_EcalBarrelScFiHitsContributions_time;   //!
   TBranch        *b_EcalBarrelScFiHitsContributions_stepPosition_x;   //!
   TBranch        *b_EcalBarrelScFiHitsContributions_stepPosition_y;   //!
   TBranch        *b_EcalBarrelScFiHitsContributions_stepPosition_z;   //!
   TBranch        *b__EcalBarrelScFiHitsContributions_particle_;   //!
   TBranch        *b__EcalBarrelScFiHitsContributions_particle_index;   //!
   TBranch        *b__EcalBarrelScFiHitsContributions_particle_collectionID;   //!
   TBranch        *b_EcalEndcapNHits_;   //!
   TBranch        *b_EcalEndcapNHits_cellID;   //!
   TBranch        *b_EcalEndcapNHits_energy;   //!
   TBranch        *b_EcalEndcapNHits_position_x;   //!
   TBranch        *b_EcalEndcapNHits_position_y;   //!
   TBranch        *b_EcalEndcapNHits_position_z;   //!
   TBranch        *b_EcalEndcapNHits_contributions_begin;   //!
   TBranch        *b_EcalEndcapNHits_contributions_end;   //!
   TBranch        *b__EcalEndcapNHits_contributions_;   //!
   TBranch        *b__EcalEndcapNHits_contributions_index;   //!
   TBranch        *b__EcalEndcapNHits_contributions_collectionID;   //!
   TBranch        *b_EcalEndcapNHitsContributions_;   //!
   TBranch        *b_EcalEndcapNHitsContributions_PDG;   //!
   TBranch        *b_EcalEndcapNHitsContributions_energy;   //!
   TBranch        *b_EcalEndcapNHitsContributions_time;   //!
   TBranch        *b_EcalEndcapNHitsContributions_stepPosition_x;   //!
   TBranch        *b_EcalEndcapNHitsContributions_stepPosition_y;   //!
   TBranch        *b_EcalEndcapNHitsContributions_stepPosition_z;   //!
   TBranch        *b__EcalEndcapNHitsContributions_particle_;   //!
   TBranch        *b__EcalEndcapNHitsContributions_particle_index;   //!
   TBranch        *b__EcalEndcapNHitsContributions_particle_collectionID;   //!
   TBranch        *b_EcalEndcapPHits_;   //!
   TBranch        *b_EcalEndcapPHits_cellID;   //!
   TBranch        *b_EcalEndcapPHits_energy;   //!
   TBranch        *b_EcalEndcapPHits_position_x;   //!
   TBranch        *b_EcalEndcapPHits_position_y;   //!
   TBranch        *b_EcalEndcapPHits_position_z;   //!
   TBranch        *b_EcalEndcapPHits_contributions_begin;   //!
   TBranch        *b_EcalEndcapPHits_contributions_end;   //!
   TBranch        *b__EcalEndcapPHits_contributions_;   //!
   TBranch        *b__EcalEndcapPHits_contributions_index;   //!
   TBranch        *b__EcalEndcapPHits_contributions_collectionID;   //!
   TBranch        *b_EcalEndcapPHitsContributions_;   //!
   TBranch        *b_EcalEndcapPHitsContributions_PDG;   //!
   TBranch        *b_EcalEndcapPHitsContributions_energy;   //!
   TBranch        *b_EcalEndcapPHitsContributions_time;   //!
   TBranch        *b_EcalEndcapPHitsContributions_stepPosition_x;   //!
   TBranch        *b_EcalEndcapPHitsContributions_stepPosition_y;   //!
   TBranch        *b_EcalEndcapPHitsContributions_stepPosition_z;   //!
   TBranch        *b__EcalEndcapPHitsContributions_particle_;   //!
   TBranch        *b__EcalEndcapPHitsContributions_particle_index;   //!
   TBranch        *b__EcalEndcapPHitsContributions_particle_collectionID;   //!
   TBranch        *b_EcalEndcapPInsertHits_;   //!
   TBranch        *b_EcalEndcapPInsertHits_cellID;   //!
   TBranch        *b_EcalEndcapPInsertHits_energy;   //!
   TBranch        *b_EcalEndcapPInsertHits_position_x;   //!
   TBranch        *b_EcalEndcapPInsertHits_position_y;   //!
   TBranch        *b_EcalEndcapPInsertHits_position_z;   //!
   TBranch        *b_EcalEndcapPInsertHits_contributions_begin;   //!
   TBranch        *b_EcalEndcapPInsertHits_contributions_end;   //!
   TBranch        *b__EcalEndcapPInsertHits_contributions_;   //!
   TBranch        *b__EcalEndcapPInsertHits_contributions_index;   //!
   TBranch        *b__EcalEndcapPInsertHits_contributions_collectionID;   //!
   TBranch        *b_EcalEndcapPInsertHitsContributions_;   //!
   TBranch        *b_EcalEndcapPInsertHitsContributions_PDG;   //!
   TBranch        *b_EcalEndcapPInsertHitsContributions_energy;   //!
   TBranch        *b_EcalEndcapPInsertHitsContributions_time;   //!
   TBranch        *b_EcalEndcapPInsertHitsContributions_stepPosition_x;   //!
   TBranch        *b_EcalEndcapPInsertHitsContributions_stepPosition_y;   //!
   TBranch        *b_EcalEndcapPInsertHitsContributions_stepPosition_z;   //!
   TBranch        *b__EcalEndcapPInsertHitsContributions_particle_;   //!
   TBranch        *b__EcalEndcapPInsertHitsContributions_particle_index;   //!
   TBranch        *b__EcalEndcapPInsertHitsContributions_particle_collectionID;   //!
   TBranch        *b_EcalFarForwardZDCHits_;   //!
   TBranch        *b_EcalFarForwardZDCHits_cellID;   //!
   TBranch        *b_EcalFarForwardZDCHits_energy;   //!
   TBranch        *b_EcalFarForwardZDCHits_position_x;   //!
   TBranch        *b_EcalFarForwardZDCHits_position_y;   //!
   TBranch        *b_EcalFarForwardZDCHits_position_z;   //!
   TBranch        *b_EcalFarForwardZDCHits_contributions_begin;   //!
   TBranch        *b_EcalFarForwardZDCHits_contributions_end;   //!
   TBranch        *b__EcalFarForwardZDCHits_contributions_;   //!
   TBranch        *b__EcalFarForwardZDCHits_contributions_index;   //!
   TBranch        *b__EcalFarForwardZDCHits_contributions_collectionID;   //!
   TBranch        *b_EcalFarForwardZDCHitsContributions_;   //!
   TBranch        *b_EcalFarForwardZDCHitsContributions_PDG;   //!
   TBranch        *b_EcalFarForwardZDCHitsContributions_energy;   //!
   TBranch        *b_EcalFarForwardZDCHitsContributions_time;   //!
   TBranch        *b_EcalFarForwardZDCHitsContributions_stepPosition_x;   //!
   TBranch        *b_EcalFarForwardZDCHitsContributions_stepPosition_y;   //!
   TBranch        *b_EcalFarForwardZDCHitsContributions_stepPosition_z;   //!
   TBranch        *b__EcalFarForwardZDCHitsContributions_particle_;   //!
   TBranch        *b__EcalFarForwardZDCHitsContributions_particle_index;   //!
   TBranch        *b__EcalFarForwardZDCHitsContributions_particle_collectionID;   //!
   TBranch        *b_EcalLumiSpecHits_;   //!
   TBranch        *b_EcalLumiSpecHits_cellID;   //!
   TBranch        *b_EcalLumiSpecHits_energy;   //!
   TBranch        *b_EcalLumiSpecHits_position_x;   //!
   TBranch        *b_EcalLumiSpecHits_position_y;   //!
   TBranch        *b_EcalLumiSpecHits_position_z;   //!
   TBranch        *b_EcalLumiSpecHits_contributions_begin;   //!
   TBranch        *b_EcalLumiSpecHits_contributions_end;   //!
   TBranch        *b__EcalLumiSpecHits_contributions_;   //!
   TBranch        *b__EcalLumiSpecHits_contributions_index;   //!
   TBranch        *b__EcalLumiSpecHits_contributions_collectionID;   //!
   TBranch        *b_EcalLumiSpecHitsContributions_;   //!
   TBranch        *b_EcalLumiSpecHitsContributions_PDG;   //!
   TBranch        *b_EcalLumiSpecHitsContributions_energy;   //!
   TBranch        *b_EcalLumiSpecHitsContributions_time;   //!
   TBranch        *b_EcalLumiSpecHitsContributions_stepPosition_x;   //!
   TBranch        *b_EcalLumiSpecHitsContributions_stepPosition_y;   //!
   TBranch        *b_EcalLumiSpecHitsContributions_stepPosition_z;   //!
   TBranch        *b__EcalLumiSpecHitsContributions_particle_;   //!
   TBranch        *b__EcalLumiSpecHitsContributions_particle_index;   //!
   TBranch        *b__EcalLumiSpecHitsContributions_particle_collectionID;   //!
   TBranch        *b_EventHeader_;   //!
   TBranch        *b_EventHeader_eventNumber;   //!
   TBranch        *b_EventHeader_runNumber;   //!
   TBranch        *b_EventHeader_timeStamp;   //!
   TBranch        *b_EventHeader_weight;   //!
   TBranch        *b_ForwardOffMTrackerHits_;   //!
   TBranch        *b_ForwardOffMTrackerHits_cellID;   //!
   TBranch        *b_ForwardOffMTrackerHits_EDep;   //!
   TBranch        *b_ForwardOffMTrackerHits_time;   //!
   TBranch        *b_ForwardOffMTrackerHits_pathLength;   //!
   TBranch        *b_ForwardOffMTrackerHits_quality;   //!
   TBranch        *b_ForwardOffMTrackerHits_position_x;   //!
   TBranch        *b_ForwardOffMTrackerHits_position_y;   //!
   TBranch        *b_ForwardOffMTrackerHits_position_z;   //!
   TBranch        *b_ForwardOffMTrackerHits_momentum_x;   //!
   TBranch        *b_ForwardOffMTrackerHits_momentum_y;   //!
   TBranch        *b_ForwardOffMTrackerHits_momentum_z;   //!
   TBranch        *b__ForwardOffMTrackerHits_MCParticle_;   //!
   TBranch        *b__ForwardOffMTrackerHits_MCParticle_index;   //!
   TBranch        *b__ForwardOffMTrackerHits_MCParticle_collectionID;   //!
   TBranch        *b_ForwardRomanPotHits_;   //!
   TBranch        *b_ForwardRomanPotHits_cellID;   //!
   TBranch        *b_ForwardRomanPotHits_EDep;   //!
   TBranch        *b_ForwardRomanPotHits_time;   //!
   TBranch        *b_ForwardRomanPotHits_pathLength;   //!
   TBranch        *b_ForwardRomanPotHits_quality;   //!
   TBranch        *b_ForwardRomanPotHits_position_x;   //!
   TBranch        *b_ForwardRomanPotHits_position_y;   //!
   TBranch        *b_ForwardRomanPotHits_position_z;   //!
   TBranch        *b_ForwardRomanPotHits_momentum_x;   //!
   TBranch        *b_ForwardRomanPotHits_momentum_y;   //!
   TBranch        *b_ForwardRomanPotHits_momentum_z;   //!
   TBranch        *b__ForwardRomanPotHits_MCParticle_;   //!
   TBranch        *b__ForwardRomanPotHits_MCParticle_index;   //!
   TBranch        *b__ForwardRomanPotHits_MCParticle_collectionID;   //!
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
   TBranch        *b_HcalEndcapNHits_;   //!
   TBranch        *b_HcalEndcapNHits_cellID;   //!
   TBranch        *b_HcalEndcapNHits_energy;   //!
   TBranch        *b_HcalEndcapNHits_position_x;   //!
   TBranch        *b_HcalEndcapNHits_position_y;   //!
   TBranch        *b_HcalEndcapNHits_position_z;   //!
   TBranch        *b_HcalEndcapNHits_contributions_begin;   //!
   TBranch        *b_HcalEndcapNHits_contributions_end;   //!
   TBranch        *b__HcalEndcapNHits_contributions_;   //!
   TBranch        *b__HcalEndcapNHits_contributions_index;   //!
   TBranch        *b__HcalEndcapNHits_contributions_collectionID;   //!
   TBranch        *b_HcalEndcapNHitsContributions_;   //!
   TBranch        *b_HcalEndcapNHitsContributions_PDG;   //!
   TBranch        *b_HcalEndcapNHitsContributions_energy;   //!
   TBranch        *b_HcalEndcapNHitsContributions_time;   //!
   TBranch        *b_HcalEndcapNHitsContributions_stepPosition_x;   //!
   TBranch        *b_HcalEndcapNHitsContributions_stepPosition_y;   //!
   TBranch        *b_HcalEndcapNHitsContributions_stepPosition_z;   //!
   TBranch        *b__HcalEndcapNHitsContributions_particle_;   //!
   TBranch        *b__HcalEndcapNHitsContributions_particle_index;   //!
   TBranch        *b__HcalEndcapNHitsContributions_particle_collectionID;   //!
   TBranch        *b_HcalEndcapPInsertHits_;   //!
   TBranch        *b_HcalEndcapPInsertHits_cellID;   //!
   TBranch        *b_HcalEndcapPInsertHits_energy;   //!
   TBranch        *b_HcalEndcapPInsertHits_position_x;   //!
   TBranch        *b_HcalEndcapPInsertHits_position_y;   //!
   TBranch        *b_HcalEndcapPInsertHits_position_z;   //!
   TBranch        *b_HcalEndcapPInsertHits_contributions_begin;   //!
   TBranch        *b_HcalEndcapPInsertHits_contributions_end;   //!
   TBranch        *b__HcalEndcapPInsertHits_contributions_;   //!
   TBranch        *b__HcalEndcapPInsertHits_contributions_index;   //!
   TBranch        *b__HcalEndcapPInsertHits_contributions_collectionID;   //!
   TBranch        *b_HcalEndcapPInsertHitsContributions_;   //!
   TBranch        *b_HcalEndcapPInsertHitsContributions_PDG;   //!
   TBranch        *b_HcalEndcapPInsertHitsContributions_energy;   //!
   TBranch        *b_HcalEndcapPInsertHitsContributions_time;   //!
   TBranch        *b_HcalEndcapPInsertHitsContributions_stepPosition_x;   //!
   TBranch        *b_HcalEndcapPInsertHitsContributions_stepPosition_y;   //!
   TBranch        *b_HcalEndcapPInsertHitsContributions_stepPosition_z;   //!
   TBranch        *b__HcalEndcapPInsertHitsContributions_particle_;   //!
   TBranch        *b__HcalEndcapPInsertHitsContributions_particle_index;   //!
   TBranch        *b__HcalEndcapPInsertHitsContributions_particle_collectionID;   //!
   TBranch        *b_HcalFarForwardZDCHits_;   //!
   TBranch        *b_HcalFarForwardZDCHits_cellID;   //!
   TBranch        *b_HcalFarForwardZDCHits_energy;   //!
   TBranch        *b_HcalFarForwardZDCHits_position_x;   //!
   TBranch        *b_HcalFarForwardZDCHits_position_y;   //!
   TBranch        *b_HcalFarForwardZDCHits_position_z;   //!
   TBranch        *b_HcalFarForwardZDCHits_contributions_begin;   //!
   TBranch        *b_HcalFarForwardZDCHits_contributions_end;   //!
   TBranch        *b__HcalFarForwardZDCHits_contributions_;   //!
   TBranch        *b__HcalFarForwardZDCHits_contributions_index;   //!
   TBranch        *b__HcalFarForwardZDCHits_contributions_collectionID;   //!
   TBranch        *b_HcalFarForwardZDCHitsContributions_;   //!
   TBranch        *b_HcalFarForwardZDCHitsContributions_PDG;   //!
   TBranch        *b_HcalFarForwardZDCHitsContributions_energy;   //!
   TBranch        *b_HcalFarForwardZDCHitsContributions_time;   //!
   TBranch        *b_HcalFarForwardZDCHitsContributions_stepPosition_x;   //!
   TBranch        *b_HcalFarForwardZDCHitsContributions_stepPosition_y;   //!
   TBranch        *b_HcalFarForwardZDCHitsContributions_stepPosition_z;   //!
   TBranch        *b__HcalFarForwardZDCHitsContributions_particle_;   //!
   TBranch        *b__HcalFarForwardZDCHitsContributions_particle_index;   //!
   TBranch        *b__HcalFarForwardZDCHitsContributions_particle_collectionID;   //!
   TBranch        *b_LFHCALHits_;   //!
   TBranch        *b_LFHCALHits_cellID;   //!
   TBranch        *b_LFHCALHits_energy;   //!
   TBranch        *b_LFHCALHits_position_x;   //!
   TBranch        *b_LFHCALHits_position_y;   //!
   TBranch        *b_LFHCALHits_position_z;   //!
   TBranch        *b_LFHCALHits_contributions_begin;   //!
   TBranch        *b_LFHCALHits_contributions_end;   //!
   TBranch        *b__LFHCALHits_contributions_;   //!
   TBranch        *b__LFHCALHits_contributions_index;   //!
   TBranch        *b__LFHCALHits_contributions_collectionID;   //!
   TBranch        *b_LFHCALHitsContributions_;   //!
   TBranch        *b_LFHCALHitsContributions_PDG;   //!
   TBranch        *b_LFHCALHitsContributions_energy;   //!
   TBranch        *b_LFHCALHitsContributions_time;   //!
   TBranch        *b_LFHCALHitsContributions_stepPosition_x;   //!
   TBranch        *b_LFHCALHitsContributions_stepPosition_y;   //!
   TBranch        *b_LFHCALHitsContributions_stepPosition_z;   //!
   TBranch        *b__LFHCALHitsContributions_particle_;   //!
   TBranch        *b__LFHCALHitsContributions_particle_index;   //!
   TBranch        *b__LFHCALHitsContributions_particle_collectionID;   //!
   TBranch        *b_LumiDirectPCALHits_;   //!
   TBranch        *b_LumiDirectPCALHits_cellID;   //!
   TBranch        *b_LumiDirectPCALHits_energy;   //!
   TBranch        *b_LumiDirectPCALHits_position_x;   //!
   TBranch        *b_LumiDirectPCALHits_position_y;   //!
   TBranch        *b_LumiDirectPCALHits_position_z;   //!
   TBranch        *b_LumiDirectPCALHits_contributions_begin;   //!
   TBranch        *b_LumiDirectPCALHits_contributions_end;   //!
   TBranch        *b__LumiDirectPCALHits_contributions_;   //!
   TBranch        *b__LumiDirectPCALHits_contributions_index;   //!
   TBranch        *b__LumiDirectPCALHits_contributions_collectionID;   //!
   TBranch        *b_LumiDirectPCALHitsContributions_;   //!
   TBranch        *b_LumiDirectPCALHitsContributions_PDG;   //!
   TBranch        *b_LumiDirectPCALHitsContributions_energy;   //!
   TBranch        *b_LumiDirectPCALHitsContributions_time;   //!
   TBranch        *b_LumiDirectPCALHitsContributions_stepPosition_x;   //!
   TBranch        *b_LumiDirectPCALHitsContributions_stepPosition_y;   //!
   TBranch        *b_LumiDirectPCALHitsContributions_stepPosition_z;   //!
   TBranch        *b__LumiDirectPCALHitsContributions_particle_;   //!
   TBranch        *b__LumiDirectPCALHitsContributions_particle_index;   //!
   TBranch        *b__LumiDirectPCALHitsContributions_particle_collectionID;   //!
   TBranch        *b_LumiSpecTrackerHits_;   //!
   TBranch        *b_LumiSpecTrackerHits_cellID;   //!
   TBranch        *b_LumiSpecTrackerHits_EDep;   //!
   TBranch        *b_LumiSpecTrackerHits_time;   //!
   TBranch        *b_LumiSpecTrackerHits_pathLength;   //!
   TBranch        *b_LumiSpecTrackerHits_quality;   //!
   TBranch        *b_LumiSpecTrackerHits_position_x;   //!
   TBranch        *b_LumiSpecTrackerHits_position_y;   //!
   TBranch        *b_LumiSpecTrackerHits_position_z;   //!
   TBranch        *b_LumiSpecTrackerHits_momentum_x;   //!
   TBranch        *b_LumiSpecTrackerHits_momentum_y;   //!
   TBranch        *b_LumiSpecTrackerHits_momentum_z;   //!
   TBranch        *b__LumiSpecTrackerHits_MCParticle_;   //!
   TBranch        *b__LumiSpecTrackerHits_MCParticle_index;   //!
   TBranch        *b__LumiSpecTrackerHits_MCParticle_collectionID;   //!
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
   TBranch        *b_MPGDBarrelHits_;   //!
   TBranch        *b_MPGDBarrelHits_cellID;   //!
   TBranch        *b_MPGDBarrelHits_EDep;   //!
   TBranch        *b_MPGDBarrelHits_time;   //!
   TBranch        *b_MPGDBarrelHits_pathLength;   //!
   TBranch        *b_MPGDBarrelHits_quality;   //!
   TBranch        *b_MPGDBarrelHits_position_x;   //!
   TBranch        *b_MPGDBarrelHits_position_y;   //!
   TBranch        *b_MPGDBarrelHits_position_z;   //!
   TBranch        *b_MPGDBarrelHits_momentum_x;   //!
   TBranch        *b_MPGDBarrelHits_momentum_y;   //!
   TBranch        *b_MPGDBarrelHits_momentum_z;   //!
   TBranch        *b__MPGDBarrelHits_MCParticle_;   //!
   TBranch        *b__MPGDBarrelHits_MCParticle_index;   //!
   TBranch        *b__MPGDBarrelHits_MCParticle_collectionID;   //!
   TBranch        *b_MPGDDIRCHits_;   //!
   TBranch        *b_MPGDDIRCHits_cellID;   //!
   TBranch        *b_MPGDDIRCHits_EDep;   //!
   TBranch        *b_MPGDDIRCHits_time;   //!
   TBranch        *b_MPGDDIRCHits_pathLength;   //!
   TBranch        *b_MPGDDIRCHits_quality;   //!
   TBranch        *b_MPGDDIRCHits_position_x;   //!
   TBranch        *b_MPGDDIRCHits_position_y;   //!
   TBranch        *b_MPGDDIRCHits_position_z;   //!
   TBranch        *b_MPGDDIRCHits_momentum_x;   //!
   TBranch        *b_MPGDDIRCHits_momentum_y;   //!
   TBranch        *b_MPGDDIRCHits_momentum_z;   //!
   TBranch        *b__MPGDDIRCHits_MCParticle_;   //!
   TBranch        *b__MPGDDIRCHits_MCParticle_index;   //!
   TBranch        *b__MPGDDIRCHits_MCParticle_collectionID;   //!
   TBranch        *b_PFRICHHits_;   //!
   TBranch        *b_PFRICHHits_cellID;   //!
   TBranch        *b_PFRICHHits_EDep;   //!
   TBranch        *b_PFRICHHits_time;   //!
   TBranch        *b_PFRICHHits_pathLength;   //!
   TBranch        *b_PFRICHHits_quality;   //!
   TBranch        *b_PFRICHHits_position_x;   //!
   TBranch        *b_PFRICHHits_position_y;   //!
   TBranch        *b_PFRICHHits_position_z;   //!
   TBranch        *b_PFRICHHits_momentum_x;   //!
   TBranch        *b_PFRICHHits_momentum_y;   //!
   TBranch        *b_PFRICHHits_momentum_z;   //!
   TBranch        *b__PFRICHHits_MCParticle_;   //!
   TBranch        *b__PFRICHHits_MCParticle_index;   //!
   TBranch        *b__PFRICHHits_MCParticle_collectionID;   //!
   TBranch        *b_SiBarrelHits_;   //!
   TBranch        *b_SiBarrelHits_cellID;   //!
   TBranch        *b_SiBarrelHits_EDep;   //!
   TBranch        *b_SiBarrelHits_time;   //!
   TBranch        *b_SiBarrelHits_pathLength;   //!
   TBranch        *b_SiBarrelHits_quality;   //!
   TBranch        *b_SiBarrelHits_position_x;   //!
   TBranch        *b_SiBarrelHits_position_y;   //!
   TBranch        *b_SiBarrelHits_position_z;   //!
   TBranch        *b_SiBarrelHits_momentum_x;   //!
   TBranch        *b_SiBarrelHits_momentum_y;   //!
   TBranch        *b_SiBarrelHits_momentum_z;   //!
   TBranch        *b__SiBarrelHits_MCParticle_;   //!
   TBranch        *b__SiBarrelHits_MCParticle_index;   //!
   TBranch        *b__SiBarrelHits_MCParticle_collectionID;   //!
   TBranch        *b_TaggerTrackerHits_;   //!
   TBranch        *b_TaggerTrackerHits_cellID;   //!
   TBranch        *b_TaggerTrackerHits_EDep;   //!
   TBranch        *b_TaggerTrackerHits_time;   //!
   TBranch        *b_TaggerTrackerHits_pathLength;   //!
   TBranch        *b_TaggerTrackerHits_quality;   //!
   TBranch        *b_TaggerTrackerHits_position_x;   //!
   TBranch        *b_TaggerTrackerHits_position_y;   //!
   TBranch        *b_TaggerTrackerHits_position_z;   //!
   TBranch        *b_TaggerTrackerHits_momentum_x;   //!
   TBranch        *b_TaggerTrackerHits_momentum_y;   //!
   TBranch        *b_TaggerTrackerHits_momentum_z;   //!
   TBranch        *b__TaggerTrackerHits_MCParticle_;   //!
   TBranch        *b__TaggerTrackerHits_MCParticle_index;   //!
   TBranch        *b__TaggerTrackerHits_MCParticle_collectionID;   //!
   TBranch        *b_TOFBarrelHits_;   //!
   TBranch        *b_TOFBarrelHits_cellID;   //!
   TBranch        *b_TOFBarrelHits_EDep;   //!
   TBranch        *b_TOFBarrelHits_time;   //!
   TBranch        *b_TOFBarrelHits_pathLength;   //!
   TBranch        *b_TOFBarrelHits_quality;   //!
   TBranch        *b_TOFBarrelHits_position_x;   //!
   TBranch        *b_TOFBarrelHits_position_y;   //!
   TBranch        *b_TOFBarrelHits_position_z;   //!
   TBranch        *b_TOFBarrelHits_momentum_x;   //!
   TBranch        *b_TOFBarrelHits_momentum_y;   //!
   TBranch        *b_TOFBarrelHits_momentum_z;   //!
   TBranch        *b__TOFBarrelHits_MCParticle_;   //!
   TBranch        *b__TOFBarrelHits_MCParticle_index;   //!
   TBranch        *b__TOFBarrelHits_MCParticle_collectionID;   //!
   TBranch        *b_TOFEndcapHits_;   //!
   TBranch        *b_TOFEndcapHits_cellID;   //!
   TBranch        *b_TOFEndcapHits_EDep;   //!
   TBranch        *b_TOFEndcapHits_time;   //!
   TBranch        *b_TOFEndcapHits_pathLength;   //!
   TBranch        *b_TOFEndcapHits_quality;   //!
   TBranch        *b_TOFEndcapHits_position_x;   //!
   TBranch        *b_TOFEndcapHits_position_y;   //!
   TBranch        *b_TOFEndcapHits_position_z;   //!
   TBranch        *b_TOFEndcapHits_momentum_x;   //!
   TBranch        *b_TOFEndcapHits_momentum_y;   //!
   TBranch        *b_TOFEndcapHits_momentum_z;   //!
   TBranch        *b__TOFEndcapHits_MCParticle_;   //!
   TBranch        *b__TOFEndcapHits_MCParticle_index;   //!
   TBranch        *b__TOFEndcapHits_MCParticle_collectionID;   //!
   TBranch        *b_TrackerEndcapHits_;   //!
   TBranch        *b_TrackerEndcapHits_cellID;   //!
   TBranch        *b_TrackerEndcapHits_EDep;   //!
   TBranch        *b_TrackerEndcapHits_time;   //!
   TBranch        *b_TrackerEndcapHits_pathLength;   //!
   TBranch        *b_TrackerEndcapHits_quality;   //!
   TBranch        *b_TrackerEndcapHits_position_x;   //!
   TBranch        *b_TrackerEndcapHits_position_y;   //!
   TBranch        *b_TrackerEndcapHits_position_z;   //!
   TBranch        *b_TrackerEndcapHits_momentum_x;   //!
   TBranch        *b_TrackerEndcapHits_momentum_y;   //!
   TBranch        *b_TrackerEndcapHits_momentum_z;   //!
   TBranch        *b__TrackerEndcapHits_MCParticle_;   //!
   TBranch        *b__TrackerEndcapHits_MCParticle_index;   //!
   TBranch        *b__TrackerEndcapHits_MCParticle_collectionID;   //!
   TBranch        *b_VertexBarrelHits_;   //!
   TBranch        *b_VertexBarrelHits_cellID;   //!
   TBranch        *b_VertexBarrelHits_EDep;   //!
   TBranch        *b_VertexBarrelHits_time;   //!
   TBranch        *b_VertexBarrelHits_pathLength;   //!
   TBranch        *b_VertexBarrelHits_quality;   //!
   TBranch        *b_VertexBarrelHits_position_x;   //!
   TBranch        *b_VertexBarrelHits_position_y;   //!
   TBranch        *b_VertexBarrelHits_position_z;   //!
   TBranch        *b_VertexBarrelHits_momentum_x;   //!
   TBranch        *b_VertexBarrelHits_momentum_y;   //!
   TBranch        *b_VertexBarrelHits_momentum_z;   //!
   TBranch        *b__VertexBarrelHits_MCParticle_;   //!
   TBranch        *b__VertexBarrelHits_MCParticle_index;   //!
   TBranch        *b__VertexBarrelHits_MCParticle_collectionID;   //!
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

   mu_endpoint(TTree *tree=0);
   virtual ~mu_endpoint();
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   virtual void     Loop();
   virtual Bool_t   Notify();
   virtual void     Show(Long64_t entry = -1);
};

#endif

#ifdef mu_endpoint_cxx
mu_endpoint::mu_endpoint(TTree *tree) : fChain(0) 
{
// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
   if (tree == 0) {
      TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("/home/rowan/eic/work_eic/root_files/EPIC_KLM_1GeV_mu_50000.edm4hep.root");
      if (!f || !f->IsOpen()) {
         f = new TFile("/home/rowan/eic/work_eic/root_files/EPIC_KLM_1GeV_mu_50000.edm4hep.root");
      }
      f->GetObject("events",tree);

   }
   Init(tree);
   Loop();
}

mu_endpoint::~mu_endpoint()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t mu_endpoint::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t mu_endpoint::LoadTree(Long64_t entry)
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

void mu_endpoint::Init(TTree *tree)
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

   fChain->SetBranchAddress("B0ECalHits", &B0ECalHits_, &b_B0ECalHits_);
   fChain->SetBranchAddress("B0ECalHits.cellID", &B0ECalHits_cellID, &b_B0ECalHits_cellID);
   fChain->SetBranchAddress("B0ECalHits.energy", &B0ECalHits_energy, &b_B0ECalHits_energy);
   fChain->SetBranchAddress("B0ECalHits.position.x", &B0ECalHits_position_x, &b_B0ECalHits_position_x);
   fChain->SetBranchAddress("B0ECalHits.position.y", &B0ECalHits_position_y, &b_B0ECalHits_position_y);
   fChain->SetBranchAddress("B0ECalHits.position.z", &B0ECalHits_position_z, &b_B0ECalHits_position_z);
   fChain->SetBranchAddress("B0ECalHits.contributions_begin", &B0ECalHits_contributions_begin, &b_B0ECalHits_contributions_begin);
   fChain->SetBranchAddress("B0ECalHits.contributions_end", &B0ECalHits_contributions_end, &b_B0ECalHits_contributions_end);
   fChain->SetBranchAddress("_B0ECalHits_contributions", &_B0ECalHits_contributions_, &b__B0ECalHits_contributions_);
   fChain->SetBranchAddress("_B0ECalHits_contributions.index", &_B0ECalHits_contributions_index, &b__B0ECalHits_contributions_index);
   fChain->SetBranchAddress("_B0ECalHits_contributions.collectionID", &_B0ECalHits_contributions_collectionID, &b__B0ECalHits_contributions_collectionID);
   fChain->SetBranchAddress("B0ECalHitsContributions", &B0ECalHitsContributions_, &b_B0ECalHitsContributions_);
   fChain->SetBranchAddress("B0ECalHitsContributions.PDG", &B0ECalHitsContributions_PDG, &b_B0ECalHitsContributions_PDG);
   fChain->SetBranchAddress("B0ECalHitsContributions.energy", &B0ECalHitsContributions_energy, &b_B0ECalHitsContributions_energy);
   fChain->SetBranchAddress("B0ECalHitsContributions.time", &B0ECalHitsContributions_time, &b_B0ECalHitsContributions_time);
   fChain->SetBranchAddress("B0ECalHitsContributions.stepPosition.x", &B0ECalHitsContributions_stepPosition_x, &b_B0ECalHitsContributions_stepPosition_x);
   fChain->SetBranchAddress("B0ECalHitsContributions.stepPosition.y", &B0ECalHitsContributions_stepPosition_y, &b_B0ECalHitsContributions_stepPosition_y);
   fChain->SetBranchAddress("B0ECalHitsContributions.stepPosition.z", &B0ECalHitsContributions_stepPosition_z, &b_B0ECalHitsContributions_stepPosition_z);
   fChain->SetBranchAddress("_B0ECalHitsContributions_particle", &_B0ECalHitsContributions_particle_, &b__B0ECalHitsContributions_particle_);
   fChain->SetBranchAddress("_B0ECalHitsContributions_particle.index", &_B0ECalHitsContributions_particle_index, &b__B0ECalHitsContributions_particle_index);
   fChain->SetBranchAddress("_B0ECalHitsContributions_particle.collectionID", &_B0ECalHitsContributions_particle_collectionID, &b__B0ECalHitsContributions_particle_collectionID);
   fChain->SetBranchAddress("B0TrackerHits", &B0TrackerHits_, &b_B0TrackerHits_);
   fChain->SetBranchAddress("B0TrackerHits.cellID", &B0TrackerHits_cellID, &b_B0TrackerHits_cellID);
   fChain->SetBranchAddress("B0TrackerHits.EDep", &B0TrackerHits_EDep, &b_B0TrackerHits_EDep);
   fChain->SetBranchAddress("B0TrackerHits.time", &B0TrackerHits_time, &b_B0TrackerHits_time);
   fChain->SetBranchAddress("B0TrackerHits.pathLength", &B0TrackerHits_pathLength, &b_B0TrackerHits_pathLength);
   fChain->SetBranchAddress("B0TrackerHits.quality", &B0TrackerHits_quality, &b_B0TrackerHits_quality);
   fChain->SetBranchAddress("B0TrackerHits.position.x", &B0TrackerHits_position_x, &b_B0TrackerHits_position_x);
   fChain->SetBranchAddress("B0TrackerHits.position.y", &B0TrackerHits_position_y, &b_B0TrackerHits_position_y);
   fChain->SetBranchAddress("B0TrackerHits.position.z", &B0TrackerHits_position_z, &b_B0TrackerHits_position_z);
   fChain->SetBranchAddress("B0TrackerHits.momentum.x", &B0TrackerHits_momentum_x, &b_B0TrackerHits_momentum_x);
   fChain->SetBranchAddress("B0TrackerHits.momentum.y", &B0TrackerHits_momentum_y, &b_B0TrackerHits_momentum_y);
   fChain->SetBranchAddress("B0TrackerHits.momentum.z", &B0TrackerHits_momentum_z, &b_B0TrackerHits_momentum_z);
   fChain->SetBranchAddress("_B0TrackerHits_MCParticle", &_B0TrackerHits_MCParticle_, &b__B0TrackerHits_MCParticle_);
   fChain->SetBranchAddress("_B0TrackerHits_MCParticle.index", &_B0TrackerHits_MCParticle_index, &b__B0TrackerHits_MCParticle_index);
   fChain->SetBranchAddress("_B0TrackerHits_MCParticle.collectionID", &_B0TrackerHits_MCParticle_collectionID, &b__B0TrackerHits_MCParticle_collectionID);
   fChain->SetBranchAddress("DIRCBarHits", &DIRCBarHits_, &b_DIRCBarHits_);
   fChain->SetBranchAddress("DIRCBarHits.cellID", &DIRCBarHits_cellID, &b_DIRCBarHits_cellID);
   fChain->SetBranchAddress("DIRCBarHits.EDep", &DIRCBarHits_EDep, &b_DIRCBarHits_EDep);
   fChain->SetBranchAddress("DIRCBarHits.time", &DIRCBarHits_time, &b_DIRCBarHits_time);
   fChain->SetBranchAddress("DIRCBarHits.pathLength", &DIRCBarHits_pathLength, &b_DIRCBarHits_pathLength);
   fChain->SetBranchAddress("DIRCBarHits.quality", &DIRCBarHits_quality, &b_DIRCBarHits_quality);
   fChain->SetBranchAddress("DIRCBarHits.position.x", &DIRCBarHits_position_x, &b_DIRCBarHits_position_x);
   fChain->SetBranchAddress("DIRCBarHits.position.y", &DIRCBarHits_position_y, &b_DIRCBarHits_position_y);
   fChain->SetBranchAddress("DIRCBarHits.position.z", &DIRCBarHits_position_z, &b_DIRCBarHits_position_z);
   fChain->SetBranchAddress("DIRCBarHits.momentum.x", &DIRCBarHits_momentum_x, &b_DIRCBarHits_momentum_x);
   fChain->SetBranchAddress("DIRCBarHits.momentum.y", &DIRCBarHits_momentum_y, &b_DIRCBarHits_momentum_y);
   fChain->SetBranchAddress("DIRCBarHits.momentum.z", &DIRCBarHits_momentum_z, &b_DIRCBarHits_momentum_z);
   fChain->SetBranchAddress("_DIRCBarHits_MCParticle", &_DIRCBarHits_MCParticle_, &b__DIRCBarHits_MCParticle_);
   fChain->SetBranchAddress("_DIRCBarHits_MCParticle.index", &_DIRCBarHits_MCParticle_index, &b__DIRCBarHits_MCParticle_index);
   fChain->SetBranchAddress("_DIRCBarHits_MCParticle.collectionID", &_DIRCBarHits_MCParticle_collectionID, &b__DIRCBarHits_MCParticle_collectionID);
   fChain->SetBranchAddress("DRICHHits", &DRICHHits_, &b_DRICHHits_);
   fChain->SetBranchAddress("DRICHHits.cellID", &DRICHHits_cellID, &b_DRICHHits_cellID);
   fChain->SetBranchAddress("DRICHHits.EDep", &DRICHHits_EDep, &b_DRICHHits_EDep);
   fChain->SetBranchAddress("DRICHHits.time", &DRICHHits_time, &b_DRICHHits_time);
   fChain->SetBranchAddress("DRICHHits.pathLength", &DRICHHits_pathLength, &b_DRICHHits_pathLength);
   fChain->SetBranchAddress("DRICHHits.quality", &DRICHHits_quality, &b_DRICHHits_quality);
   fChain->SetBranchAddress("DRICHHits.position.x", &DRICHHits_position_x, &b_DRICHHits_position_x);
   fChain->SetBranchAddress("DRICHHits.position.y", &DRICHHits_position_y, &b_DRICHHits_position_y);
   fChain->SetBranchAddress("DRICHHits.position.z", &DRICHHits_position_z, &b_DRICHHits_position_z);
   fChain->SetBranchAddress("DRICHHits.momentum.x", &DRICHHits_momentum_x, &b_DRICHHits_momentum_x);
   fChain->SetBranchAddress("DRICHHits.momentum.y", &DRICHHits_momentum_y, &b_DRICHHits_momentum_y);
   fChain->SetBranchAddress("DRICHHits.momentum.z", &DRICHHits_momentum_z, &b_DRICHHits_momentum_z);
   fChain->SetBranchAddress("_DRICHHits_MCParticle", &_DRICHHits_MCParticle_, &b__DRICHHits_MCParticle_);
   fChain->SetBranchAddress("_DRICHHits_MCParticle.index", &_DRICHHits_MCParticle_index, &b__DRICHHits_MCParticle_index);
   fChain->SetBranchAddress("_DRICHHits_MCParticle.collectionID", &_DRICHHits_MCParticle_collectionID, &b__DRICHHits_MCParticle_collectionID);
   fChain->SetBranchAddress("EcalBarrelImagingHits", &EcalBarrelImagingHits_, &b_EcalBarrelImagingHits_);
   fChain->SetBranchAddress("EcalBarrelImagingHits.cellID", EcalBarrelImagingHits_cellID, &b_EcalBarrelImagingHits_cellID);
   fChain->SetBranchAddress("EcalBarrelImagingHits.energy", EcalBarrelImagingHits_energy, &b_EcalBarrelImagingHits_energy);
   fChain->SetBranchAddress("EcalBarrelImagingHits.position.x", EcalBarrelImagingHits_position_x, &b_EcalBarrelImagingHits_position_x);
   fChain->SetBranchAddress("EcalBarrelImagingHits.position.y", EcalBarrelImagingHits_position_y, &b_EcalBarrelImagingHits_position_y);
   fChain->SetBranchAddress("EcalBarrelImagingHits.position.z", EcalBarrelImagingHits_position_z, &b_EcalBarrelImagingHits_position_z);
   fChain->SetBranchAddress("EcalBarrelImagingHits.contributions_begin", EcalBarrelImagingHits_contributions_begin, &b_EcalBarrelImagingHits_contributions_begin);
   fChain->SetBranchAddress("EcalBarrelImagingHits.contributions_end", EcalBarrelImagingHits_contributions_end, &b_EcalBarrelImagingHits_contributions_end);
   fChain->SetBranchAddress("_EcalBarrelImagingHits_contributions", &_EcalBarrelImagingHits_contributions_, &b__EcalBarrelImagingHits_contributions_);
   fChain->SetBranchAddress("_EcalBarrelImagingHits_contributions.index", _EcalBarrelImagingHits_contributions_index, &b__EcalBarrelImagingHits_contributions_index);
   fChain->SetBranchAddress("_EcalBarrelImagingHits_contributions.collectionID", _EcalBarrelImagingHits_contributions_collectionID, &b__EcalBarrelImagingHits_contributions_collectionID);
   fChain->SetBranchAddress("EcalBarrelImagingHitsContributions", &EcalBarrelImagingHitsContributions_, &b_EcalBarrelImagingHitsContributions_);
   fChain->SetBranchAddress("EcalBarrelImagingHitsContributions.PDG", EcalBarrelImagingHitsContributions_PDG, &b_EcalBarrelImagingHitsContributions_PDG);
   fChain->SetBranchAddress("EcalBarrelImagingHitsContributions.energy", EcalBarrelImagingHitsContributions_energy, &b_EcalBarrelImagingHitsContributions_energy);
   fChain->SetBranchAddress("EcalBarrelImagingHitsContributions.time", EcalBarrelImagingHitsContributions_time, &b_EcalBarrelImagingHitsContributions_time);
   fChain->SetBranchAddress("EcalBarrelImagingHitsContributions.stepPosition.x", EcalBarrelImagingHitsContributions_stepPosition_x, &b_EcalBarrelImagingHitsContributions_stepPosition_x);
   fChain->SetBranchAddress("EcalBarrelImagingHitsContributions.stepPosition.y", EcalBarrelImagingHitsContributions_stepPosition_y, &b_EcalBarrelImagingHitsContributions_stepPosition_y);
   fChain->SetBranchAddress("EcalBarrelImagingHitsContributions.stepPosition.z", EcalBarrelImagingHitsContributions_stepPosition_z, &b_EcalBarrelImagingHitsContributions_stepPosition_z);
   fChain->SetBranchAddress("_EcalBarrelImagingHitsContributions_particle", &_EcalBarrelImagingHitsContributions_particle_, &b__EcalBarrelImagingHitsContributions_particle_);
   fChain->SetBranchAddress("_EcalBarrelImagingHitsContributions_particle.index", _EcalBarrelImagingHitsContributions_particle_index, &b__EcalBarrelImagingHitsContributions_particle_index);
   fChain->SetBranchAddress("_EcalBarrelImagingHitsContributions_particle.collectionID", _EcalBarrelImagingHitsContributions_particle_collectionID, &b__EcalBarrelImagingHitsContributions_particle_collectionID);
   fChain->SetBranchAddress("EcalBarrelScFiHits", &EcalBarrelScFiHits_, &b_EcalBarrelScFiHits_);
   fChain->SetBranchAddress("EcalBarrelScFiHits.cellID", EcalBarrelScFiHits_cellID, &b_EcalBarrelScFiHits_cellID);
   fChain->SetBranchAddress("EcalBarrelScFiHits.energy", EcalBarrelScFiHits_energy, &b_EcalBarrelScFiHits_energy);
   fChain->SetBranchAddress("EcalBarrelScFiHits.position.x", EcalBarrelScFiHits_position_x, &b_EcalBarrelScFiHits_position_x);
   fChain->SetBranchAddress("EcalBarrelScFiHits.position.y", EcalBarrelScFiHits_position_y, &b_EcalBarrelScFiHits_position_y);
   fChain->SetBranchAddress("EcalBarrelScFiHits.position.z", EcalBarrelScFiHits_position_z, &b_EcalBarrelScFiHits_position_z);
   fChain->SetBranchAddress("EcalBarrelScFiHits.contributions_begin", EcalBarrelScFiHits_contributions_begin, &b_EcalBarrelScFiHits_contributions_begin);
   fChain->SetBranchAddress("EcalBarrelScFiHits.contributions_end", EcalBarrelScFiHits_contributions_end, &b_EcalBarrelScFiHits_contributions_end);
   fChain->SetBranchAddress("_EcalBarrelScFiHits_contributions", &_EcalBarrelScFiHits_contributions_, &b__EcalBarrelScFiHits_contributions_);
   fChain->SetBranchAddress("_EcalBarrelScFiHits_contributions.index", _EcalBarrelScFiHits_contributions_index, &b__EcalBarrelScFiHits_contributions_index);
   fChain->SetBranchAddress("_EcalBarrelScFiHits_contributions.collectionID", _EcalBarrelScFiHits_contributions_collectionID, &b__EcalBarrelScFiHits_contributions_collectionID);
   fChain->SetBranchAddress("EcalBarrelScFiHitsContributions", &EcalBarrelScFiHitsContributions_, &b_EcalBarrelScFiHitsContributions_);
   fChain->SetBranchAddress("EcalBarrelScFiHitsContributions.PDG", EcalBarrelScFiHitsContributions_PDG, &b_EcalBarrelScFiHitsContributions_PDG);
   fChain->SetBranchAddress("EcalBarrelScFiHitsContributions.energy", EcalBarrelScFiHitsContributions_energy, &b_EcalBarrelScFiHitsContributions_energy);
   fChain->SetBranchAddress("EcalBarrelScFiHitsContributions.time", EcalBarrelScFiHitsContributions_time, &b_EcalBarrelScFiHitsContributions_time);
   fChain->SetBranchAddress("EcalBarrelScFiHitsContributions.stepPosition.x", EcalBarrelScFiHitsContributions_stepPosition_x, &b_EcalBarrelScFiHitsContributions_stepPosition_x);
   fChain->SetBranchAddress("EcalBarrelScFiHitsContributions.stepPosition.y", EcalBarrelScFiHitsContributions_stepPosition_y, &b_EcalBarrelScFiHitsContributions_stepPosition_y);
   fChain->SetBranchAddress("EcalBarrelScFiHitsContributions.stepPosition.z", EcalBarrelScFiHitsContributions_stepPosition_z, &b_EcalBarrelScFiHitsContributions_stepPosition_z);
   fChain->SetBranchAddress("_EcalBarrelScFiHitsContributions_particle", &_EcalBarrelScFiHitsContributions_particle_, &b__EcalBarrelScFiHitsContributions_particle_);
   fChain->SetBranchAddress("_EcalBarrelScFiHitsContributions_particle.index", _EcalBarrelScFiHitsContributions_particle_index, &b__EcalBarrelScFiHitsContributions_particle_index);
   fChain->SetBranchAddress("_EcalBarrelScFiHitsContributions_particle.collectionID", _EcalBarrelScFiHitsContributions_particle_collectionID, &b__EcalBarrelScFiHitsContributions_particle_collectionID);
   fChain->SetBranchAddress("EcalEndcapNHits", &EcalEndcapNHits_, &b_EcalEndcapNHits_);
   fChain->SetBranchAddress("EcalEndcapNHits.cellID", EcalEndcapNHits_cellID, &b_EcalEndcapNHits_cellID);
   fChain->SetBranchAddress("EcalEndcapNHits.energy", EcalEndcapNHits_energy, &b_EcalEndcapNHits_energy);
   fChain->SetBranchAddress("EcalEndcapNHits.position.x", EcalEndcapNHits_position_x, &b_EcalEndcapNHits_position_x);
   fChain->SetBranchAddress("EcalEndcapNHits.position.y", EcalEndcapNHits_position_y, &b_EcalEndcapNHits_position_y);
   fChain->SetBranchAddress("EcalEndcapNHits.position.z", EcalEndcapNHits_position_z, &b_EcalEndcapNHits_position_z);
   fChain->SetBranchAddress("EcalEndcapNHits.contributions_begin", EcalEndcapNHits_contributions_begin, &b_EcalEndcapNHits_contributions_begin);
   fChain->SetBranchAddress("EcalEndcapNHits.contributions_end", EcalEndcapNHits_contributions_end, &b_EcalEndcapNHits_contributions_end);
   fChain->SetBranchAddress("_EcalEndcapNHits_contributions", &_EcalEndcapNHits_contributions_, &b__EcalEndcapNHits_contributions_);
   fChain->SetBranchAddress("_EcalEndcapNHits_contributions.index", _EcalEndcapNHits_contributions_index, &b__EcalEndcapNHits_contributions_index);
   fChain->SetBranchAddress("_EcalEndcapNHits_contributions.collectionID", _EcalEndcapNHits_contributions_collectionID, &b__EcalEndcapNHits_contributions_collectionID);
   fChain->SetBranchAddress("EcalEndcapNHitsContributions", &EcalEndcapNHitsContributions_, &b_EcalEndcapNHitsContributions_);
   fChain->SetBranchAddress("EcalEndcapNHitsContributions.PDG", EcalEndcapNHitsContributions_PDG, &b_EcalEndcapNHitsContributions_PDG);
   fChain->SetBranchAddress("EcalEndcapNHitsContributions.energy", EcalEndcapNHitsContributions_energy, &b_EcalEndcapNHitsContributions_energy);
   fChain->SetBranchAddress("EcalEndcapNHitsContributions.time", EcalEndcapNHitsContributions_time, &b_EcalEndcapNHitsContributions_time);
   fChain->SetBranchAddress("EcalEndcapNHitsContributions.stepPosition.x", EcalEndcapNHitsContributions_stepPosition_x, &b_EcalEndcapNHitsContributions_stepPosition_x);
   fChain->SetBranchAddress("EcalEndcapNHitsContributions.stepPosition.y", EcalEndcapNHitsContributions_stepPosition_y, &b_EcalEndcapNHitsContributions_stepPosition_y);
   fChain->SetBranchAddress("EcalEndcapNHitsContributions.stepPosition.z", EcalEndcapNHitsContributions_stepPosition_z, &b_EcalEndcapNHitsContributions_stepPosition_z);
   fChain->SetBranchAddress("_EcalEndcapNHitsContributions_particle", &_EcalEndcapNHitsContributions_particle_, &b__EcalEndcapNHitsContributions_particle_);
   fChain->SetBranchAddress("_EcalEndcapNHitsContributions_particle.index", _EcalEndcapNHitsContributions_particle_index, &b__EcalEndcapNHitsContributions_particle_index);
   fChain->SetBranchAddress("_EcalEndcapNHitsContributions_particle.collectionID", _EcalEndcapNHitsContributions_particle_collectionID, &b__EcalEndcapNHitsContributions_particle_collectionID);
   fChain->SetBranchAddress("EcalEndcapPHits", &EcalEndcapPHits_, &b_EcalEndcapPHits_);
   fChain->SetBranchAddress("EcalEndcapPHits.cellID", EcalEndcapPHits_cellID, &b_EcalEndcapPHits_cellID);
   fChain->SetBranchAddress("EcalEndcapPHits.energy", EcalEndcapPHits_energy, &b_EcalEndcapPHits_energy);
   fChain->SetBranchAddress("EcalEndcapPHits.position.x", EcalEndcapPHits_position_x, &b_EcalEndcapPHits_position_x);
   fChain->SetBranchAddress("EcalEndcapPHits.position.y", EcalEndcapPHits_position_y, &b_EcalEndcapPHits_position_y);
   fChain->SetBranchAddress("EcalEndcapPHits.position.z", EcalEndcapPHits_position_z, &b_EcalEndcapPHits_position_z);
   fChain->SetBranchAddress("EcalEndcapPHits.contributions_begin", EcalEndcapPHits_contributions_begin, &b_EcalEndcapPHits_contributions_begin);
   fChain->SetBranchAddress("EcalEndcapPHits.contributions_end", EcalEndcapPHits_contributions_end, &b_EcalEndcapPHits_contributions_end);
   fChain->SetBranchAddress("_EcalEndcapPHits_contributions", &_EcalEndcapPHits_contributions_, &b__EcalEndcapPHits_contributions_);
   fChain->SetBranchAddress("_EcalEndcapPHits_contributions.index", _EcalEndcapPHits_contributions_index, &b__EcalEndcapPHits_contributions_index);
   fChain->SetBranchAddress("_EcalEndcapPHits_contributions.collectionID", _EcalEndcapPHits_contributions_collectionID, &b__EcalEndcapPHits_contributions_collectionID);
   fChain->SetBranchAddress("EcalEndcapPHitsContributions", &EcalEndcapPHitsContributions_, &b_EcalEndcapPHitsContributions_);
   fChain->SetBranchAddress("EcalEndcapPHitsContributions.PDG", EcalEndcapPHitsContributions_PDG, &b_EcalEndcapPHitsContributions_PDG);
   fChain->SetBranchAddress("EcalEndcapPHitsContributions.energy", EcalEndcapPHitsContributions_energy, &b_EcalEndcapPHitsContributions_energy);
   fChain->SetBranchAddress("EcalEndcapPHitsContributions.time", EcalEndcapPHitsContributions_time, &b_EcalEndcapPHitsContributions_time);
   fChain->SetBranchAddress("EcalEndcapPHitsContributions.stepPosition.x", EcalEndcapPHitsContributions_stepPosition_x, &b_EcalEndcapPHitsContributions_stepPosition_x);
   fChain->SetBranchAddress("EcalEndcapPHitsContributions.stepPosition.y", EcalEndcapPHitsContributions_stepPosition_y, &b_EcalEndcapPHitsContributions_stepPosition_y);
   fChain->SetBranchAddress("EcalEndcapPHitsContributions.stepPosition.z", EcalEndcapPHitsContributions_stepPosition_z, &b_EcalEndcapPHitsContributions_stepPosition_z);
   fChain->SetBranchAddress("_EcalEndcapPHitsContributions_particle", &_EcalEndcapPHitsContributions_particle_, &b__EcalEndcapPHitsContributions_particle_);
   fChain->SetBranchAddress("_EcalEndcapPHitsContributions_particle.index", _EcalEndcapPHitsContributions_particle_index, &b__EcalEndcapPHitsContributions_particle_index);
   fChain->SetBranchAddress("_EcalEndcapPHitsContributions_particle.collectionID", _EcalEndcapPHitsContributions_particle_collectionID, &b__EcalEndcapPHitsContributions_particle_collectionID);
   fChain->SetBranchAddress("EcalEndcapPInsertHits", &EcalEndcapPInsertHits_, &b_EcalEndcapPInsertHits_);
   fChain->SetBranchAddress("EcalEndcapPInsertHits.cellID", EcalEndcapPInsertHits_cellID, &b_EcalEndcapPInsertHits_cellID);
   fChain->SetBranchAddress("EcalEndcapPInsertHits.energy", EcalEndcapPInsertHits_energy, &b_EcalEndcapPInsertHits_energy);
   fChain->SetBranchAddress("EcalEndcapPInsertHits.position.x", EcalEndcapPInsertHits_position_x, &b_EcalEndcapPInsertHits_position_x);
   fChain->SetBranchAddress("EcalEndcapPInsertHits.position.y", EcalEndcapPInsertHits_position_y, &b_EcalEndcapPInsertHits_position_y);
   fChain->SetBranchAddress("EcalEndcapPInsertHits.position.z", EcalEndcapPInsertHits_position_z, &b_EcalEndcapPInsertHits_position_z);
   fChain->SetBranchAddress("EcalEndcapPInsertHits.contributions_begin", EcalEndcapPInsertHits_contributions_begin, &b_EcalEndcapPInsertHits_contributions_begin);
   fChain->SetBranchAddress("EcalEndcapPInsertHits.contributions_end", EcalEndcapPInsertHits_contributions_end, &b_EcalEndcapPInsertHits_contributions_end);
   fChain->SetBranchAddress("_EcalEndcapPInsertHits_contributions", &_EcalEndcapPInsertHits_contributions_, &b__EcalEndcapPInsertHits_contributions_);
   fChain->SetBranchAddress("_EcalEndcapPInsertHits_contributions.index", _EcalEndcapPInsertHits_contributions_index, &b__EcalEndcapPInsertHits_contributions_index);
   fChain->SetBranchAddress("_EcalEndcapPInsertHits_contributions.collectionID", _EcalEndcapPInsertHits_contributions_collectionID, &b__EcalEndcapPInsertHits_contributions_collectionID);
   fChain->SetBranchAddress("EcalEndcapPInsertHitsContributions", &EcalEndcapPInsertHitsContributions_, &b_EcalEndcapPInsertHitsContributions_);
   fChain->SetBranchAddress("EcalEndcapPInsertHitsContributions.PDG", EcalEndcapPInsertHitsContributions_PDG, &b_EcalEndcapPInsertHitsContributions_PDG);
   fChain->SetBranchAddress("EcalEndcapPInsertHitsContributions.energy", EcalEndcapPInsertHitsContributions_energy, &b_EcalEndcapPInsertHitsContributions_energy);
   fChain->SetBranchAddress("EcalEndcapPInsertHitsContributions.time", EcalEndcapPInsertHitsContributions_time, &b_EcalEndcapPInsertHitsContributions_time);
   fChain->SetBranchAddress("EcalEndcapPInsertHitsContributions.stepPosition.x", EcalEndcapPInsertHitsContributions_stepPosition_x, &b_EcalEndcapPInsertHitsContributions_stepPosition_x);
   fChain->SetBranchAddress("EcalEndcapPInsertHitsContributions.stepPosition.y", EcalEndcapPInsertHitsContributions_stepPosition_y, &b_EcalEndcapPInsertHitsContributions_stepPosition_y);
   fChain->SetBranchAddress("EcalEndcapPInsertHitsContributions.stepPosition.z", EcalEndcapPInsertHitsContributions_stepPosition_z, &b_EcalEndcapPInsertHitsContributions_stepPosition_z);
   fChain->SetBranchAddress("_EcalEndcapPInsertHitsContributions_particle", &_EcalEndcapPInsertHitsContributions_particle_, &b__EcalEndcapPInsertHitsContributions_particle_);
   fChain->SetBranchAddress("_EcalEndcapPInsertHitsContributions_particle.index", _EcalEndcapPInsertHitsContributions_particle_index, &b__EcalEndcapPInsertHitsContributions_particle_index);
   fChain->SetBranchAddress("_EcalEndcapPInsertHitsContributions_particle.collectionID", _EcalEndcapPInsertHitsContributions_particle_collectionID, &b__EcalEndcapPInsertHitsContributions_particle_collectionID);
   fChain->SetBranchAddress("EcalFarForwardZDCHits", &EcalFarForwardZDCHits_, &b_EcalFarForwardZDCHits_);
   fChain->SetBranchAddress("EcalFarForwardZDCHits.cellID", &EcalFarForwardZDCHits_cellID, &b_EcalFarForwardZDCHits_cellID);
   fChain->SetBranchAddress("EcalFarForwardZDCHits.energy", &EcalFarForwardZDCHits_energy, &b_EcalFarForwardZDCHits_energy);
   fChain->SetBranchAddress("EcalFarForwardZDCHits.position.x", &EcalFarForwardZDCHits_position_x, &b_EcalFarForwardZDCHits_position_x);
   fChain->SetBranchAddress("EcalFarForwardZDCHits.position.y", &EcalFarForwardZDCHits_position_y, &b_EcalFarForwardZDCHits_position_y);
   fChain->SetBranchAddress("EcalFarForwardZDCHits.position.z", &EcalFarForwardZDCHits_position_z, &b_EcalFarForwardZDCHits_position_z);
   fChain->SetBranchAddress("EcalFarForwardZDCHits.contributions_begin", &EcalFarForwardZDCHits_contributions_begin, &b_EcalFarForwardZDCHits_contributions_begin);
   fChain->SetBranchAddress("EcalFarForwardZDCHits.contributions_end", &EcalFarForwardZDCHits_contributions_end, &b_EcalFarForwardZDCHits_contributions_end);
   fChain->SetBranchAddress("_EcalFarForwardZDCHits_contributions", &_EcalFarForwardZDCHits_contributions_, &b__EcalFarForwardZDCHits_contributions_);
   fChain->SetBranchAddress("_EcalFarForwardZDCHits_contributions.index", &_EcalFarForwardZDCHits_contributions_index, &b__EcalFarForwardZDCHits_contributions_index);
   fChain->SetBranchAddress("_EcalFarForwardZDCHits_contributions.collectionID", &_EcalFarForwardZDCHits_contributions_collectionID, &b__EcalFarForwardZDCHits_contributions_collectionID);
   fChain->SetBranchAddress("EcalFarForwardZDCHitsContributions", &EcalFarForwardZDCHitsContributions_, &b_EcalFarForwardZDCHitsContributions_);
   fChain->SetBranchAddress("EcalFarForwardZDCHitsContributions.PDG", &EcalFarForwardZDCHitsContributions_PDG, &b_EcalFarForwardZDCHitsContributions_PDG);
   fChain->SetBranchAddress("EcalFarForwardZDCHitsContributions.energy", &EcalFarForwardZDCHitsContributions_energy, &b_EcalFarForwardZDCHitsContributions_energy);
   fChain->SetBranchAddress("EcalFarForwardZDCHitsContributions.time", &EcalFarForwardZDCHitsContributions_time, &b_EcalFarForwardZDCHitsContributions_time);
   fChain->SetBranchAddress("EcalFarForwardZDCHitsContributions.stepPosition.x", &EcalFarForwardZDCHitsContributions_stepPosition_x, &b_EcalFarForwardZDCHitsContributions_stepPosition_x);
   fChain->SetBranchAddress("EcalFarForwardZDCHitsContributions.stepPosition.y", &EcalFarForwardZDCHitsContributions_stepPosition_y, &b_EcalFarForwardZDCHitsContributions_stepPosition_y);
   fChain->SetBranchAddress("EcalFarForwardZDCHitsContributions.stepPosition.z", &EcalFarForwardZDCHitsContributions_stepPosition_z, &b_EcalFarForwardZDCHitsContributions_stepPosition_z);
   fChain->SetBranchAddress("_EcalFarForwardZDCHitsContributions_particle", &_EcalFarForwardZDCHitsContributions_particle_, &b__EcalFarForwardZDCHitsContributions_particle_);
   fChain->SetBranchAddress("_EcalFarForwardZDCHitsContributions_particle.index", &_EcalFarForwardZDCHitsContributions_particle_index, &b__EcalFarForwardZDCHitsContributions_particle_index);
   fChain->SetBranchAddress("_EcalFarForwardZDCHitsContributions_particle.collectionID", &_EcalFarForwardZDCHitsContributions_particle_collectionID, &b__EcalFarForwardZDCHitsContributions_particle_collectionID);
   fChain->SetBranchAddress("EcalLumiSpecHits", &EcalLumiSpecHits_, &b_EcalLumiSpecHits_);
   fChain->SetBranchAddress("EcalLumiSpecHits.cellID", &EcalLumiSpecHits_cellID, &b_EcalLumiSpecHits_cellID);
   fChain->SetBranchAddress("EcalLumiSpecHits.energy", &EcalLumiSpecHits_energy, &b_EcalLumiSpecHits_energy);
   fChain->SetBranchAddress("EcalLumiSpecHits.position.x", &EcalLumiSpecHits_position_x, &b_EcalLumiSpecHits_position_x);
   fChain->SetBranchAddress("EcalLumiSpecHits.position.y", &EcalLumiSpecHits_position_y, &b_EcalLumiSpecHits_position_y);
   fChain->SetBranchAddress("EcalLumiSpecHits.position.z", &EcalLumiSpecHits_position_z, &b_EcalLumiSpecHits_position_z);
   fChain->SetBranchAddress("EcalLumiSpecHits.contributions_begin", &EcalLumiSpecHits_contributions_begin, &b_EcalLumiSpecHits_contributions_begin);
   fChain->SetBranchAddress("EcalLumiSpecHits.contributions_end", &EcalLumiSpecHits_contributions_end, &b_EcalLumiSpecHits_contributions_end);
   fChain->SetBranchAddress("_EcalLumiSpecHits_contributions", &_EcalLumiSpecHits_contributions_, &b__EcalLumiSpecHits_contributions_);
   fChain->SetBranchAddress("_EcalLumiSpecHits_contributions.index", &_EcalLumiSpecHits_contributions_index, &b__EcalLumiSpecHits_contributions_index);
   fChain->SetBranchAddress("_EcalLumiSpecHits_contributions.collectionID", &_EcalLumiSpecHits_contributions_collectionID, &b__EcalLumiSpecHits_contributions_collectionID);
   fChain->SetBranchAddress("EcalLumiSpecHitsContributions", &EcalLumiSpecHitsContributions_, &b_EcalLumiSpecHitsContributions_);
   fChain->SetBranchAddress("EcalLumiSpecHitsContributions.PDG", &EcalLumiSpecHitsContributions_PDG, &b_EcalLumiSpecHitsContributions_PDG);
   fChain->SetBranchAddress("EcalLumiSpecHitsContributions.energy", &EcalLumiSpecHitsContributions_energy, &b_EcalLumiSpecHitsContributions_energy);
   fChain->SetBranchAddress("EcalLumiSpecHitsContributions.time", &EcalLumiSpecHitsContributions_time, &b_EcalLumiSpecHitsContributions_time);
   fChain->SetBranchAddress("EcalLumiSpecHitsContributions.stepPosition.x", &EcalLumiSpecHitsContributions_stepPosition_x, &b_EcalLumiSpecHitsContributions_stepPosition_x);
   fChain->SetBranchAddress("EcalLumiSpecHitsContributions.stepPosition.y", &EcalLumiSpecHitsContributions_stepPosition_y, &b_EcalLumiSpecHitsContributions_stepPosition_y);
   fChain->SetBranchAddress("EcalLumiSpecHitsContributions.stepPosition.z", &EcalLumiSpecHitsContributions_stepPosition_z, &b_EcalLumiSpecHitsContributions_stepPosition_z);
   fChain->SetBranchAddress("_EcalLumiSpecHitsContributions_particle", &_EcalLumiSpecHitsContributions_particle_, &b__EcalLumiSpecHitsContributions_particle_);
   fChain->SetBranchAddress("_EcalLumiSpecHitsContributions_particle.index", &_EcalLumiSpecHitsContributions_particle_index, &b__EcalLumiSpecHitsContributions_particle_index);
   fChain->SetBranchAddress("_EcalLumiSpecHitsContributions_particle.collectionID", &_EcalLumiSpecHitsContributions_particle_collectionID, &b__EcalLumiSpecHitsContributions_particle_collectionID);
   fChain->SetBranchAddress("EventHeader", &EventHeader_, &b_EventHeader_);
   fChain->SetBranchAddress("EventHeader.eventNumber", EventHeader_eventNumber, &b_EventHeader_eventNumber);
   fChain->SetBranchAddress("EventHeader.runNumber", EventHeader_runNumber, &b_EventHeader_runNumber);
   fChain->SetBranchAddress("EventHeader.timeStamp", EventHeader_timeStamp, &b_EventHeader_timeStamp);
   fChain->SetBranchAddress("EventHeader.weight", EventHeader_weight, &b_EventHeader_weight);
   fChain->SetBranchAddress("ForwardOffMTrackerHits", &ForwardOffMTrackerHits_, &b_ForwardOffMTrackerHits_);
   fChain->SetBranchAddress("ForwardOffMTrackerHits.cellID", &ForwardOffMTrackerHits_cellID, &b_ForwardOffMTrackerHits_cellID);
   fChain->SetBranchAddress("ForwardOffMTrackerHits.EDep", &ForwardOffMTrackerHits_EDep, &b_ForwardOffMTrackerHits_EDep);
   fChain->SetBranchAddress("ForwardOffMTrackerHits.time", &ForwardOffMTrackerHits_time, &b_ForwardOffMTrackerHits_time);
   fChain->SetBranchAddress("ForwardOffMTrackerHits.pathLength", &ForwardOffMTrackerHits_pathLength, &b_ForwardOffMTrackerHits_pathLength);
   fChain->SetBranchAddress("ForwardOffMTrackerHits.quality", &ForwardOffMTrackerHits_quality, &b_ForwardOffMTrackerHits_quality);
   fChain->SetBranchAddress("ForwardOffMTrackerHits.position.x", &ForwardOffMTrackerHits_position_x, &b_ForwardOffMTrackerHits_position_x);
   fChain->SetBranchAddress("ForwardOffMTrackerHits.position.y", &ForwardOffMTrackerHits_position_y, &b_ForwardOffMTrackerHits_position_y);
   fChain->SetBranchAddress("ForwardOffMTrackerHits.position.z", &ForwardOffMTrackerHits_position_z, &b_ForwardOffMTrackerHits_position_z);
   fChain->SetBranchAddress("ForwardOffMTrackerHits.momentum.x", &ForwardOffMTrackerHits_momentum_x, &b_ForwardOffMTrackerHits_momentum_x);
   fChain->SetBranchAddress("ForwardOffMTrackerHits.momentum.y", &ForwardOffMTrackerHits_momentum_y, &b_ForwardOffMTrackerHits_momentum_y);
   fChain->SetBranchAddress("ForwardOffMTrackerHits.momentum.z", &ForwardOffMTrackerHits_momentum_z, &b_ForwardOffMTrackerHits_momentum_z);
   fChain->SetBranchAddress("_ForwardOffMTrackerHits_MCParticle", &_ForwardOffMTrackerHits_MCParticle_, &b__ForwardOffMTrackerHits_MCParticle_);
   fChain->SetBranchAddress("_ForwardOffMTrackerHits_MCParticle.index", &_ForwardOffMTrackerHits_MCParticle_index, &b__ForwardOffMTrackerHits_MCParticle_index);
   fChain->SetBranchAddress("_ForwardOffMTrackerHits_MCParticle.collectionID", &_ForwardOffMTrackerHits_MCParticle_collectionID, &b__ForwardOffMTrackerHits_MCParticle_collectionID);
   fChain->SetBranchAddress("ForwardRomanPotHits", &ForwardRomanPotHits_, &b_ForwardRomanPotHits_);
   fChain->SetBranchAddress("ForwardRomanPotHits.cellID", &ForwardRomanPotHits_cellID, &b_ForwardRomanPotHits_cellID);
   fChain->SetBranchAddress("ForwardRomanPotHits.EDep", &ForwardRomanPotHits_EDep, &b_ForwardRomanPotHits_EDep);
   fChain->SetBranchAddress("ForwardRomanPotHits.time", &ForwardRomanPotHits_time, &b_ForwardRomanPotHits_time);
   fChain->SetBranchAddress("ForwardRomanPotHits.pathLength", &ForwardRomanPotHits_pathLength, &b_ForwardRomanPotHits_pathLength);
   fChain->SetBranchAddress("ForwardRomanPotHits.quality", &ForwardRomanPotHits_quality, &b_ForwardRomanPotHits_quality);
   fChain->SetBranchAddress("ForwardRomanPotHits.position.x", &ForwardRomanPotHits_position_x, &b_ForwardRomanPotHits_position_x);
   fChain->SetBranchAddress("ForwardRomanPotHits.position.y", &ForwardRomanPotHits_position_y, &b_ForwardRomanPotHits_position_y);
   fChain->SetBranchAddress("ForwardRomanPotHits.position.z", &ForwardRomanPotHits_position_z, &b_ForwardRomanPotHits_position_z);
   fChain->SetBranchAddress("ForwardRomanPotHits.momentum.x", &ForwardRomanPotHits_momentum_x, &b_ForwardRomanPotHits_momentum_x);
   fChain->SetBranchAddress("ForwardRomanPotHits.momentum.y", &ForwardRomanPotHits_momentum_y, &b_ForwardRomanPotHits_momentum_y);
   fChain->SetBranchAddress("ForwardRomanPotHits.momentum.z", &ForwardRomanPotHits_momentum_z, &b_ForwardRomanPotHits_momentum_z);
   fChain->SetBranchAddress("_ForwardRomanPotHits_MCParticle", &_ForwardRomanPotHits_MCParticle_, &b__ForwardRomanPotHits_MCParticle_);
   fChain->SetBranchAddress("_ForwardRomanPotHits_MCParticle.index", &_ForwardRomanPotHits_MCParticle_index, &b__ForwardRomanPotHits_MCParticle_index);
   fChain->SetBranchAddress("_ForwardRomanPotHits_MCParticle.collectionID", &_ForwardRomanPotHits_MCParticle_collectionID, &b__ForwardRomanPotHits_MCParticle_collectionID);
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
   fChain->SetBranchAddress("HcalEndcapNHits", &HcalEndcapNHits_, &b_HcalEndcapNHits_);
   fChain->SetBranchAddress("HcalEndcapNHits.cellID", HcalEndcapNHits_cellID, &b_HcalEndcapNHits_cellID);
   fChain->SetBranchAddress("HcalEndcapNHits.energy", HcalEndcapNHits_energy, &b_HcalEndcapNHits_energy);
   fChain->SetBranchAddress("HcalEndcapNHits.position.x", HcalEndcapNHits_position_x, &b_HcalEndcapNHits_position_x);
   fChain->SetBranchAddress("HcalEndcapNHits.position.y", HcalEndcapNHits_position_y, &b_HcalEndcapNHits_position_y);
   fChain->SetBranchAddress("HcalEndcapNHits.position.z", HcalEndcapNHits_position_z, &b_HcalEndcapNHits_position_z);
   fChain->SetBranchAddress("HcalEndcapNHits.contributions_begin", HcalEndcapNHits_contributions_begin, &b_HcalEndcapNHits_contributions_begin);
   fChain->SetBranchAddress("HcalEndcapNHits.contributions_end", HcalEndcapNHits_contributions_end, &b_HcalEndcapNHits_contributions_end);
   fChain->SetBranchAddress("_HcalEndcapNHits_contributions", &_HcalEndcapNHits_contributions_, &b__HcalEndcapNHits_contributions_);
   fChain->SetBranchAddress("_HcalEndcapNHits_contributions.index", _HcalEndcapNHits_contributions_index, &b__HcalEndcapNHits_contributions_index);
   fChain->SetBranchAddress("_HcalEndcapNHits_contributions.collectionID", _HcalEndcapNHits_contributions_collectionID, &b__HcalEndcapNHits_contributions_collectionID);
   fChain->SetBranchAddress("HcalEndcapNHitsContributions", &HcalEndcapNHitsContributions_, &b_HcalEndcapNHitsContributions_);
   fChain->SetBranchAddress("HcalEndcapNHitsContributions.PDG", HcalEndcapNHitsContributions_PDG, &b_HcalEndcapNHitsContributions_PDG);
   fChain->SetBranchAddress("HcalEndcapNHitsContributions.energy", HcalEndcapNHitsContributions_energy, &b_HcalEndcapNHitsContributions_energy);
   fChain->SetBranchAddress("HcalEndcapNHitsContributions.time", HcalEndcapNHitsContributions_time, &b_HcalEndcapNHitsContributions_time);
   fChain->SetBranchAddress("HcalEndcapNHitsContributions.stepPosition.x", HcalEndcapNHitsContributions_stepPosition_x, &b_HcalEndcapNHitsContributions_stepPosition_x);
   fChain->SetBranchAddress("HcalEndcapNHitsContributions.stepPosition.y", HcalEndcapNHitsContributions_stepPosition_y, &b_HcalEndcapNHitsContributions_stepPosition_y);
   fChain->SetBranchAddress("HcalEndcapNHitsContributions.stepPosition.z", HcalEndcapNHitsContributions_stepPosition_z, &b_HcalEndcapNHitsContributions_stepPosition_z);
   fChain->SetBranchAddress("_HcalEndcapNHitsContributions_particle", &_HcalEndcapNHitsContributions_particle_, &b__HcalEndcapNHitsContributions_particle_);
   fChain->SetBranchAddress("_HcalEndcapNHitsContributions_particle.index", _HcalEndcapNHitsContributions_particle_index, &b__HcalEndcapNHitsContributions_particle_index);
   fChain->SetBranchAddress("_HcalEndcapNHitsContributions_particle.collectionID", _HcalEndcapNHitsContributions_particle_collectionID, &b__HcalEndcapNHitsContributions_particle_collectionID);
   fChain->SetBranchAddress("HcalEndcapPInsertHits", &HcalEndcapPInsertHits_, &b_HcalEndcapPInsertHits_);
   fChain->SetBranchAddress("HcalEndcapPInsertHits.cellID", &HcalEndcapPInsertHits_cellID, &b_HcalEndcapPInsertHits_cellID);
   fChain->SetBranchAddress("HcalEndcapPInsertHits.energy", &HcalEndcapPInsertHits_energy, &b_HcalEndcapPInsertHits_energy);
   fChain->SetBranchAddress("HcalEndcapPInsertHits.position.x", &HcalEndcapPInsertHits_position_x, &b_HcalEndcapPInsertHits_position_x);
   fChain->SetBranchAddress("HcalEndcapPInsertHits.position.y", &HcalEndcapPInsertHits_position_y, &b_HcalEndcapPInsertHits_position_y);
   fChain->SetBranchAddress("HcalEndcapPInsertHits.position.z", &HcalEndcapPInsertHits_position_z, &b_HcalEndcapPInsertHits_position_z);
   fChain->SetBranchAddress("HcalEndcapPInsertHits.contributions_begin", &HcalEndcapPInsertHits_contributions_begin, &b_HcalEndcapPInsertHits_contributions_begin);
   fChain->SetBranchAddress("HcalEndcapPInsertHits.contributions_end", &HcalEndcapPInsertHits_contributions_end, &b_HcalEndcapPInsertHits_contributions_end);
   fChain->SetBranchAddress("_HcalEndcapPInsertHits_contributions", &_HcalEndcapPInsertHits_contributions_, &b__HcalEndcapPInsertHits_contributions_);
   fChain->SetBranchAddress("_HcalEndcapPInsertHits_contributions.index", &_HcalEndcapPInsertHits_contributions_index, &b__HcalEndcapPInsertHits_contributions_index);
   fChain->SetBranchAddress("_HcalEndcapPInsertHits_contributions.collectionID", &_HcalEndcapPInsertHits_contributions_collectionID, &b__HcalEndcapPInsertHits_contributions_collectionID);
   fChain->SetBranchAddress("HcalEndcapPInsertHitsContributions", &HcalEndcapPInsertHitsContributions_, &b_HcalEndcapPInsertHitsContributions_);
   fChain->SetBranchAddress("HcalEndcapPInsertHitsContributions.PDG", &HcalEndcapPInsertHitsContributions_PDG, &b_HcalEndcapPInsertHitsContributions_PDG);
   fChain->SetBranchAddress("HcalEndcapPInsertHitsContributions.energy", &HcalEndcapPInsertHitsContributions_energy, &b_HcalEndcapPInsertHitsContributions_energy);
   fChain->SetBranchAddress("HcalEndcapPInsertHitsContributions.time", &HcalEndcapPInsertHitsContributions_time, &b_HcalEndcapPInsertHitsContributions_time);
   fChain->SetBranchAddress("HcalEndcapPInsertHitsContributions.stepPosition.x", &HcalEndcapPInsertHitsContributions_stepPosition_x, &b_HcalEndcapPInsertHitsContributions_stepPosition_x);
   fChain->SetBranchAddress("HcalEndcapPInsertHitsContributions.stepPosition.y", &HcalEndcapPInsertHitsContributions_stepPosition_y, &b_HcalEndcapPInsertHitsContributions_stepPosition_y);
   fChain->SetBranchAddress("HcalEndcapPInsertHitsContributions.stepPosition.z", &HcalEndcapPInsertHitsContributions_stepPosition_z, &b_HcalEndcapPInsertHitsContributions_stepPosition_z);
   fChain->SetBranchAddress("_HcalEndcapPInsertHitsContributions_particle", &_HcalEndcapPInsertHitsContributions_particle_, &b__HcalEndcapPInsertHitsContributions_particle_);
   fChain->SetBranchAddress("_HcalEndcapPInsertHitsContributions_particle.index", &_HcalEndcapPInsertHitsContributions_particle_index, &b__HcalEndcapPInsertHitsContributions_particle_index);
   fChain->SetBranchAddress("_HcalEndcapPInsertHitsContributions_particle.collectionID", &_HcalEndcapPInsertHitsContributions_particle_collectionID, &b__HcalEndcapPInsertHitsContributions_particle_collectionID);
   fChain->SetBranchAddress("HcalFarForwardZDCHits", &HcalFarForwardZDCHits_, &b_HcalFarForwardZDCHits_);
   fChain->SetBranchAddress("HcalFarForwardZDCHits.cellID", &HcalFarForwardZDCHits_cellID, &b_HcalFarForwardZDCHits_cellID);
   fChain->SetBranchAddress("HcalFarForwardZDCHits.energy", &HcalFarForwardZDCHits_energy, &b_HcalFarForwardZDCHits_energy);
   fChain->SetBranchAddress("HcalFarForwardZDCHits.position.x", &HcalFarForwardZDCHits_position_x, &b_HcalFarForwardZDCHits_position_x);
   fChain->SetBranchAddress("HcalFarForwardZDCHits.position.y", &HcalFarForwardZDCHits_position_y, &b_HcalFarForwardZDCHits_position_y);
   fChain->SetBranchAddress("HcalFarForwardZDCHits.position.z", &HcalFarForwardZDCHits_position_z, &b_HcalFarForwardZDCHits_position_z);
   fChain->SetBranchAddress("HcalFarForwardZDCHits.contributions_begin", &HcalFarForwardZDCHits_contributions_begin, &b_HcalFarForwardZDCHits_contributions_begin);
   fChain->SetBranchAddress("HcalFarForwardZDCHits.contributions_end", &HcalFarForwardZDCHits_contributions_end, &b_HcalFarForwardZDCHits_contributions_end);
   fChain->SetBranchAddress("_HcalFarForwardZDCHits_contributions", &_HcalFarForwardZDCHits_contributions_, &b__HcalFarForwardZDCHits_contributions_);
   fChain->SetBranchAddress("_HcalFarForwardZDCHits_contributions.index", &_HcalFarForwardZDCHits_contributions_index, &b__HcalFarForwardZDCHits_contributions_index);
   fChain->SetBranchAddress("_HcalFarForwardZDCHits_contributions.collectionID", &_HcalFarForwardZDCHits_contributions_collectionID, &b__HcalFarForwardZDCHits_contributions_collectionID);
   fChain->SetBranchAddress("HcalFarForwardZDCHitsContributions", &HcalFarForwardZDCHitsContributions_, &b_HcalFarForwardZDCHitsContributions_);
   fChain->SetBranchAddress("HcalFarForwardZDCHitsContributions.PDG", &HcalFarForwardZDCHitsContributions_PDG, &b_HcalFarForwardZDCHitsContributions_PDG);
   fChain->SetBranchAddress("HcalFarForwardZDCHitsContributions.energy", &HcalFarForwardZDCHitsContributions_energy, &b_HcalFarForwardZDCHitsContributions_energy);
   fChain->SetBranchAddress("HcalFarForwardZDCHitsContributions.time", &HcalFarForwardZDCHitsContributions_time, &b_HcalFarForwardZDCHitsContributions_time);
   fChain->SetBranchAddress("HcalFarForwardZDCHitsContributions.stepPosition.x", &HcalFarForwardZDCHitsContributions_stepPosition_x, &b_HcalFarForwardZDCHitsContributions_stepPosition_x);
   fChain->SetBranchAddress("HcalFarForwardZDCHitsContributions.stepPosition.y", &HcalFarForwardZDCHitsContributions_stepPosition_y, &b_HcalFarForwardZDCHitsContributions_stepPosition_y);
   fChain->SetBranchAddress("HcalFarForwardZDCHitsContributions.stepPosition.z", &HcalFarForwardZDCHitsContributions_stepPosition_z, &b_HcalFarForwardZDCHitsContributions_stepPosition_z);
   fChain->SetBranchAddress("_HcalFarForwardZDCHitsContributions_particle", &_HcalFarForwardZDCHitsContributions_particle_, &b__HcalFarForwardZDCHitsContributions_particle_);
   fChain->SetBranchAddress("_HcalFarForwardZDCHitsContributions_particle.index", &_HcalFarForwardZDCHitsContributions_particle_index, &b__HcalFarForwardZDCHitsContributions_particle_index);
   fChain->SetBranchAddress("_HcalFarForwardZDCHitsContributions_particle.collectionID", &_HcalFarForwardZDCHitsContributions_particle_collectionID, &b__HcalFarForwardZDCHitsContributions_particle_collectionID);
   fChain->SetBranchAddress("LFHCALHits", &LFHCALHits_, &b_LFHCALHits_);
   fChain->SetBranchAddress("LFHCALHits.cellID", LFHCALHits_cellID, &b_LFHCALHits_cellID);
   fChain->SetBranchAddress("LFHCALHits.energy", LFHCALHits_energy, &b_LFHCALHits_energy);
   fChain->SetBranchAddress("LFHCALHits.position.x", LFHCALHits_position_x, &b_LFHCALHits_position_x);
   fChain->SetBranchAddress("LFHCALHits.position.y", LFHCALHits_position_y, &b_LFHCALHits_position_y);
   fChain->SetBranchAddress("LFHCALHits.position.z", LFHCALHits_position_z, &b_LFHCALHits_position_z);
   fChain->SetBranchAddress("LFHCALHits.contributions_begin", LFHCALHits_contributions_begin, &b_LFHCALHits_contributions_begin);
   fChain->SetBranchAddress("LFHCALHits.contributions_end", LFHCALHits_contributions_end, &b_LFHCALHits_contributions_end);
   fChain->SetBranchAddress("_LFHCALHits_contributions", &_LFHCALHits_contributions_, &b__LFHCALHits_contributions_);
   fChain->SetBranchAddress("_LFHCALHits_contributions.index", _LFHCALHits_contributions_index, &b__LFHCALHits_contributions_index);
   fChain->SetBranchAddress("_LFHCALHits_contributions.collectionID", _LFHCALHits_contributions_collectionID, &b__LFHCALHits_contributions_collectionID);
   fChain->SetBranchAddress("LFHCALHitsContributions", &LFHCALHitsContributions_, &b_LFHCALHitsContributions_);
   fChain->SetBranchAddress("LFHCALHitsContributions.PDG", LFHCALHitsContributions_PDG, &b_LFHCALHitsContributions_PDG);
   fChain->SetBranchAddress("LFHCALHitsContributions.energy", LFHCALHitsContributions_energy, &b_LFHCALHitsContributions_energy);
   fChain->SetBranchAddress("LFHCALHitsContributions.time", LFHCALHitsContributions_time, &b_LFHCALHitsContributions_time);
   fChain->SetBranchAddress("LFHCALHitsContributions.stepPosition.x", LFHCALHitsContributions_stepPosition_x, &b_LFHCALHitsContributions_stepPosition_x);
   fChain->SetBranchAddress("LFHCALHitsContributions.stepPosition.y", LFHCALHitsContributions_stepPosition_y, &b_LFHCALHitsContributions_stepPosition_y);
   fChain->SetBranchAddress("LFHCALHitsContributions.stepPosition.z", LFHCALHitsContributions_stepPosition_z, &b_LFHCALHitsContributions_stepPosition_z);
   fChain->SetBranchAddress("_LFHCALHitsContributions_particle", &_LFHCALHitsContributions_particle_, &b__LFHCALHitsContributions_particle_);
   fChain->SetBranchAddress("_LFHCALHitsContributions_particle.index", _LFHCALHitsContributions_particle_index, &b__LFHCALHitsContributions_particle_index);
   fChain->SetBranchAddress("_LFHCALHitsContributions_particle.collectionID", _LFHCALHitsContributions_particle_collectionID, &b__LFHCALHitsContributions_particle_collectionID);
   fChain->SetBranchAddress("LumiDirectPCALHits", &LumiDirectPCALHits_, &b_LumiDirectPCALHits_);
   fChain->SetBranchAddress("LumiDirectPCALHits.cellID", &LumiDirectPCALHits_cellID, &b_LumiDirectPCALHits_cellID);
   fChain->SetBranchAddress("LumiDirectPCALHits.energy", &LumiDirectPCALHits_energy, &b_LumiDirectPCALHits_energy);
   fChain->SetBranchAddress("LumiDirectPCALHits.position.x", &LumiDirectPCALHits_position_x, &b_LumiDirectPCALHits_position_x);
   fChain->SetBranchAddress("LumiDirectPCALHits.position.y", &LumiDirectPCALHits_position_y, &b_LumiDirectPCALHits_position_y);
   fChain->SetBranchAddress("LumiDirectPCALHits.position.z", &LumiDirectPCALHits_position_z, &b_LumiDirectPCALHits_position_z);
   fChain->SetBranchAddress("LumiDirectPCALHits.contributions_begin", &LumiDirectPCALHits_contributions_begin, &b_LumiDirectPCALHits_contributions_begin);
   fChain->SetBranchAddress("LumiDirectPCALHits.contributions_end", &LumiDirectPCALHits_contributions_end, &b_LumiDirectPCALHits_contributions_end);
   fChain->SetBranchAddress("_LumiDirectPCALHits_contributions", &_LumiDirectPCALHits_contributions_, &b__LumiDirectPCALHits_contributions_);
   fChain->SetBranchAddress("_LumiDirectPCALHits_contributions.index", &_LumiDirectPCALHits_contributions_index, &b__LumiDirectPCALHits_contributions_index);
   fChain->SetBranchAddress("_LumiDirectPCALHits_contributions.collectionID", &_LumiDirectPCALHits_contributions_collectionID, &b__LumiDirectPCALHits_contributions_collectionID);
   fChain->SetBranchAddress("LumiDirectPCALHitsContributions", &LumiDirectPCALHitsContributions_, &b_LumiDirectPCALHitsContributions_);
   fChain->SetBranchAddress("LumiDirectPCALHitsContributions.PDG", &LumiDirectPCALHitsContributions_PDG, &b_LumiDirectPCALHitsContributions_PDG);
   fChain->SetBranchAddress("LumiDirectPCALHitsContributions.energy", &LumiDirectPCALHitsContributions_energy, &b_LumiDirectPCALHitsContributions_energy);
   fChain->SetBranchAddress("LumiDirectPCALHitsContributions.time", &LumiDirectPCALHitsContributions_time, &b_LumiDirectPCALHitsContributions_time);
   fChain->SetBranchAddress("LumiDirectPCALHitsContributions.stepPosition.x", &LumiDirectPCALHitsContributions_stepPosition_x, &b_LumiDirectPCALHitsContributions_stepPosition_x);
   fChain->SetBranchAddress("LumiDirectPCALHitsContributions.stepPosition.y", &LumiDirectPCALHitsContributions_stepPosition_y, &b_LumiDirectPCALHitsContributions_stepPosition_y);
   fChain->SetBranchAddress("LumiDirectPCALHitsContributions.stepPosition.z", &LumiDirectPCALHitsContributions_stepPosition_z, &b_LumiDirectPCALHitsContributions_stepPosition_z);
   fChain->SetBranchAddress("_LumiDirectPCALHitsContributions_particle", &_LumiDirectPCALHitsContributions_particle_, &b__LumiDirectPCALHitsContributions_particle_);
   fChain->SetBranchAddress("_LumiDirectPCALHitsContributions_particle.index", &_LumiDirectPCALHitsContributions_particle_index, &b__LumiDirectPCALHitsContributions_particle_index);
   fChain->SetBranchAddress("_LumiDirectPCALHitsContributions_particle.collectionID", &_LumiDirectPCALHitsContributions_particle_collectionID, &b__LumiDirectPCALHitsContributions_particle_collectionID);
   fChain->SetBranchAddress("LumiSpecTrackerHits", &LumiSpecTrackerHits_, &b_LumiSpecTrackerHits_);
   fChain->SetBranchAddress("LumiSpecTrackerHits.cellID", &LumiSpecTrackerHits_cellID, &b_LumiSpecTrackerHits_cellID);
   fChain->SetBranchAddress("LumiSpecTrackerHits.EDep", &LumiSpecTrackerHits_EDep, &b_LumiSpecTrackerHits_EDep);
   fChain->SetBranchAddress("LumiSpecTrackerHits.time", &LumiSpecTrackerHits_time, &b_LumiSpecTrackerHits_time);
   fChain->SetBranchAddress("LumiSpecTrackerHits.pathLength", &LumiSpecTrackerHits_pathLength, &b_LumiSpecTrackerHits_pathLength);
   fChain->SetBranchAddress("LumiSpecTrackerHits.quality", &LumiSpecTrackerHits_quality, &b_LumiSpecTrackerHits_quality);
   fChain->SetBranchAddress("LumiSpecTrackerHits.position.x", &LumiSpecTrackerHits_position_x, &b_LumiSpecTrackerHits_position_x);
   fChain->SetBranchAddress("LumiSpecTrackerHits.position.y", &LumiSpecTrackerHits_position_y, &b_LumiSpecTrackerHits_position_y);
   fChain->SetBranchAddress("LumiSpecTrackerHits.position.z", &LumiSpecTrackerHits_position_z, &b_LumiSpecTrackerHits_position_z);
   fChain->SetBranchAddress("LumiSpecTrackerHits.momentum.x", &LumiSpecTrackerHits_momentum_x, &b_LumiSpecTrackerHits_momentum_x);
   fChain->SetBranchAddress("LumiSpecTrackerHits.momentum.y", &LumiSpecTrackerHits_momentum_y, &b_LumiSpecTrackerHits_momentum_y);
   fChain->SetBranchAddress("LumiSpecTrackerHits.momentum.z", &LumiSpecTrackerHits_momentum_z, &b_LumiSpecTrackerHits_momentum_z);
   fChain->SetBranchAddress("_LumiSpecTrackerHits_MCParticle", &_LumiSpecTrackerHits_MCParticle_, &b__LumiSpecTrackerHits_MCParticle_);
   fChain->SetBranchAddress("_LumiSpecTrackerHits_MCParticle.index", &_LumiSpecTrackerHits_MCParticle_index, &b__LumiSpecTrackerHits_MCParticle_index);
   fChain->SetBranchAddress("_LumiSpecTrackerHits_MCParticle.collectionID", &_LumiSpecTrackerHits_MCParticle_collectionID, &b__LumiSpecTrackerHits_MCParticle_collectionID);
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
   fChain->SetBranchAddress("MPGDBarrelHits", &MPGDBarrelHits_, &b_MPGDBarrelHits_);
   fChain->SetBranchAddress("MPGDBarrelHits.cellID", MPGDBarrelHits_cellID, &b_MPGDBarrelHits_cellID);
   fChain->SetBranchAddress("MPGDBarrelHits.EDep", MPGDBarrelHits_EDep, &b_MPGDBarrelHits_EDep);
   fChain->SetBranchAddress("MPGDBarrelHits.time", MPGDBarrelHits_time, &b_MPGDBarrelHits_time);
   fChain->SetBranchAddress("MPGDBarrelHits.pathLength", MPGDBarrelHits_pathLength, &b_MPGDBarrelHits_pathLength);
   fChain->SetBranchAddress("MPGDBarrelHits.quality", MPGDBarrelHits_quality, &b_MPGDBarrelHits_quality);
   fChain->SetBranchAddress("MPGDBarrelHits.position.x", MPGDBarrelHits_position_x, &b_MPGDBarrelHits_position_x);
   fChain->SetBranchAddress("MPGDBarrelHits.position.y", MPGDBarrelHits_position_y, &b_MPGDBarrelHits_position_y);
   fChain->SetBranchAddress("MPGDBarrelHits.position.z", MPGDBarrelHits_position_z, &b_MPGDBarrelHits_position_z);
   fChain->SetBranchAddress("MPGDBarrelHits.momentum.x", MPGDBarrelHits_momentum_x, &b_MPGDBarrelHits_momentum_x);
   fChain->SetBranchAddress("MPGDBarrelHits.momentum.y", MPGDBarrelHits_momentum_y, &b_MPGDBarrelHits_momentum_y);
   fChain->SetBranchAddress("MPGDBarrelHits.momentum.z", MPGDBarrelHits_momentum_z, &b_MPGDBarrelHits_momentum_z);
   fChain->SetBranchAddress("_MPGDBarrelHits_MCParticle", &_MPGDBarrelHits_MCParticle_, &b__MPGDBarrelHits_MCParticle_);
   fChain->SetBranchAddress("_MPGDBarrelHits_MCParticle.index", _MPGDBarrelHits_MCParticle_index, &b__MPGDBarrelHits_MCParticle_index);
   fChain->SetBranchAddress("_MPGDBarrelHits_MCParticle.collectionID", _MPGDBarrelHits_MCParticle_collectionID, &b__MPGDBarrelHits_MCParticle_collectionID);
   fChain->SetBranchAddress("MPGDDIRCHits", &MPGDDIRCHits_, &b_MPGDDIRCHits_);
   fChain->SetBranchAddress("MPGDDIRCHits.cellID", MPGDDIRCHits_cellID, &b_MPGDDIRCHits_cellID);
   fChain->SetBranchAddress("MPGDDIRCHits.EDep", MPGDDIRCHits_EDep, &b_MPGDDIRCHits_EDep);
   fChain->SetBranchAddress("MPGDDIRCHits.time", MPGDDIRCHits_time, &b_MPGDDIRCHits_time);
   fChain->SetBranchAddress("MPGDDIRCHits.pathLength", MPGDDIRCHits_pathLength, &b_MPGDDIRCHits_pathLength);
   fChain->SetBranchAddress("MPGDDIRCHits.quality", MPGDDIRCHits_quality, &b_MPGDDIRCHits_quality);
   fChain->SetBranchAddress("MPGDDIRCHits.position.x", MPGDDIRCHits_position_x, &b_MPGDDIRCHits_position_x);
   fChain->SetBranchAddress("MPGDDIRCHits.position.y", MPGDDIRCHits_position_y, &b_MPGDDIRCHits_position_y);
   fChain->SetBranchAddress("MPGDDIRCHits.position.z", MPGDDIRCHits_position_z, &b_MPGDDIRCHits_position_z);
   fChain->SetBranchAddress("MPGDDIRCHits.momentum.x", MPGDDIRCHits_momentum_x, &b_MPGDDIRCHits_momentum_x);
   fChain->SetBranchAddress("MPGDDIRCHits.momentum.y", MPGDDIRCHits_momentum_y, &b_MPGDDIRCHits_momentum_y);
   fChain->SetBranchAddress("MPGDDIRCHits.momentum.z", MPGDDIRCHits_momentum_z, &b_MPGDDIRCHits_momentum_z);
   fChain->SetBranchAddress("_MPGDDIRCHits_MCParticle", &_MPGDDIRCHits_MCParticle_, &b__MPGDDIRCHits_MCParticle_);
   fChain->SetBranchAddress("_MPGDDIRCHits_MCParticle.index", _MPGDDIRCHits_MCParticle_index, &b__MPGDDIRCHits_MCParticle_index);
   fChain->SetBranchAddress("_MPGDDIRCHits_MCParticle.collectionID", _MPGDDIRCHits_MCParticle_collectionID, &b__MPGDDIRCHits_MCParticle_collectionID);
   fChain->SetBranchAddress("PFRICHHits", &PFRICHHits_, &b_PFRICHHits_);
   fChain->SetBranchAddress("PFRICHHits.cellID", &PFRICHHits_cellID, &b_PFRICHHits_cellID);
   fChain->SetBranchAddress("PFRICHHits.EDep", &PFRICHHits_EDep, &b_PFRICHHits_EDep);
   fChain->SetBranchAddress("PFRICHHits.time", &PFRICHHits_time, &b_PFRICHHits_time);
   fChain->SetBranchAddress("PFRICHHits.pathLength", &PFRICHHits_pathLength, &b_PFRICHHits_pathLength);
   fChain->SetBranchAddress("PFRICHHits.quality", &PFRICHHits_quality, &b_PFRICHHits_quality);
   fChain->SetBranchAddress("PFRICHHits.position.x", &PFRICHHits_position_x, &b_PFRICHHits_position_x);
   fChain->SetBranchAddress("PFRICHHits.position.y", &PFRICHHits_position_y, &b_PFRICHHits_position_y);
   fChain->SetBranchAddress("PFRICHHits.position.z", &PFRICHHits_position_z, &b_PFRICHHits_position_z);
   fChain->SetBranchAddress("PFRICHHits.momentum.x", &PFRICHHits_momentum_x, &b_PFRICHHits_momentum_x);
   fChain->SetBranchAddress("PFRICHHits.momentum.y", &PFRICHHits_momentum_y, &b_PFRICHHits_momentum_y);
   fChain->SetBranchAddress("PFRICHHits.momentum.z", &PFRICHHits_momentum_z, &b_PFRICHHits_momentum_z);
   fChain->SetBranchAddress("_PFRICHHits_MCParticle", &_PFRICHHits_MCParticle_, &b__PFRICHHits_MCParticle_);
   fChain->SetBranchAddress("_PFRICHHits_MCParticle.index", &_PFRICHHits_MCParticle_index, &b__PFRICHHits_MCParticle_index);
   fChain->SetBranchAddress("_PFRICHHits_MCParticle.collectionID", &_PFRICHHits_MCParticle_collectionID, &b__PFRICHHits_MCParticle_collectionID);
   fChain->SetBranchAddress("SiBarrelHits", &SiBarrelHits_, &b_SiBarrelHits_);
   fChain->SetBranchAddress("SiBarrelHits.cellID", SiBarrelHits_cellID, &b_SiBarrelHits_cellID);
   fChain->SetBranchAddress("SiBarrelHits.EDep", SiBarrelHits_EDep, &b_SiBarrelHits_EDep);
   fChain->SetBranchAddress("SiBarrelHits.time", SiBarrelHits_time, &b_SiBarrelHits_time);
   fChain->SetBranchAddress("SiBarrelHits.pathLength", SiBarrelHits_pathLength, &b_SiBarrelHits_pathLength);
   fChain->SetBranchAddress("SiBarrelHits.quality", SiBarrelHits_quality, &b_SiBarrelHits_quality);
   fChain->SetBranchAddress("SiBarrelHits.position.x", SiBarrelHits_position_x, &b_SiBarrelHits_position_x);
   fChain->SetBranchAddress("SiBarrelHits.position.y", SiBarrelHits_position_y, &b_SiBarrelHits_position_y);
   fChain->SetBranchAddress("SiBarrelHits.position.z", SiBarrelHits_position_z, &b_SiBarrelHits_position_z);
   fChain->SetBranchAddress("SiBarrelHits.momentum.x", SiBarrelHits_momentum_x, &b_SiBarrelHits_momentum_x);
   fChain->SetBranchAddress("SiBarrelHits.momentum.y", SiBarrelHits_momentum_y, &b_SiBarrelHits_momentum_y);
   fChain->SetBranchAddress("SiBarrelHits.momentum.z", SiBarrelHits_momentum_z, &b_SiBarrelHits_momentum_z);
   fChain->SetBranchAddress("_SiBarrelHits_MCParticle", &_SiBarrelHits_MCParticle_, &b__SiBarrelHits_MCParticle_);
   fChain->SetBranchAddress("_SiBarrelHits_MCParticle.index", _SiBarrelHits_MCParticle_index, &b__SiBarrelHits_MCParticle_index);
   fChain->SetBranchAddress("_SiBarrelHits_MCParticle.collectionID", _SiBarrelHits_MCParticle_collectionID, &b__SiBarrelHits_MCParticle_collectionID);
   fChain->SetBranchAddress("TaggerTrackerHits", &TaggerTrackerHits_, &b_TaggerTrackerHits_);
   fChain->SetBranchAddress("TaggerTrackerHits.cellID", &TaggerTrackerHits_cellID, &b_TaggerTrackerHits_cellID);
   fChain->SetBranchAddress("TaggerTrackerHits.EDep", &TaggerTrackerHits_EDep, &b_TaggerTrackerHits_EDep);
   fChain->SetBranchAddress("TaggerTrackerHits.time", &TaggerTrackerHits_time, &b_TaggerTrackerHits_time);
   fChain->SetBranchAddress("TaggerTrackerHits.pathLength", &TaggerTrackerHits_pathLength, &b_TaggerTrackerHits_pathLength);
   fChain->SetBranchAddress("TaggerTrackerHits.quality", &TaggerTrackerHits_quality, &b_TaggerTrackerHits_quality);
   fChain->SetBranchAddress("TaggerTrackerHits.position.x", &TaggerTrackerHits_position_x, &b_TaggerTrackerHits_position_x);
   fChain->SetBranchAddress("TaggerTrackerHits.position.y", &TaggerTrackerHits_position_y, &b_TaggerTrackerHits_position_y);
   fChain->SetBranchAddress("TaggerTrackerHits.position.z", &TaggerTrackerHits_position_z, &b_TaggerTrackerHits_position_z);
   fChain->SetBranchAddress("TaggerTrackerHits.momentum.x", &TaggerTrackerHits_momentum_x, &b_TaggerTrackerHits_momentum_x);
   fChain->SetBranchAddress("TaggerTrackerHits.momentum.y", &TaggerTrackerHits_momentum_y, &b_TaggerTrackerHits_momentum_y);
   fChain->SetBranchAddress("TaggerTrackerHits.momentum.z", &TaggerTrackerHits_momentum_z, &b_TaggerTrackerHits_momentum_z);
   fChain->SetBranchAddress("_TaggerTrackerHits_MCParticle", &_TaggerTrackerHits_MCParticle_, &b__TaggerTrackerHits_MCParticle_);
   fChain->SetBranchAddress("_TaggerTrackerHits_MCParticle.index", &_TaggerTrackerHits_MCParticle_index, &b__TaggerTrackerHits_MCParticle_index);
   fChain->SetBranchAddress("_TaggerTrackerHits_MCParticle.collectionID", &_TaggerTrackerHits_MCParticle_collectionID, &b__TaggerTrackerHits_MCParticle_collectionID);
   fChain->SetBranchAddress("TOFBarrelHits", &TOFBarrelHits_, &b_TOFBarrelHits_);
   fChain->SetBranchAddress("TOFBarrelHits.cellID", TOFBarrelHits_cellID, &b_TOFBarrelHits_cellID);
   fChain->SetBranchAddress("TOFBarrelHits.EDep", TOFBarrelHits_EDep, &b_TOFBarrelHits_EDep);
   fChain->SetBranchAddress("TOFBarrelHits.time", TOFBarrelHits_time, &b_TOFBarrelHits_time);
   fChain->SetBranchAddress("TOFBarrelHits.pathLength", TOFBarrelHits_pathLength, &b_TOFBarrelHits_pathLength);
   fChain->SetBranchAddress("TOFBarrelHits.quality", TOFBarrelHits_quality, &b_TOFBarrelHits_quality);
   fChain->SetBranchAddress("TOFBarrelHits.position.x", TOFBarrelHits_position_x, &b_TOFBarrelHits_position_x);
   fChain->SetBranchAddress("TOFBarrelHits.position.y", TOFBarrelHits_position_y, &b_TOFBarrelHits_position_y);
   fChain->SetBranchAddress("TOFBarrelHits.position.z", TOFBarrelHits_position_z, &b_TOFBarrelHits_position_z);
   fChain->SetBranchAddress("TOFBarrelHits.momentum.x", TOFBarrelHits_momentum_x, &b_TOFBarrelHits_momentum_x);
   fChain->SetBranchAddress("TOFBarrelHits.momentum.y", TOFBarrelHits_momentum_y, &b_TOFBarrelHits_momentum_y);
   fChain->SetBranchAddress("TOFBarrelHits.momentum.z", TOFBarrelHits_momentum_z, &b_TOFBarrelHits_momentum_z);
   fChain->SetBranchAddress("_TOFBarrelHits_MCParticle", &_TOFBarrelHits_MCParticle_, &b__TOFBarrelHits_MCParticle_);
   fChain->SetBranchAddress("_TOFBarrelHits_MCParticle.index", _TOFBarrelHits_MCParticle_index, &b__TOFBarrelHits_MCParticle_index);
   fChain->SetBranchAddress("_TOFBarrelHits_MCParticle.collectionID", _TOFBarrelHits_MCParticle_collectionID, &b__TOFBarrelHits_MCParticle_collectionID);
   fChain->SetBranchAddress("TOFEndcapHits", &TOFEndcapHits_, &b_TOFEndcapHits_);
   fChain->SetBranchAddress("TOFEndcapHits.cellID", TOFEndcapHits_cellID, &b_TOFEndcapHits_cellID);
   fChain->SetBranchAddress("TOFEndcapHits.EDep", TOFEndcapHits_EDep, &b_TOFEndcapHits_EDep);
   fChain->SetBranchAddress("TOFEndcapHits.time", TOFEndcapHits_time, &b_TOFEndcapHits_time);
   fChain->SetBranchAddress("TOFEndcapHits.pathLength", TOFEndcapHits_pathLength, &b_TOFEndcapHits_pathLength);
   fChain->SetBranchAddress("TOFEndcapHits.quality", TOFEndcapHits_quality, &b_TOFEndcapHits_quality);
   fChain->SetBranchAddress("TOFEndcapHits.position.x", TOFEndcapHits_position_x, &b_TOFEndcapHits_position_x);
   fChain->SetBranchAddress("TOFEndcapHits.position.y", TOFEndcapHits_position_y, &b_TOFEndcapHits_position_y);
   fChain->SetBranchAddress("TOFEndcapHits.position.z", TOFEndcapHits_position_z, &b_TOFEndcapHits_position_z);
   fChain->SetBranchAddress("TOFEndcapHits.momentum.x", TOFEndcapHits_momentum_x, &b_TOFEndcapHits_momentum_x);
   fChain->SetBranchAddress("TOFEndcapHits.momentum.y", TOFEndcapHits_momentum_y, &b_TOFEndcapHits_momentum_y);
   fChain->SetBranchAddress("TOFEndcapHits.momentum.z", TOFEndcapHits_momentum_z, &b_TOFEndcapHits_momentum_z);
   fChain->SetBranchAddress("_TOFEndcapHits_MCParticle", &_TOFEndcapHits_MCParticle_, &b__TOFEndcapHits_MCParticle_);
   fChain->SetBranchAddress("_TOFEndcapHits_MCParticle.index", _TOFEndcapHits_MCParticle_index, &b__TOFEndcapHits_MCParticle_index);
   fChain->SetBranchAddress("_TOFEndcapHits_MCParticle.collectionID", _TOFEndcapHits_MCParticle_collectionID, &b__TOFEndcapHits_MCParticle_collectionID);
   fChain->SetBranchAddress("TrackerEndcapHits", &TrackerEndcapHits_, &b_TrackerEndcapHits_);
   fChain->SetBranchAddress("TrackerEndcapHits.cellID", TrackerEndcapHits_cellID, &b_TrackerEndcapHits_cellID);
   fChain->SetBranchAddress("TrackerEndcapHits.EDep", TrackerEndcapHits_EDep, &b_TrackerEndcapHits_EDep);
   fChain->SetBranchAddress("TrackerEndcapHits.time", TrackerEndcapHits_time, &b_TrackerEndcapHits_time);
   fChain->SetBranchAddress("TrackerEndcapHits.pathLength", TrackerEndcapHits_pathLength, &b_TrackerEndcapHits_pathLength);
   fChain->SetBranchAddress("TrackerEndcapHits.quality", TrackerEndcapHits_quality, &b_TrackerEndcapHits_quality);
   fChain->SetBranchAddress("TrackerEndcapHits.position.x", TrackerEndcapHits_position_x, &b_TrackerEndcapHits_position_x);
   fChain->SetBranchAddress("TrackerEndcapHits.position.y", TrackerEndcapHits_position_y, &b_TrackerEndcapHits_position_y);
   fChain->SetBranchAddress("TrackerEndcapHits.position.z", TrackerEndcapHits_position_z, &b_TrackerEndcapHits_position_z);
   fChain->SetBranchAddress("TrackerEndcapHits.momentum.x", TrackerEndcapHits_momentum_x, &b_TrackerEndcapHits_momentum_x);
   fChain->SetBranchAddress("TrackerEndcapHits.momentum.y", TrackerEndcapHits_momentum_y, &b_TrackerEndcapHits_momentum_y);
   fChain->SetBranchAddress("TrackerEndcapHits.momentum.z", TrackerEndcapHits_momentum_z, &b_TrackerEndcapHits_momentum_z);
   fChain->SetBranchAddress("_TrackerEndcapHits_MCParticle", &_TrackerEndcapHits_MCParticle_, &b__TrackerEndcapHits_MCParticle_);
   fChain->SetBranchAddress("_TrackerEndcapHits_MCParticle.index", _TrackerEndcapHits_MCParticle_index, &b__TrackerEndcapHits_MCParticle_index);
   fChain->SetBranchAddress("_TrackerEndcapHits_MCParticle.collectionID", _TrackerEndcapHits_MCParticle_collectionID, &b__TrackerEndcapHits_MCParticle_collectionID);
   fChain->SetBranchAddress("VertexBarrelHits", &VertexBarrelHits_, &b_VertexBarrelHits_);
   fChain->SetBranchAddress("VertexBarrelHits.cellID", VertexBarrelHits_cellID, &b_VertexBarrelHits_cellID);
   fChain->SetBranchAddress("VertexBarrelHits.EDep", VertexBarrelHits_EDep, &b_VertexBarrelHits_EDep);
   fChain->SetBranchAddress("VertexBarrelHits.time", VertexBarrelHits_time, &b_VertexBarrelHits_time);
   fChain->SetBranchAddress("VertexBarrelHits.pathLength", VertexBarrelHits_pathLength, &b_VertexBarrelHits_pathLength);
   fChain->SetBranchAddress("VertexBarrelHits.quality", VertexBarrelHits_quality, &b_VertexBarrelHits_quality);
   fChain->SetBranchAddress("VertexBarrelHits.position.x", VertexBarrelHits_position_x, &b_VertexBarrelHits_position_x);
   fChain->SetBranchAddress("VertexBarrelHits.position.y", VertexBarrelHits_position_y, &b_VertexBarrelHits_position_y);
   fChain->SetBranchAddress("VertexBarrelHits.position.z", VertexBarrelHits_position_z, &b_VertexBarrelHits_position_z);
   fChain->SetBranchAddress("VertexBarrelHits.momentum.x", VertexBarrelHits_momentum_x, &b_VertexBarrelHits_momentum_x);
   fChain->SetBranchAddress("VertexBarrelHits.momentum.y", VertexBarrelHits_momentum_y, &b_VertexBarrelHits_momentum_y);
   fChain->SetBranchAddress("VertexBarrelHits.momentum.z", VertexBarrelHits_momentum_z, &b_VertexBarrelHits_momentum_z);
   fChain->SetBranchAddress("_VertexBarrelHits_MCParticle", &_VertexBarrelHits_MCParticle_, &b__VertexBarrelHits_MCParticle_);
   fChain->SetBranchAddress("_VertexBarrelHits_MCParticle.index", _VertexBarrelHits_MCParticle_index, &b__VertexBarrelHits_MCParticle_index);
   fChain->SetBranchAddress("_VertexBarrelHits_MCParticle.collectionID", _VertexBarrelHits_MCParticle_collectionID, &b__VertexBarrelHits_MCParticle_collectionID);
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

Bool_t mu_endpoint::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void mu_endpoint::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t mu_endpoint::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef mu_endpoint_cxx
