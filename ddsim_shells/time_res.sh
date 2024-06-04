
npsim --steeringFile ./steering/steer.py --compactFile $DETECTOR_PATH/epic_klmws_only.xml --macroFile ddsim_shells/myvis.mac --runType "batch" -G -N 1000 --gun.particle "mu-"  --outputFile root_files/time_res/one_segment_test/mu_5GeV_1kevents_1_5m_1cm_3cm_test.edm4hep.root --part.userParticleHandler="" 

