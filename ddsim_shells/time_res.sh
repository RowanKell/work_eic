npsim --steeringFile ./steering/sensor_sensitive/variation_pos_keepALL.py --compactFile $DETECTOR_PATH/epic_klmws_only.xml --macroFile ddsim_shells/myvis.mac --runType "batch" -G -N 500 --gun.particle "mu-"  --outputFile root_files/test/mu_w_QE_scint.edm4hep.root --part.userParticleHandler="" 

