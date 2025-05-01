#!/bin/bash

#Set your work_eic location
export WORK_EIC='/hpc/group/vossenlab/rck32/eic/work_eic/'

EPIC_DIR='/hpc/group/vossenlab/rck32/eic/epic_klm/'
EIC_SHELL_DIR='/hpc/group/vossenlab/rck32/eic'
ML_VENV_DIR="/hpc/group/vossenlab/rck32/ML_venv/"

export MAIL_USER="rck32@duke.edu"
export ML_VENV_HOME=$ML_VENV_DIR

export EIC_SHELL_HOME=$EIC_SHELL_DIR
export EPIC_HOME=$EPIC_DIR

export DETECTOR_PATH=$EPIC_HOME
# export DETECTOR_CONFIG=epic_klmws_w_solenoid
export DETECTOR_CONFIG=epic_klmws_only