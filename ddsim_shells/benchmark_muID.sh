#!/bin/bash
# Generate mu- and pi- simulation files for conventional muID benchmark.
# Matches genMomentumScan_job.sh parameters exactly (no steering file, pure CLI).
#
# Usage: ./benchmark_muID.sh <momentum_GeV> <n_events> <theta_min_deg> <theta_max_deg> <output_dir>
# Example: ./benchmark_muID.sh 5 500 78 102 root_files/benchmark_muID
#
# Runs inside eic-shell. Source setup.sh and epic_klm install/setup.sh first.

if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <momentum_GeV> <n_events> <theta_min_deg> <theta_max_deg> <output_dir>"
    exit 1
fi

P=$1
N=$2
THETA_MIN=$3
THETA_MAX=$4
OUTDIR=$5

mkdir -p $OUTDIR

for PARTICLE in "mu-" "pi-"; do
    OUTFILE="${OUTDIR}/benchmark_${PARTICLE}_p_${P}GeV_${N}events.edm4hep.root"
    echo "Running DDSim: ${PARTICLE} at ${P} GeV, ${N} events, theta=[${THETA_MIN},${THETA_MAX}] deg"
    echo "  Output: ${OUTFILE}"

    ddsim \
        --compactFile ${DETECTOR_PATH}/${DETECTOR_CONFIG}.xml \
        --runType "batch" \
        -G \
        -N ${N} \
        --gun.particle "${PARTICLE}" \
        --gun.momentumMin "${P}*GeV" \
        --gun.momentumMax "${P}*GeV" \
        --gun.thetaMin "${THETA_MIN}*deg" \
        --gun.thetaMax "${THETA_MAX}*deg" \
        --gun.distribution "uniform" \
        --physics.list "FTFP_BERT" \
        --part.userParticleHandler="" \
        --outputFile "${OUTFILE}"

    echo "Done with ${PARTICLE} at ${P} GeV"
done
