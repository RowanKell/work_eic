#!/bin/bash
# Submit SLURM jobs for conventional muID benchmark.
# Generates mu-/pi- pairs at multiple momenta, then runs the analysis.
#
# Usage: ./submit_benchmark_muID.sh
# Edit the variables below to change momenta, events, etc.

MOMENTA=(1 5)       # GeV
N_EVENTS=5000             # events per particle per momentum
THETA_MIN=78              # degrees
THETA_MAX=102             # degrees

WORK_EIC="/hpc/group/vossenlab/rck32/eic/work_eic"
EIC_SHELL="/hpc/group/vossenlab/rck32/eic/eic-shell"
OUTDIR="${WORK_EIC}/root_files/benchmark_muID"

mkdir -p ${OUTDIR}
mkdir -p ${WORK_EIC}/slurm/logs

echo "Submitting muID benchmark jobs for momenta: ${MOMENTA[@]} GeV"
echo "Events per particle: ${N_EVENTS}"
echo "Output directory: ${OUTDIR}"

JOB_IDS=()

for P in "${MOMENTA[@]}"; do
    JOB_NAME="muID_bench_${P}GeV"

    SLURM_SCRIPT="${WORK_EIC}/slurm/shells/benchmark_muID_${P}GeV.slurm"
    mkdir -p ${WORK_EIC}/slurm/shells

    cat > ${SLURM_SCRIPT} << SLURM_EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=vossenlab
#SBATCH --partition=common
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --output=${WORK_EIC}/slurm/logs/benchmark_muID_${P}GeV_%j.out
#SBATCH --error=${WORK_EIC}/slurm/logs/benchmark_muID_${P}GeV_%j.err

cd ${WORK_EIC}

cat << EOF | ${EIC_SHELL}
source ${WORK_EIC}/setup.sh
source /hpc/group/vossenlab/rck32/eic/epic_klm/install/setup.sh
bash ddsim_shells/benchmark_muID.sh ${P} ${N_EVENTS} ${THETA_MIN} ${THETA_MAX} ${OUTDIR}
EOF
SLURM_EOF

    JOB_ID=$(sbatch ${SLURM_SCRIPT} | awk '{print $4}')
    echo "  Submitted ${P} GeV job: ${JOB_ID}"
    JOB_IDS+=("${JOB_ID}")
done

echo ""
echo "All jobs submitted. Job IDs: ${JOB_IDS[@]}"
echo ""
echo "Monitor with: squeue -u \$USER"
echo ""
echo "After all jobs complete, run the analysis:"
echo "  source /hpc/group/vossenlab/rck32/ML_venv/bin/activate"
echo "  cd ${WORK_EIC}/macros/Timing_estimation"
echo "  python conventional_muID_benchmark.py \\"
echo "    --mu_files ${OUTDIR}/benchmark_mu-_p_1GeV_${N_EVENTS}events.edm4hep.root,${OUTDIR}/benchmark_mu-_p_2GeV_${N_EVENTS}events.edm4hep.root,${OUTDIR}/benchmark_mu-_p_5GeV_${N_EVENTS}events.edm4hep.root,${OUTDIR}/benchmark_mu-_p_10GeV_${N_EVENTS}events.edm4hep.root \\"
echo "    --pi_files ${OUTDIR}/benchmark_pi-_p_1GeV_${N_EVENTS}events.edm4hep.root,${OUTDIR}/benchmark_pi-_p_2GeV_${N_EVENTS}events.edm4hep.root,${OUTDIR}/benchmark_pi-_p_5GeV_${N_EVENTS}events.edm4hep.root,${OUTDIR}/benchmark_pi-_p_10GeV_${N_EVENTS}events.edm4hep.root \\"
echo "    --p_labels '1 GeV,2 GeV,5 GeV,10 GeV' \\"
echo "    --output ${OUTDIR}/conventional_muID_roc.png"
