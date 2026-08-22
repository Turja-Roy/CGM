#!/bin/bash
#SBATCH -J reanalyze                 # Job name
#SBATCH -o logs/reanalyze_%A_%a.out  # Output file (%A=job ID, %a=array task ID)
#SBATCH -e logs/reanalyze_%A_%a.err  # Error file
#SBATCH -p normal                    # Partition/queue (Lonestar6)
#SBATCH -N 1                         # One whole node per task
#SBATCH -n 48                        # Cores
#SBATCH -t 04:00:00                  # Time limit
#SBATCH --array=1-40                 # 10 sims (p1/p2) x 4 existing snapshots
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=turja.roy@uta.edu

# ====================================================================
# ANALYZE-ONLY pass over the p1/p2 spectra that already exist on
# /work (10 sims x snaps 024/032/044/080, generated 2026-06-15).
# No generate step: that is the expensive half and it is already paid
# for. Minutes per task, not hours.
#
# Picks up, in one pass:
#   - the corrected CDDF (branch cddf-check: cells, sum, X(z), per-dN)
#   - flux_pdf.csv / tau_pdf.csv, which cannot be reconstructed later
#     because the spectra get deleted
#   - mean-flux rescaling factors + rescaled P_F(k)
#   - error bars: mean_flux_err, beta_fit_err, gamma_err, f_N_HI_err
#
# NOT included: the total-hydrogen line needs a regenerate, and the new
# low-z snapshots need a download first -- see run_lowz_set.sbatch.
#
# PREREQUISITE (once, on the login node, BEFORE submitting):
#   module load cmake gcc/13.2.0 impi/19.0.9 eigen/3.4.0 fftw3/3.3.10
#   ./compile.sh
# beta_fit_err / gamma_err / T0_err are new C++ fields; without the
# rebuild they silently come out NaN.
#
# Task map: task = (snap_index * 10) + sim_index + 1
#   sims  1-10 : shell_scripts/simulations_p1p2.txt
#   snaps      : 024 032 044 080  (z = 4.00, 3.00, 2.00, 0.27)
# ====================================================================

N_SIGHTLINES=10000

SNAPS=(024 032 044 080)

IDX=$((SLURM_ARRAY_TASK_ID - 1))
SIM_IDX=$((IDX % 10))
SNAP_IDX=$((IDX / 10))

SNAP_NUM=${SNAPS[$SNAP_IDX]}
SIMULATION=$(sed -n "$((SIM_IDX + 1))p" shell_scripts/simulations_p1p2.txt)

if [ -z "$SIMULATION" ] || [ -z "$SNAP_NUM" ]; then
    echo "ERROR: bad task mapping (task=$SLURM_ARRAY_TASK_ID sim=$SIM_IDX snap=$SNAP_IDX)"
    exit 1
fi

source shell_scripts/timing.sh
timing_context "$SIMULATION" "$SNAP_NUM"

echo "========================================"
echo "Task $SLURM_ARRAY_TASK_ID: $SIMULATION snap_$SNAP_NUM on $(hostname)"
echo "Started: $(date)"
echo "========================================"

module load cmake
module load gcc/13.2.0
module load impi/19.0.9
module load eigen/3.4.0
module load fftw3/3.3.10

source .venv/bin/activate || { echo "ERROR: venv activation failed"; exit 1; }

# Resolve by glob rather than by a fixed name: files generated with the
# total-H line are called camel_lya_lya_h_spectra_snap_*, and hardcoding
# camel_lya_spectra_snap_* would silently miss them.
SPECTRA_DIR="spectra/IllustrisTNG/1P/${SIMULATION}"
SPECTRA_PATH=$(ls -1 "${SPECTRA_DIR}"/camel_*_spectra_snap_${SNAP_NUM}_n${N_SIGHTLINES}.hdf5 2>/dev/null | head -1)

if [ -z "$SPECTRA_PATH" ] || [ ! -f "$SPECTRA_PATH" ]; then
    echo "ERROR: no spectra found in ${SPECTRA_DIR} for snap_${SNAP_NUM}"
    echo "This script is analyze-only by design. Use run_lowz_set.sbatch to generate."
    exit 1
fi

echo "analyze: $SPECTRA_PATH"
step_start analyze
OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-128} python analyze_spectra.py analyze "$SPECTRA_PATH" --workers 1
rc=$?
step_end $rc
if [ $rc -ne 0 ]; then
    echo "ERROR: analyze failed ($SIMULATION snap_$SNAP_NUM)"
    timing_total $rc
    exit 1
fi

rm -f "$SPECTRA_PATH"

timing_total 0

echo "========================================"
echo "$SIMULATION snap_$SNAP_NUM done: $(date)"
echo "========================================"
exit 0
