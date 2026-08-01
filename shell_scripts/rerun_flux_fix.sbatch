#!/bin/bash
#SBATCH -J flux_fix_rerun          # Job name
#SBATCH -o logs/rerun_%A_%a.out    # Output file (%A=job ID, %a=array task ID)
#SBATCH -e logs/rerun_%A_%a.err    # Error file
#SBATCH -p normal                  # Partition/queue (Lonestar6)
#SBATCH -N 1                       # One whole node per task
#SBATCH -n 48                      # Cores
#SBATCH -t 48:00:00                # Time limit
#SBATCH --array=1-60               # 10 sims (p1/p2) x 6 snapshots
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=turja.roy@uta.edu

# ====================================================================
# Full re-run of generate + analyze for the p1/p2 scans after the
# float32-saturation fix in src/cpp/analysis/flux_stats.cpp.
# Every mean_flux / tau_eff CSV produced before 2026-07-24 at z <~ 3
# is invalid; spectra were deleted, so both steps must be redone.
#
# PREREQUISITE (once, on the login node, BEFORE submitting):
#   module load cmake gcc/13.2.0 impi/19.0.9 eigen/3.4.0 fftw3/3.3.10
#   ./compile.sh
# The rebuilt .so is shared by all array tasks; do NOT compile in-job.
#
# Task map: task = (snap_index * 10) + sim_index + 1
#   sims  1-10 : shell_scripts/simulations_p1p2.txt
#   snaps      : 014 018 024 032 044 080
# ====================================================================

N_SIGHTLINES=10000
SIGHTLINES_FILE="output/sightlines/analysis_sightlines_2025-01-28.hdf5"

SNAPS=(014 018 024 032 044 080)

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

SNAPSHOT_PATH="data/IllustrisTNG/1P/${SIMULATION}/snap_${SNAP_NUM}.hdf5"
SPECTRA_PATH="spectra/IllustrisTNG/1P/${SIMULATION}/camel_lya_spectra_snap_${SNAP_NUM}_n${N_SIGHTLINES}.hdf5"

if [ ! -f "$SNAPSHOT_PATH" ]; then
    echo "ERROR: snapshot not found: $SNAPSHOT_PATH"
    exit 1
fi

if [ -n "$SIGHTLINES_FILE" ] && [ -f "$SIGHTLINES_FILE" ]; then
    SIGHTLINES_ARG="--sightlines-from $SIGHTLINES_FILE"
else
    echo "ERROR: shared sightline file missing: $SIGHTLINES_FILE"
    echo "Refusing to fall back to random sightlines: the shared master set is"
    echo "what makes inter-variant comparisons sample-variance free."
    exit 1
fi

# Step 1: generate spectra (single process; fake_spectra threads via OpenMP).
# Skipped when the spectra file already exists: the 2026-07-24 bug was in the
# analyze-side C++ statistics only, so surviving raw-tau spectra are valid.
if [ -f "$SPECTRA_PATH" ]; then
    echo "[1/2] generate: SKIPPED, spectra file exists: $SPECTRA_PATH"
else
    echo "[1/2] generate: $SNAPSHOT_PATH"
    step_start generate
    OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-128} python analyze_spectra.py generate "$SNAPSHOT_PATH" \
        $SIGHTLINES_ARG \
        -n $N_SIGHTLINES
    rc=$?
    step_end $rc
    if [ $rc -ne 0 ]; then
        echo "ERROR: generate failed ($SIMULATION snap_$SNAP_NUM)"
        timing_total $rc
        exit 1
    fi
fi

# Step 2: analyze (single process; scipy FFT + OpenMP C++ kernels)
echo "[2/2] analyze: $SPECTRA_PATH"
step_start analyze
OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-128} python analyze_spectra.py analyze "$SPECTRA_PATH" --workers 1
rc=$?
step_end $rc
if [ $rc -ne 0 ]; then
    echo "ERROR: analyze failed ($SIMULATION snap_$SNAP_NUM)"
    timing_total $rc
    exit 1
fi

timing_total 0

echo "========================================"
echo "$SIMULATION snap_$SNAP_NUM done: $(date)"
echo "========================================"
exit 0
