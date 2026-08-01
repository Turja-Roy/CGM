#!/bin/bash
#SBATCH -J lowz_set                # Job name
#SBATCH -o logs/lowz_%A_%a.out     # Output file (%A=job ID, %a=array task ID)
#SBATCH -e logs/lowz_%A_%a.err     # Error file
#SBATCH -p normal                  # Partition/queue (Lonestar6)
#SBATCH -N 1                       # One whole node per task
#SBATCH -n 48                      # Cores
#SBATCH -t 48:00:00                # Time limit
#SBATCH --array=1-60               # 10 sims (p1/p2) x 6 new snapshots
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=turja.roy@uta.edu

# ====================================================================
# GENERATE + ANALYZE for the six snapshots added 2026-08-01, with the
# total-hydrogen line in the same pass.
#
# Redshifts measured from group-catalog headers, not interpolated:
#   028 -> 3.4962   038 -> 2.4647   050 -> 1.6012
#   060 -> 1.0476   072 -> 0.5373   090 -> 0.0
#
# Why these matter: at z = 2-4 roughly 90% of the tau_eff spread across
# the Omega_0 scan is the 1/H(z) geometric factor, which sigma_8 cannot
# touch. At z = 0.27 that scaling breaks down, so the sub-z=2 points are
# the only ones where tau_eff carries information beyond the expansion
# history. They are the most valuable snapshots for a degeneracy-breaking
# claim, not the least, despite their larger error bars.
#
# snap 090 is z = 0: the forest is nearly transparent there (tau_eff of
# order 0.01) and only a handful of absorbers survive per sightline, so
# expect wide error bars and few populated high-N CDDF bins. It anchors
# the z-evolution baseline and matches HST/COS territory.
#
# --line lya,lya_h computes BOTH the HI Lyman-alpha optical depth and the
# total-hydrogen (ion = -1, Gunn-Peterson) one in a single pass, sharing
# the snapshot load and the sightlines. Doing it as a second job would
# pay the generate cost twice. Output file is therefore named
# camel_lya_lya_h_spectra_snap_NNN_n10000.hdf5.
#
# PREREQUISITES (on the login node, BEFORE submitting):
#   1. bash shell_scripts/download_lowz_snapshots.sh      # 50 x ~2.5 GB
#   2. module load cmake gcc/13.2.0 impi/19.0.9 eigen/3.4.0 fftw3/3.3.10
#      ./compile.sh
#
# Task map: task = (snap_index * 10) + sim_index + 1
#   sims  1-10 : shell_scripts/simulations_p1p2.txt
#   snaps      : 028 038 050 060 072 090
# ====================================================================

N_SIGHTLINES=10000
SIGHTLINES_FILE="output/sightlines/analysis_sightlines_2025-01-28.hdf5"
LINES="lya,lya_h"

SNAPS=(028 038 050 060 072 090)

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
SPECTRA_DIR="spectra/IllustrisTNG/1P/${SIMULATION}"
SPECTRA_PATH="${SPECTRA_DIR}/camel_lya_lya_h_spectra_snap_${SNAP_NUM}_n${N_SIGHTLINES}.hdf5"

if [ ! -f "$SNAPSHOT_PATH" ]; then
    echo "ERROR: snapshot not found: $SNAPSHOT_PATH"
    echo "Run shell_scripts/download_lowz_snapshots.sh first."
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

# Step 1: generate spectra (single process; fake_spectra threads via OpenMP)
if [ -f "$SPECTRA_PATH" ]; then
    echo "[1/2] generate: SKIPPED, spectra file exists: $SPECTRA_PATH"
else
    echo "[1/2] generate: $SNAPSHOT_PATH  (lines: $LINES)"
    step_start generate
    OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-128} python analyze_spectra.py generate "$SNAPSHOT_PATH" \
        $SIGHTLINES_ARG \
        --line "$LINES" \
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
