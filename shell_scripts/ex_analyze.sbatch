#!/bin/bash
#SBATCH -J ex_ana                  # Job name (submit_ex.sh overrides per EX)
#SBATCH -o logs/ex_ana_%A_%a.out   # Output file (%A=job ID, %a=array task ID)
#SBATCH -e logs/ex_ana_%A_%a.err   # Error file
#SBATCH -p normal                  # Partition/queue (normal for Lonestar6)
#SBATCH -N 1                       # Number of nodes
#SBATCH -n 48                      # Number of tasks (cores)
#SBATCH -t 48:00:00                # Time limit
# NOTE: no --mem on Lonestar6 -- normal queue is exclusive/whole-node (256GB auto).
#SBATCH --array=1-6                # Array: one task per snapshot (6 redshifts)
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=turja.roy@uta.edu

# ====================================================================
# SLURM Array Job: EX (extreme feedback) set -- ANALYZE spectra only.
# ====================================================================
# Chained per-snapshot to ex_generate.sbatch via submit_ex.sh
# (--dependency=aftercorr): analyze task i runs once generate task i for
# the same snapshot finishes. Consumes the spectra produced by generate.
# EX_NUM set by caller via --export=ALL,EX_NUM=<n>; defaults to 0.
# ====================================================================

# Configuration
N_SIGHTLINES=10000
EX_NUM="${EX_NUM:-0}"

# Array task -> snapshot number
SNAPS=(014 018 024 032 044 080)
SNAP_NUM=${SNAPS[$((SLURM_ARRAY_TASK_ID - 1))]}

if [ -z "$SNAP_NUM" ]; then
    echo "ERROR: No snapshot for array task $SLURM_ARRAY_TASK_ID (expected 1-6)"
    exit 1
fi

SIMULATION="EX_${EX_NUM}"

echo "========================================"
echo "SLURM Job Information (ANALYZE)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID   Array Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)   Cores: $SLURM_NTASKS"
echo "Simulation: $SIMULATION   Snapshot: snap_$SNAP_NUM"
echo "Threads (OpenMP): ${SLURM_CPUS_ON_NODE:-128}"
echo "Date: $(date)"
echo "========================================"
echo ""

# Load modules
module load cmake
module load gcc/13.2.0
module load impi/19.0.9
module load eigen/3.4.0
module load fftw3/3.3.10

# Paths
SPECTRA_PATH="spectra/IllustrisTNG/EX/${SIMULATION}/camel_lya_spectra_snap_${SNAP_NUM}_n${N_SIGHTLINES}.hdf5"

if [ ! -f "$SPECTRA_PATH" ]; then
    echo "ERROR: Spectra file not found (did generate succeed?): $SPECTRA_PATH"
    exit 1
fi

# Activate virtual environment
echo "[1/3] Activating virtual environment..."
source .venv/bin/activate || { echo "ERROR: venv activation failed"; exit 1; }
echo "Virtual environment activated"
echo ""

# Analyze spectra.
# Single process (--workers 1) so each stage threads across the whole node:
# scipy FFT (power spectrum) + OpenMP C++ kernels (cddf/lwd/tdens).
echo "[2/3] Analyzing spectra (OpenMP, ${SLURM_CPUS_ON_NODE:-128} threads)..."
OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-128} python analyze_spectra.py analyze "$SPECTRA_PATH" --workers 1

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Spectra analysis failed for $SIMULATION snap_$SNAP_NUM"
    exit 1
fi

# Save spectra structure/metadata (light) so it survives spectra/ cleanup.
EXPLORE_OUT="output/explore/IllustrisTNG/EX/${SIMULATION}/snap_${SNAP_NUM}.txt"
mkdir -p "$(dirname "$EXPLORE_OUT")"
echo "[3/3] Saving spectra metadata..."
python analyze_spectra.py explore "$SPECTRA_PATH" > "$EXPLORE_OUT" 2>&1 \
    && echo "Saved spectra metadata to $EXPLORE_OUT" \
    || echo "Warning: explore failed (see $EXPLORE_OUT)"

echo ""
echo "========================================"
echo "ANALYZE done: $SIMULATION snap_$SNAP_NUM  ($(date))"
echo "========================================"
exit 0
