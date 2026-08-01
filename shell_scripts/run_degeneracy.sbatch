#!/bin/bash
#SBATCH -J degen_p1p2              # Job name
#SBATCH -o logs/degen_%A_%a.out    # Output file (%A=job ID, %a=array task ID)
#SBATCH -e logs/degen_%A_%a.err    # Error file
#SBATCH -p normal                  # Partition/queue (normal for Lonestar6)
#SBATCH -N 1                       # Number of nodes
#SBATCH -n 48                      # Number of tasks (cores) - for MPI or workers
#SBATCH -t 48:00:00                # Time limit (48 hours for high-z)
# NOTE: no --mem on Lonestar6 -- normal queue is exclusive/whole-node (256GB auto).
#       Specifying --mem fails with "Requested node configuration is not available".
#SBATCH --array=1-10               # Array job with 10 tasks (one per simulation)
#SBATCH --mail-type=END,FAIL       # Email notifications (optional)
#SBATCH --mail-user=turja.roy@uta.edu  # Email address (update if needed)

# ====================================================================
# SLURM Array Job: Omega0--sigma8 degeneracy (p1 + p2), one snapshot.
# ====================================================================
# Processes the 10 p1/p2 simulations in parallel as array tasks for a
# SINGLE snapshot (set SNAP_NUM below), generate + analyze.
#
# To cover all 6 redshifts, resubmit once per snapshot, changing SNAP_NUM:
#   014 (z6)  018 (z5)  024 (z4)  032 (z3)  044 (z2)  080 (z0.27)
# ====================================================================

# Configuration
N_SIGHTLINES=10000      # Number of sightlines to generate

# Snapshot configuration  -- CHANGE THIS and resubmit for each redshift
SNAP_NUM=014
SIGHTLINES_FILE="output/sightlines/analysis_sightlines_2025-01-28.hdf5"  # Set to "" to generate random sightlines

# Print job info
echo "========================================"
echo "SLURM Job Information"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Cores: $SLURM_NTASKS"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Date: $(date)"
echo "Working directory: $(pwd)"
echo ""
echo "Threads (OpenMP, generate + analyze): ${SLURM_CPUS_ON_NODE:-128}"
echo "Snapshot: snap_$SNAP_NUM"
echo "Sightlines: ${SIGHTLINES_FILE:-random}"
echo ""

# Load modules
module load cmake
module load gcc/13.2.0
module load impi/19.0.9
module load eigen/3.4.0
module load fftw3/3.3.10

# Get simulation name from simulations_p1p2.txt based on array task ID
SIMULATION=$(sed -n "${SLURM_ARRAY_TASK_ID}p" shell_scripts/simulations_p1p2.txt)

if [ -z "$SIMULATION" ]; then
    echo "ERROR: Could not read simulation name for array task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "Processing simulation: $SIMULATION"
echo "========================================"
echo ""

# Define paths
SNAPSHOT_PATH="data/IllustrisTNG/1P/${SIMULATION}/snap_${SNAP_NUM}.hdf5"

# Handle sightlines file
if [ -n "$SIGHTLINES_FILE" ] && [ -f "$SIGHTLINES_FILE" ]; then
    SIGHTLINES_ARG="--sightlines-from $SIGHTLINES_FILE"
else
    SIGHTLINES_ARG=""
    if [ -n "$SIGHTLINES_FILE" ]; then
        echo "Warning: Sightlines file not found: $SIGHTLINES_FILE"
        echo "Generating random sightlines instead"
    fi
fi

SPECTRA_PATH="spectra/IllustrisTNG/1P/${SIMULATION}/camel_lya_spectra_snap_${SNAP_NUM}_n${N_SIGHTLINES}.hdf5"

# Check if snapshot exists
if [ ! -f "$SNAPSHOT_PATH" ]; then
    echo "ERROR: Snapshot file not found: $SNAPSHOT_PATH"
    exit 1
fi

# Activate virtual environment
echo "[1/3] Activating virtual environment..."
source .venv/bin/activate

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment"
    exit 1
fi

echo "Virtual environment activated"
echo ""

source shell_scripts/timing.sh
timing_context "$SIMULATION" "$SNAP_NUM"

# Step 1: Generate spectra
# Single process; fake_spectra's C kernel threads across the node via OpenMP.
# OMP_NUM_THREADS is scoped to this command so it does not oversubscribe the
# multiprocessing pool in the analyze step below.
echo "[2/3] Generating spectra (OpenMP, ${SLURM_CPUS_ON_NODE:-128} threads)..."

step_start generate
OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-128} python analyze_spectra.py generate "$SNAPSHOT_PATH" \
    $SIGHTLINES_ARG \
    -n $N_SIGHTLINES
rc=$?
step_end $rc

if [ $rc -ne 0 ]; then
    echo ""
    echo "ERROR: Spectra generation failed for $SIMULATION"
    timing_total $rc
    exit 1
fi

echo ""
echo "Spectra generation completed successfully"
echo ""

# Step 2: Analyze spectra.
# Single process (--workers 1) so each stage threads across the whole node:
# scipy FFT (power spectrum) + OpenMP C++ kernels (cddf/lwd/tdens). Using the
# multiprocessing pool here instead would oversubscribe cores against OMP.
echo "[3/3] Analyzing spectra (OpenMP, ${SLURM_CPUS_ON_NODE:-128} threads)..."
echo ""

step_start analyze
OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-128} python analyze_spectra.py analyze "$SPECTRA_PATH" --workers 1
rc=$?
step_end $rc

if [ $rc -ne 0 ]; then
    echo ""
    echo "ERROR: Spectra analysis failed for $SIMULATION"
    timing_total $rc
    exit 1
fi

# Save spectra structure/metadata (light) so it survives spectra/ cleanup.
EXPLORE_OUT="output/explore/IllustrisTNG/1P/${SIMULATION}/snap_${SNAP_NUM}.txt"
mkdir -p "$(dirname "$EXPLORE_OUT")"
step_start explore
python analyze_spectra.py explore "$SPECTRA_PATH" > "$EXPLORE_OUT" 2>&1
rc=$?
step_end $rc
[ $rc -eq 0 ] && echo "Saved spectra metadata to $EXPLORE_OUT" \
              || echo "Warning: explore failed (see $EXPLORE_OUT)"

timing_total 0
echo ""
echo "========================================"
echo "Simulation $SIMULATION (snap_$SNAP_NUM) completed successfully!"
echo "Finished at: $(date)"
echo "========================================"

exit 0
