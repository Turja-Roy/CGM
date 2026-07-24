#!/bin/bash
#SBATCH -J ex_gen                  # Job name (submit_ex.sh overrides per EX)
#SBATCH -o logs/ex_gen_%A_%a.out   # Output file (%A=job ID, %a=array task ID)
#SBATCH -e logs/ex_gen_%A_%a.err   # Error file
#SBATCH -p normal                  # Partition/queue (normal for Lonestar6)
#SBATCH -N 1                       # Number of nodes
#SBATCH -n 48                      # Number of tasks (cores)
#SBATCH -t 24:00:00                # Time limit (48 hours for high-z)
# NOTE: no --mem on Lonestar6 -- normal queue is exclusive/whole-node (256GB auto).
#       Specifying --mem fails with "Requested node configuration is not available".
#SBATCH --array=1-6                # Array: one task per snapshot (6 redshifts)
#SBATCH --mail-type=END,FAIL       # Email notifications (optional)
#SBATCH --mail-user=turja.roy@uta.edu

# ====================================================================
# SLURM Array Job: EX (extreme feedback) set -- GENERATE spectra only.
# ====================================================================
# One EX sim (EX_${EX_NUM}) across all 6 forest-core snapshots, run as
# array tasks. EX_NUM is set by the caller (submit_ex.sh) via
# --export=ALL,EX_NUM=<n>; defaults to 0 if run standalone.
#
# Array task -> snapshot mapping:
#   1->014(z6) 2->018(z5) 3->024(z4) 4->032(z3) 5->044(z2) 6->080(z0.27)
# ====================================================================

# Configuration
N_SIGHTLINES=10000
EX_NUM="${EX_NUM:-0}"     # override with: sbatch --export=ALL,EX_NUM=1 ...
SIGHTLINES_FILE="output/sightlines/analysis_sightlines_2025-01-28.hdf5"

# Array task -> snapshot number
SNAPS=(014 018 024 032 044 080)
SNAP_NUM=${SNAPS[$((SLURM_ARRAY_TASK_ID - 1))]}

if [ -z "$SNAP_NUM" ]; then
    echo "ERROR: No snapshot for array task $SLURM_ARRAY_TASK_ID (expected 1-6)"
    exit 1
fi

SIMULATION="EX_${EX_NUM}"

echo "========================================"
echo "SLURM Job Information (GENERATE)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID   Array Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)   Cores: $SLURM_NTASKS"
echo "Simulation: $SIMULATION   Snapshot: snap_$SNAP_NUM"
echo "Sightlines: $SIGHTLINES_FILE"
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
SNAPSHOT_PATH="data/IllustrisTNG/EX/${SIMULATION}/snap_${SNAP_NUM}.hdf5"

if [ ! -f "$SNAPSHOT_PATH" ]; then
    echo "ERROR: Snapshot file not found: $SNAPSHOT_PATH"
    exit 1
fi

# Sightlines file (reused from all prior 1P analysis; same 25 Mpc/h box)
if [ -n "$SIGHTLINES_FILE" ] && [ -f "$SIGHTLINES_FILE" ]; then
    SIGHTLINES_ARG="--sightlines-from $SIGHTLINES_FILE"
else
    echo "ERROR: Sightlines file not found: $SIGHTLINES_FILE"
    exit 1
fi

# Activate virtual environment
echo "[1/2] Activating virtual environment..."
source .venv/bin/activate || { echo "ERROR: venv activation failed"; exit 1; }
echo "Virtual environment activated"
echo ""

# Generate spectra.
# Single process; fake_spectra's C kernel threads across the node via OpenMP.
echo "[2/2] Generating spectra (OpenMP, ${SLURM_CPUS_ON_NODE:-128} threads)..."
OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-128} python analyze_spectra.py generate "$SNAPSHOT_PATH" \
    $SIGHTLINES_ARG \
    -n $N_SIGHTLINES

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Spectra generation failed for $SIMULATION snap_$SNAP_NUM"
    exit 1
fi

echo ""
echo "========================================"
echo "GENERATE done: $SIMULATION snap_$SNAP_NUM  ($(date))"
echo "========================================"
exit 0
