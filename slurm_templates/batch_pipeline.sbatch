#!/bin/bash
#SBATCH --job-name=camel_pipeline
#SBATCH --output=logs/pipeline_%A_%a.out
#SBATCH --error=logs/pipeline_%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --array=0-9%3           # Process 10 snapshots, max 3 at a time

# CAMEL Full Pipeline - SLURM Array Job Template
# 
# This script runs the full pipeline (generate + analyze) for multiple snapshots.
# Each array task processes one snapshot from start to finish.
#
# Usage:
#   1. Edit SNAPSHOT_DIR and parameters below
#   2. Submit with: sbatch slurm_templates/batch_pipeline.sbatch
#   3. Monitor with: squeue -u $USER

# ============================================================================
# CONFIGURATION
# ============================================================================

# Directory containing snapshot files
SNAPSHOT_DIR="/path/to/IllustrisTNG/LH/LH_0"

# Generation parameters
NUM_SIGHTLINES=10000
SPECTRAL_LINES="lya"
PIXEL_RESOLUTION=0.1
RANDOM_SEED=42

# Python environment
CONDA_ENV="spectra_analysis"

# ============================================================================
# SETUP
# ============================================================================

# Load modules
# module load python/3.11
# module load hdf5/1.12

# Activate environment
cd $SLURM_SUBMIT_DIR
source venv/bin/activate

# Create logs directory
mkdir -p logs

# Find all snapshots and select one for this array task
SNAPSHOTS=($SNAPSHOT_DIR/snap_*.hdf5)
SNAPSHOT_FILE="${SNAPSHOTS[$SLURM_ARRAY_TASK_ID]}"

if [ ! -f "$SNAPSHOT_FILE" ]; then
    echo "Error: Snapshot not found: $SNAPSHOT_FILE"
    exit 1
fi

# ============================================================================
# EXECUTION
# ============================================================================

echo "========================================================================"
echo "CAMEL FULL PIPELINE"
echo "========================================================================"
echo "Job ID:         $SLURM_JOB_ID"
echo "Array Task ID:  $SLURM_ARRAY_TASK_ID"
echo "Node:           $SLURMD_NODENAME"
echo "Snapshot:       $SNAPSHOT_FILE"
echo "========================================================================"
echo ""

# Run full pipeline (generate + analyze)
python3 analyze_spectra.py pipeline \
    "$SNAPSHOT_FILE" \
    -n $NUM_SIGHTLINES \
    --line $SPECTRAL_LINES \
    --res $PIXEL_RESOLUTION \
    --seed $RANDOM_SEED

EXIT_CODE=$?

echo ""
echo "========================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Full pipeline completed"
else
    echo "FAILED: Pipeline failed with exit code $EXIT_CODE"
fi
echo "========================================================================"

exit $EXIT_CODE
