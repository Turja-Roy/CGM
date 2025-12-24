#!/bin/bash
#SBATCH --job-name=camel_generate
#SBATCH --output=logs/generate_%A_%a.out
#SBATCH --error=logs/generate_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --array=0-9%5           # Process 10 snapshots, max 5 at a time

# CAMEL Spectra Generation - SLURM Array Job Template
# 
# This script generates Lyman-alpha spectra for multiple snapshots in parallel.
# Each array task processes one snapshot file.
#
# Usage:
#   1. Edit SNAPSHOT_DIR and other parameters below
#   2. Submit with: sbatch slurm_templates/generate_spectra.sbatch
#   3. Monitor with: squeue -u $USER

# ============================================================================
# CONFIGURATION
# ============================================================================

# Directory containing snapshot files
SNAPSHOT_DIR="/path/to/IllustrisTNG/LH/LH_0"

# Spectra generation parameters
NUM_SIGHTLINES=10000
SPECTRAL_LINES="lya"           # Can be comma-separated: lya,lyb,civ
PIXEL_RESOLUTION=0.1           # km/s per pixel
RANDOM_SEED=42

# Output directory (optional - leave empty to save in snapshot directory)
OUTPUT_DIR=""

# Python environment
CONDA_ENV="spectra_analysis"   # Or use venv: source /path/to/venv/bin/activate

# ============================================================================
# SETUP
# ============================================================================

# Load modules (adjust for your HPC system)
# module load python/3.11
# module load hdf5/1.12

# Activate conda environment
# conda activate $CONDA_ENV

# Or activate virtual environment
cd $SLURM_SUBMIT_DIR
source venv/bin/activate

# Create logs directory
mkdir -p logs

# Find all snapshots and select one for this array task
SNAPSHOTS=($SNAPSHOT_DIR/snap_*.hdf5)
SNAPSHOT_FILE="${SNAPSHOTS[$SLURM_ARRAY_TASK_ID]}"

# Check if snapshot exists
if [ ! -f "$SNAPSHOT_FILE" ]; then
    echo "Error: Snapshot not found: $SNAPSHOT_FILE"
    exit 1
fi

# ============================================================================
# EXECUTION
# ============================================================================

echo "========================================================================"
echo "CAMEL SPECTRA GENERATION"
echo "========================================================================"
echo "Job ID:         $SLURM_JOB_ID"
echo "Array Task ID:  $SLURM_ARRAY_TASK_ID"
echo "Node:           $SLURMD_NODENAME"
echo "CPUs:           $SLURM_CPUS_PER_TASK"
echo "Memory:         $SLURM_MEM_PER_NODE MB"
echo "Snapshot:       $SNAPSHOT_FILE"
echo "Sightlines:     $NUM_SIGHTLINES"
echo "Lines:          $SPECTRAL_LINES"
echo "========================================================================"
echo ""

# Run generation
if [ -z "$OUTPUT_DIR" ]; then
    python3 analyze_spectra.py generate \
        "$SNAPSHOT_FILE" \
        -n $NUM_SIGHTLINES \
        --line $SPECTRAL_LINES \
        --res $PIXEL_RESOLUTION \
        --seed $RANDOM_SEED
else
    python3 analyze_spectra.py generate \
        "$SNAPSHOT_FILE" \
        -n $NUM_SIGHTLINES \
        --line $SPECTRAL_LINES \
        --res $PIXEL_RESOLUTION \
        --seed $RANDOM_SEED \
        -o "$OUTPUT_DIR/$(basename $SNAPSHOT_FILE .hdf5)_spectra.hdf5"
fi

EXIT_CODE=$?

echo ""
echo "========================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Spectra generation completed"
else
    echo "FAILED: Spectra generation failed with exit code $EXIT_CODE"
fi
echo "========================================================================"

exit $EXIT_CODE
