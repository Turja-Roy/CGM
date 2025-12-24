#!/bin/bash
#SBATCH --job-name=camel_analyze
#SBATCH --output=logs/analyze_%A_%a.out
#SBATCH --error=logs/analyze_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --array=0-9%5           # Process 10 spectra files, max 5 at a time

# CAMEL Spectra Analysis - SLURM Array Job Template
# 
# This script runs comprehensive analysis on generated spectra files.
# Each array task processes one spectra file.
#
# Usage:
#   1. Edit SPECTRA_DIR below
#   2. Submit with: sbatch slurm_templates/analyze_spectra.sbatch
#   3. Monitor with: squeue -u $USER

# ============================================================================
# CONFIGURATION
# ============================================================================

# Directory containing spectra files (can use wildcards)
SPECTRA_DIR="/path/to/IllustrisTNG/LH/LH_0"
SPECTRA_PATTERN="camel_lya_spectra_*.hdf5"

# Analysis parameters
SPECTRAL_LINE="lya"            # Which line to analyze (auto-detect if empty)

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

# Find all spectra files and select one for this array task
SPECTRA_FILES=($SPECTRA_DIR/$SPECTRA_PATTERN)
SPECTRA_FILE="${SPECTRA_FILES[$SLURM_ARRAY_TASK_ID]}"

# Check if file exists
if [ ! -f "$SPECTRA_FILE" ]; then
    echo "Error: Spectra file not found: $SPECTRA_FILE"
    exit 1
fi

# ============================================================================
# EXECUTION
# ============================================================================

echo "========================================================================"
echo "CAMEL SPECTRA ANALYSIS"
echo "========================================================================"
echo "Job ID:         $SLURM_JOB_ID"
echo "Array Task ID:  $SLURM_ARRAY_TASK_ID"
echo "Node:           $SLURMD_NODENAME"
echo "CPUs:           $SLURM_CPUS_PER_TASK"
echo "Memory:         $SLURM_MEM_PER_NODE MB"
echo "Spectra File:   $SPECTRA_FILE"
echo "========================================================================"
echo ""

# Run comprehensive analysis
if [ -z "$SPECTRAL_LINE" ]; then
    python3 analyze_spectra.py analyze "$SPECTRA_FILE"
else
    python3 analyze_spectra.py analyze "$SPECTRA_FILE" --line $SPECTRAL_LINE
fi

EXIT_CODE=$?

echo ""
echo "========================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Analysis completed"
else
    echo "FAILED: Analysis failed with exit code $EXIT_CODE"
fi
echo "========================================================================"

exit $EXIT_CODE
