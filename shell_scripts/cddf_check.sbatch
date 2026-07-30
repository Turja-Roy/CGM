#!/bin/bash
#SBATCH -J cddf_check              # Job name
#SBATCH -o logs/cddf_check_%A_%a.out  # Output file (%A=job ID, %a=array task ID)
#SBATCH -e logs/cddf_check_%A_%a.err  # Error file
#SBATCH -p normal                  # Partition/queue (Lonestar6)
#SBATCH -N 1                       # One whole node per task
#SBATCH -n 48                      # Cores
#SBATCH -t 02:00:00                # Time limit (minutes of compute; the 4 GB read dominates)
# NOTE: no --mem on Lonestar6 -- normal queue is exclusive/whole-node (256GB auto).
#SBATCH --array=1-3                # One task per snapshot
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=turja.roy@uta.edu

# ====================================================================
# CDDF threshold and implementation comparison (branch cddf-check).
#
# Runs scripts/cddf_threshold_test.py, which computes ~17 CDDF variants
# from one spectra file: the tau threshold sweep, max vs sum, the two
# normalisations, and our C++ configured to be arithmetically identical
# to fake_spectra's column_density_function. See docs/cddf_check/README.md.
#
# This is read-only with respect to the production outputs: it writes only
# to output/cddf_test/ and plots/cddf_test/, and never touches
# output/analysis/.../cddf.csv.
#
# PREREQUISITE (once, on the login node, BEFORE submitting):
#   module load cmake gcc/13.2.0 impi/19.0.9 eigen/3.4.0 fftw3/3.3.10
#   ./compile.sh
# The CDDF mode flags are new C++; an un-rebuilt .so fails with a
# TypeError on the argument list. Do NOT compile in-job -- all three
# array tasks share the rebuilt .so.
#
# Snapshot choice: two low redshifts plus one saturated high redshift.
#   task 1 -> snap 044, z = 2.00   unsaturated forest; the only task that
#                                  can exercise gate G1 against the
#                                  committed cddf.csv at 10000 sightlines
#   task 2 -> snap 080, z = 0.27   sparse forest; threshold acts as a pure
#                                  completeness cut
#   task 3 -> snap 014, z = 5.99   saturated/blended; where the committed
#                                  beta_fit is negative and n_absorbers is
#                                  ~one per sightline
#
# Override the simulation with --export=ALL,SIMULATION=1P_p1_n2 etc.
# ====================================================================

N_SIGHTLINES=10000
SIM_SET="${SIM_SET:-1P}"
SIMULATION="${SIMULATION:-1P_p1_0}"

# Array task -> snapshot number
SNAPS=(044 080 014)
SNAP_NUM=${SNAPS[$((SLURM_ARRAY_TASK_ID - 1))]}

if [ -z "$SNAP_NUM" ]; then
    echo "ERROR: No snapshot for array task $SLURM_ARRAY_TASK_ID (expected 1-3)"
    exit 1
fi

SPECTRA_PATH="spectra/IllustrisTNG/${SIM_SET}/${SIMULATION}/camel_lya_spectra_snap_${SNAP_NUM}_n${N_SIGHTLINES}.hdf5"

echo "========================================"
echo "SLURM Job Information (CDDF CHECK)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID   Array Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)   Cores: $SLURM_NTASKS"
echo "Simulation: $SIMULATION   Snapshot: snap_$SNAP_NUM"
echo "Threads (OpenMP): ${SLURM_CPUS_ON_NODE:-128}"
echo "Spectra: $SPECTRA_PATH"
echo "Date: $(date)"
echo "========================================"
echo ""

# Load modules (same set compile.sh used, so the .so finds its runtime libs)
module load cmake
module load gcc/13.2.0
module load impi/19.0.9
module load eigen/3.4.0
module load fftw3/3.3.10

if [ ! -f "$SPECTRA_PATH" ]; then
    echo "ERROR: Spectra file not found: $SPECTRA_PATH"
    echo "The spectra HDF5 are intermediates and may have been cleaned up;"
    echo "regenerate this snapshot before running the CDDF check."
    exit 1
fi

echo "[1/2] Activating virtual environment..."
source .venv/bin/activate || { echo "ERROR: venv activation failed"; exit 1; }
echo "Virtual environment activated"
echo ""

# Single process so the C++ CDDF kernel threads across the whole node.
echo "[2/2] Running CDDF threshold and implementation test..."
OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-128} \
    python scripts/cddf_threshold_test.py --spectra "$SPECTRA_PATH"
STATUS=$?

echo ""
if [ $STATUS -eq 0 ]; then
    echo "All validation gates passed."
elif [ $STATUS -eq 2 ]; then
    # The script writes its CSVs and plots before reporting gate failures, so the
    # outputs exist but must not be believed. Grep the log for "[FAIL]"/"[ERROR]".
    echo "WARNING: a validation gate FAILED (exit 2). Outputs were still written to"
    echo "  output/cddf_test/ and plots/cddf_test/ but must not be trusted."
    echo "Search this log for '[FAIL]' and '[ERROR]' to see which gate and why."
else
    echo "ERROR: cddf_threshold_test.py failed with exit $STATUS"
fi

echo ""
echo "========================================"
echo "CDDF CHECK done: $SIMULATION snap_$SNAP_NUM  (exit $STATUS, $(date))"
echo "========================================"
exit $STATUS
