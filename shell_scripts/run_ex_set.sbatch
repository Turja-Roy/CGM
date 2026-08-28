#!/bin/bash
#SBATCH -J ex_set                  # Job name
#SBATCH -o logs/ex_%A_%a.out       # Output file (%A=job ID, %a=array task ID)
#SBATCH -e logs/ex_%A_%a.err       # Error file
#SBATCH -p normal                  # Partition/queue (Lonestar6)
#SBATCH -N 1                       # One whole node per task
#SBATCH -n 48                      # Cores
#SBATCH -t 04:00:00                # Time limit
#SBATCH --array=1-40               # 4 EX sims x 10 snapshots
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=turja.roy@uta.edu

# ====================================================================
# GENERATE + ANALYZE the EX (extreme feedback) set.
#
# EX holds cosmology fixed (Omega_m = 0.3, sigma_8 = 0.8) and the IC seed
# shared (13560), varying only feedback:
#   EX_0  fiducial
#   EX_1  A_AGN1 = 100   (extreme AGN)
#   EX_2  A_SN1  = 100   (extreme SN / stellar)
#   EX_3  all A  = 0     (no feedback)
# Because the seed is shared, EX-vs-EX differences are pure feedback with
# no cosmic variance. This is the feedback-robustness arm of the
# Omega_0-sigma_8 degeneracy study: a discriminant that moves a lot with
# cosmology but little across EX is robust; one that moves as much with
# feedback is contaminated.
#
# Snapshots are the production set decided 2026-08-01, redshifts MEASURED
# from group-catalog headers rather than interpolated:
#   024 -> 3.9961   038 -> 2.4647   060 -> 1.0476   080 -> 0.2696
#   028 -> 3.4962   044 -> 2.0020   072 -> 0.5373   090 -> 0 (2.2e-16)
#   032 -> 2.9986   050 -> 1.6012
# 014 (z=6) and 018 (z=5) are excluded: TNG's UVB only switches on at
# z < 6. Every snapshot here is z < 4.5, so no CDDF beta_fit should come
# back NaN/saturated -- if one does, the snapshot list is wrong.
#
# Sightlines are the shared master set, the same file every 1P run used.
# Same 25 Mpc/h box, so EX overlays 1P directly and inter-set comparisons
# are sample-variance free.
#
# --line lya only: no metal lines. OVI/CIV cannot be added by re-running
# analyze, only by re-running generate, so that is a deliberate scope cut.
#
# Spectra are staged to node-local /tmp (288GB SSD, off the /work quota),
# then symlinked into spectra/ so analyze resolves correct output paths
# (suite/set/sim/snap from the link path). Both files are cleaned up on
# success or failure. Existing analysis output dirs cause a full skip, so
# a partial campaign restarts by plain resubmission.
#
# PREREQUISITES (on the login node, BEFORE submitting):
#   1. bash shell_scripts/download_ex.sh        # 40 files, ~2.5 GB each
#   2. module load cmake gcc/13.2.0 impi/19.0.9 eigen/3.4.0 fftw3/3.3.10
#      ./compile.sh
#
# Task map: task = (snap_index * 4) + sim_index + 1
#   sims  1-4 : EX_0 EX_1 EX_2 EX_3
#   snaps     : 024 028 032 038 044 050 060 072 080 090
# Tasks 1-4 are therefore all four sims at snap 024, giving a complete
# cross-sim comparison at one redshift as soon as they land.
# ====================================================================

N_SIGHTLINES=10000
SIGHTLINES_FILE="output/sightlines/analysis_sightlines_2025-01-28.hdf5"
LINES="lya"

SNAPS=(024 028 032 038 044 050 060 072 080 090)
SIMS=(EX_0 EX_1 EX_2 EX_3)

IDX=$((SLURM_ARRAY_TASK_ID - 1))
SIM_IDX=$((IDX % 4))
SNAP_IDX=$((IDX / 4))

SNAP_NUM=${SNAPS[$SNAP_IDX]}
SIMULATION=${SIMS[$SIM_IDX]}

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

SNAPSHOT_PATH="data/IllustrisTNG/EX/${SIMULATION}/snap_${SNAP_NUM}.hdf5"
SPECTRA_PATH="/tmp/camel_lya_spectra_snap_${SNAP_NUM}_n${N_SIGHTLINES}.hdf5"
SPECTRA_LINK="spectra/IllustrisTNG/EX/${SIMULATION}/camel_lya_spectra_snap_${SNAP_NUM}_n${N_SIGHTLINES}.hdf5"

if [ ! -f "$SNAPSHOT_PATH" ]; then
    echo "ERROR: snapshot not found: $SNAPSHOT_PATH"
    echo "Run shell_scripts/download_ex.sh first."
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
ANALYSIS_OUT="output/analysis/IllustrisTNG/EX/${SIMULATION}/snap-${SNAP_NUM}"
if [ -d "$ANALYSIS_OUT" ]; then
    echo "[1/3] SKIPPED, already analyzed: $ANALYSIS_OUT"
    exit 0
fi

rm -f "$SPECTRA_PATH" "$SPECTRA_LINK"

echo "[1/3] generate: $SNAPSHOT_PATH  (lines: $LINES)"
step_start generate
OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-128} python analyze_spectra.py generate "$SNAPSHOT_PATH" \
    $SIGHTLINES_ARG \
    --line "$LINES" \
    -n $N_SIGHTLINES \
    --output "$SPECTRA_PATH"
rc=$?
step_end $rc
if [ $rc -ne 0 ]; then
    echo "ERROR: generate failed ($SIMULATION snap_$SNAP_NUM)"
    timing_total $rc
    rm -f "$SPECTRA_PATH" "$SPECTRA_LINK"
    exit 1
fi

# Stage link so analyze resolves suite/set/sim/snap from the spectra path.
mkdir -p "$(dirname "$SPECTRA_LINK")"
ln -sf "$SPECTRA_PATH" "$SPECTRA_LINK"

# Step 2: analyze (single process; scipy FFT + OpenMP C++ kernels)
echo "[2/3] analyze: $SPECTRA_LINK"
step_start analyze
OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-128} python analyze_spectra.py analyze "$SPECTRA_LINK" --workers 1
rc=$?
step_end $rc
if [ $rc -ne 0 ]; then
    echo "ERROR: analyze failed ($SIMULATION snap_$SNAP_NUM)"
    timing_total $rc
    rm -f "$SPECTRA_PATH" "$SPECTRA_LINK"
    exit 1
fi

# Step 3: save spectra structure/metadata (light) so it survives cleanup.
EXPLORE_OUT="output/explore/IllustrisTNG/EX/${SIMULATION}/snap_${SNAP_NUM}.txt"
mkdir -p "$(dirname "$EXPLORE_OUT")"
echo "[3/3] explore: $EXPLORE_OUT"
step_start explore
python analyze_spectra.py explore "$SPECTRA_LINK" > "$EXPLORE_OUT" 2>&1
rc=$?
step_end $rc
[ $rc -eq 0 ] || echo "Warning: explore failed (see $EXPLORE_OUT)"

rm -f "$SPECTRA_PATH" "$SPECTRA_LINK"

timing_total 0

echo "========================================"
echo "$SIMULATION snap_$SNAP_NUM done: $(date)"
echo "========================================"
exit 0
