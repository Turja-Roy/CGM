#!/bin/bash
#SBATCH -J p789_080                # Job name
#SBATCH -o logs/p789_%A_%a.out     # Output file (%A=job ID, %a=array task ID)
#SBATCH -e logs/p789_%A_%a.err     # Error file
#SBATCH -p normal                  # Partition/queue (Lonestar6)
#SBATCH -N 1                       # One whole node per task
#SBATCH -n 48                      # Cores
#SBATCH -t 04:00:00                # Time limit
#SBATCH --array=1-15               # 15 sims (p7/p8/p9) x 1 snapshot
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=turja.roy@uta.edu

# ====================================================================
# REGENERATE the p7/p8/p9 scans (Omega_b, h, n_s) at snap 080 (z=0.2696).
#
# Why: these are the last CSVs in output/analysis produced by pre-correction
# code. They were last written 2026-02-03, which is BEFORE
#   af48559 (2026-03-07)  C++ acceleration -- they came out of the pure-Python
#                         path, so not even the same kernels
#   ac902ce (2026-07-24)  float32 -> double accumulators; the old numbers carry
#                         a saturated mean_tau and a meaningless
#                         deep_absorption_frac, and no tau_eff row at all
#   30f3095 (2026-07-31)  CDDF correction: 50 km/s cells, sum, X(z), per-dex
#
# Proof they cannot be plotted next to the p1/p2 points: 1P_p7_0 and 1P_p1_0
# are the SAME CAMELS fiducial simulation, yet their snap-080 cddf.csv read
#   p1_0 (2026-08-23):  # X = 1.344141e-02  # absorber_mode = 2  n_absorbers = 450000
#   p7_0 (2026-02-03):  # dX = 37.252272 Mpc (comoving)          n_absorbers = 1720
# After this job they must agree to round-off. That diff is the acceptance test.
#
# snap 080 is the only snapshot p7/p8/p9 keep: 014 (z=6) and 018 (z=5) sit above
# TNG's UVB switch-on and were retired with this change.
#
# --line lya,lya_h matches what p1/p2 snap-080 used on 2026-08-23, so
# tau_eff_H_total is filled and the two scans stay directly comparable.
#
# DELETES the existing analysis dir rather than skipping it -- replacing the
# stale results is the whole point. The old CSVs stay recoverable from git
# (commit 03df99b), so nothing needs to be moved aside first.
#
# NO ./compile.sh NEEDED: src/cpp has been unchanged since 2026-08-01 and the
# cluster .so was rebuilt 2026-08-22. Run it only if git pull touches src/cpp.
#
# PREREQUISITES (login node, before submitting):
#   1. git pull
#   2. data/IllustrisTNG/1P/1P_p{7,8,9}_*/snap_080.hdf5 present (they are)
#
# Task map: task = sim_index + 1, sims from shell_scripts/simulations_p789.txt
# ====================================================================

N_SIGHTLINES=10000
SIGHTLINES_FILE="output/sightlines/analysis_sightlines_2025-01-28.hdf5"
LINES="lya,lya_h"

SNAP_NUM=080

SIMULATION=$(sed -n "${SLURM_ARRAY_TASK_ID}p" shell_scripts/simulations_p789.txt)

if [ -z "$SIMULATION" ]; then
    echo "ERROR: bad task mapping (task=$SLURM_ARRAY_TASK_ID)"
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
# Sim name is in the /tmp filename: every task here is the same snapshot number,
# so a shared name would collide if two array tasks ever land on one node.
SPECTRA_PATH="/tmp/camel_lya_lya_h_spectra_${SIMULATION}_snap_${SNAP_NUM}_n${N_SIGHTLINES}.hdf5"
SPECTRA_LINK="spectra/IllustrisTNG/1P/${SIMULATION}/camel_lya_lya_h_spectra_snap_${SNAP_NUM}_n${N_SIGHTLINES}.hdf5"
ANALYSIS_OUT="output/analysis/IllustrisTNG/1P/${SIMULATION}/snap-${SNAP_NUM}"

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

# Replace, do not skip. The existing dir holds the 2026-02 results this job
# exists to supersede.
rm -rf "$ANALYSIS_OUT"
rm -f "$SPECTRA_PATH" "$SPECTRA_LINK"

echo "[1/2] generate: $SNAPSHOT_PATH  (lines: $LINES)"
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

# Stage link so analyze resolves suite/sim/snap from the spectra path.
mkdir -p "$(dirname "$SPECTRA_LINK")"
ln -sf "$SPECTRA_PATH" "$SPECTRA_LINK"

echo "[2/2] analyze: $SPECTRA_LINK"
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

rm -f "$SPECTRA_PATH" "$SPECTRA_LINK"

timing_total 0

echo "========================================"
echo "$SIMULATION snap_$SNAP_NUM done: $(date)"
echo "========================================"
exit 0
