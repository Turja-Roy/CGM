#!/bin/bash
#SBATCH -J lyah_2snap              # Job name
#SBATCH -o logs/lyah_%A_%a.out     # Output file (%A=job ID, %a=array task ID)
#SBATCH -e logs/lyah_%A_%a.err     # Error file
#SBATCH -p normal                  # Partition/queue (Lonestar6)
#SBATCH -N 1                       # One whole node per task
#SBATCH -n 48                      # Cores
#SBATCH -t 04:00:00                # Time limit
#SBATCH --array=1-20               # 10 sims (p1/p2) x 2 snapshots
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=turja.roy@uta.edu

# ====================================================================
# REGENERATE snaps 044 and 080 with --line lya,lya_h.
#
# reanalyze_existing.sbatch ran analyze-only over lya-only spectra, so
# tau_eff_H_total came out blank for 024/032/044/080. Only 044 (z=2.0020)
# and 080 (z=0.2696) are refilled: total-H tau is the Gunn-Peterson one
# and snap 028 (z=3.4962) already returns inf (every exp(-tau_H) underflows,
# <F> = 0), so z=3 and z=4 are GP-saturated, not missing.
#
# DELETES the existing analysis dir rather than skipping it -- the point is
# to replace the lya-only results.
#
# PREREQUISITES (login node, before submitting):
#   1. git pull            # needs the plotting.py log10(0) -> -inf guard
#   2. module load cmake gcc/13.2.0 impi/19.0.9 eigen/3.4.0 fftw3/3.3.10
#      ./compile.sh
#   3. data/IllustrisTNG/1P/1P_*/snap_{044,080}.hdf5 present. These predate
#      download_lowz_snapshots.sh; re-fetch with
#      python downloader.py --suite IllustrisTNG --set 1P --sim p1_0 --snapshot 44
#
# Task map: task = (snap_index * 10) + sim_index + 1
#   sims  1-10 : shell_scripts/simulations_p1p2.txt
#   snaps      : 044 080
# ====================================================================

N_SIGHTLINES=10000
SIGHTLINES_FILE="output/sightlines/analysis_sightlines_2025-01-28.hdf5"
LINES="lya,lya_h"

SNAPS=(044 080)

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
SPECTRA_PATH="/tmp/camel_lya_lya_h_spectra_snap_${SNAP_NUM}_n${N_SIGHTLINES}.hdf5"
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

# Replace, do not skip. The existing dir holds the lya-only results this
# job exists to supersede.
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
