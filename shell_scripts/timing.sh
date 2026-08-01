#!/bin/bash
# ====================================================================
# Step timers for the sbatch scripts. Source from the repo root:
#     source shell_scripts/timing.sh
#     timing_context "$SIMULATION" "$SNAP_NUM"
#
#     step_start generate
#     <command>
#     rc=$?
#     step_end $rc
#
# Each step emits one grep-friendly line:
#   TIMING job=3317330 task=7 sim=1P_p1_0 snap=044 step=generate status=0 \
#          seconds=7321 elapsed=02:02:01
#
# Collect across a run:
#   grep -h '^TIMING' logs/*.out > timings.txt
#
# Mean and worst case per step, for sizing the next -t request:
#   grep -h '^TIMING' logs/*.out | sed 's/.*step=\([^ ]*\).*seconds=\([0-9]*\).*/\1 \2/' \
#     | awk '{n[$1]++; s[$1]+=$2; if ($2>m[$1]) m[$1]=$2}
#            END {for (k in n) printf "%-10s n=%-4d mean=%6.0fs  max=%6.0fs\n", k, n[k], s[k]/n[k], m[k]}'
# ====================================================================

_TIMING_JOB_T0=$(date +%s)
TIMING_SIM="?"
TIMING_SNAP="?"

timing_context() {
    TIMING_SIM="${1:-?}"
    TIMING_SNAP="${2:-?}"
}

timing_fmt_hms() {
    local s=$1
    printf '%02d:%02d:%02d' $((s / 3600)) $((s % 3600 / 60)) $((s % 60))
}

timing_emit() {
    local step=$1 status=$2 seconds=$3
    printf 'TIMING job=%s task=%s sim=%s snap=%s step=%s status=%s seconds=%d elapsed=%s\n' \
        "${SLURM_JOB_ID:-none}" "${SLURM_ARRAY_TASK_ID:-0}" \
        "$TIMING_SIM" "$TIMING_SNAP" "$step" "$status" \
        "$seconds" "$(timing_fmt_hms "$seconds")"
}

step_start() {
    _STEP_NAME="$1"
    _STEP_T0=$(date +%s)
    echo "[$_STEP_NAME] started $(date '+%F %T')"
}

# Pass the command's exit status so a failed step is still timed and is
# marked status!=0, otherwise the average is computed over runs that died early.
step_end() {
    local status=${1:-0}
    local dt=$(( $(date +%s) - _STEP_T0 ))
    echo "[$_STEP_NAME] finished in $(timing_fmt_hms "$dt")  (status $status)"
    timing_emit "$_STEP_NAME" "$status" "$dt"
}

timing_total() {
    local status=${1:-0}
    local dt=$(( $(date +%s) - _TIMING_JOB_T0 ))
    echo "[total] $(timing_fmt_hms "$dt")"
    timing_emit total "$status" "$dt"
}
