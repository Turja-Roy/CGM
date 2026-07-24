#!/bin/bash
# ====================================================================
# Submit the full EX pipeline: for each EX sim (0,1,2,3), submit the
# GENERATE array, then the ANALYZE array chained per-snapshot via
# --dependency=aftercorr, i.e. analyze array task i starts as soon as
# generate array task i (same snapshot) finishes successfully -- not
# waiting on the whole generate array. Requires matching array indices
# (both are --array=1-6).
#
# The four EX chains are independent of each other, so all four
# generate arrays queue at once; each sim's analyze tracks its own
# generate task-for-task.
#
# Run from repo root:  bash shell_scripts/submit_ex.sh
# ====================================================================
set -u

GEN_SBATCH="shell_scripts/ex_generate.sbatch"
ANA_SBATCH="shell_scripts/ex_analyze.sbatch"
EX_SIMS=(0 1 2 3)

for EX in "${EX_SIMS[@]}"; do
    echo ">>> EX_${EX}: submitting generate..."
    # NOTE: TACC's sbatch wrapper prints a welcome banner to stdout even with
    # --parsable, so grab only the numeric job id (last pure-number line).
    gen_jid=$(sbatch --parsable \
        --job-name="ex${EX}_gen" \
        --export=ALL,EX_NUM=${EX} \
        "$GEN_SBATCH" | grep -E '^[0-9]+$' | tail -1)

    if [ -z "$gen_jid" ]; then
        echo "    ERROR: generate submission failed for EX_${EX}; skipping analyze."
        continue
    fi
    echo "    generate array job: $gen_jid"

    echo ">>> EX_${EX}: submitting analyze (aftercorr:$gen_jid)..."
    ana_jid=$(sbatch --parsable \
        --job-name="ex${EX}_ana" \
        --dependency=aftercorr:${gen_jid} \
        --export=ALL,EX_NUM=${EX} \
        "$ANA_SBATCH" | grep -E '^[0-9]+$' | tail -1)
    echo "    analyze array job:  $ana_jid"
    echo ""
done
