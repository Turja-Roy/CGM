#!/bin/bash
# Regenerate every cross-simulation figure from the per-snapshot CSVs.
# No spectra needed. Run from the repo root, AFTER syncing /work down.
#
#   bash shell_scripts/make_comparison_plots.sh
#
# Wipes plots/hypothesis_p1_test and plots/degeneracy_test first: both held
# pre-fix snap-014/018 output, and hypothesis moved to a <scan>/<snap>/ layout,
# so stale files would sit next to the new ones looking current. Everything in
# those two dirs is regenerated below.
set -euo pipefail

ROOT=output/analysis/IllustrisTNG/1P
COSMO=data/IllustrisTNG/1P/CosmoAstroSeed_IllustrisTNG_L25n256_1P.csv
SNAPS=snap-024,snap-028,snap-032,snap-038,snap-044,snap-050,snap-060,snap-072,snap-080,snap-090
PY=${PY:-.venv/bin/python}

# --- gate: refuse to plot a half-synced grid -------------------------------
# A missing variant silently changes which two points the delta-tau_eff figure
# treats as the ends of the scan, so the lever arm stops being the same at
# every redshift. Fail loudly instead.
missing=0
for sim in 1P_p1_n2 1P_p1_n1 1P_p1_0 1P_p1_1 1P_p1_2 \
           1P_p2_n2 1P_p2_n1 1P_p2_0 1P_p2_1 1P_p2_2; do
  for snap in ${SNAPS//,/ }; do
    f="$ROOT/$sim/$snap/flux_stats.csv"
    [ -f "$f" ] || { echo "MISSING: $f"; missing=$((missing + 1)); }
  done
done
if [ "$missing" -ne 0 ]; then
  echo
  echo "$missing of 100 snapshot dirs incomplete -- cluster jobs not finished,"
  echo "or /work not synced down. Fix that first."
  exit 1
fi
echo "grid complete: 100/100"

rm -rf plots/hypothesis_p1_test plots/degeneracy_test

echo; echo "=== [1/4] flux / tau PDF redshift overlays ==="
$PY scripts/pdf_evolution.py --analysis-root "$ROOT" --snaps "$SNAPS" \
    --scans p1,p2 --out-dir plots/pdf_evolution

echo; echo "=== [2/4] hypothesis tests (p1 + p2) ==="
$PY scripts/hypothesis_test_p1.py --analysis-root "$ROOT" --cosmo-csv "$COSMO" \
    --snaps "$SNAPS" --scans p1,p2 --out-dir plots/hypothesis_p1_test

echo; echo "=== [3/4] Omega_0-sigma_8 degeneracy ==="
$PY scripts/degeneracy_test.py --analysis-root "$ROOT" --cosmo-csv "$COSMO" \
    --snaps "$SNAPS" --out-dir plots/degeneracy_test

echo; echo "=== [4/4] multi-line plots (metal-line colden fix) ==="
$PY scripts/replot.py --root "$ROOT" --out-dir plots/IllustrisTNG/1P \
    --pattern '1P_p[12]_*' --only multi_line_comparison --workers 8

echo; echo "Done. The geometry decomposition is the thing to read first:"
for s in p1 p2; do
  echo "  plots/hypothesis_p1_test/$s/across_snapshots/delta_tau_eff_vs_z.json"
done
