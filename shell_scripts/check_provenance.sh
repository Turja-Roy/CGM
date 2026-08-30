#!/bin/bash
# Fails if any analysis CSV was written by pre-correction code.
#
# Markers: the corrected CDDF (30f3095, 2026-07-31) writes "# absorber_mode"
# and "# X ="; the old one wrote "# dX = ... Mpc". The t_eff fix (ac902ce,
# 2026-07-24) is what put tau_eff_err into flux_stats.csv.
#
# Also diffs 1P_p7_0 against 1P_p1_0 at snap-080: same CAMELS fiducial
# simulation, so their CSVs must agree once both come from current code.

cd "$(dirname "$0")/.." || exit 2
root=output/analysis
rc=0

check() {  # check <description> <files...>
    local what=$1; shift
    if [ "$#" -gt 0 ]; then
        rc=1
        echo "FAIL: $what ($# file(s))"
        printf '  %s\n' "$@" | head -5
        [ "$#" -gt 5 ] && echo "  ... and $(($# - 5)) more"
    else
        echo "ok:   $what"
    fi
}

check "every cddf.csv has the corrected header" \
      $(grep -L 'absorber_mode' $(find $root -name cddf.csv))
check "no cddf.csv keeps the old '# dX =' header" \
      $(grep -l '^# dX = ' $(find $root -name cddf.csv))
check "every flux_stats.csv has tau_eff_err" \
      $(grep -L '^tau_eff_err,' $(find $root -name flux_stats.csv))

twin=$root/IllustrisTNG/1P
for f in cddf.csv flux_stats.csv; do
    a=$twin/1P_p7_0/snap-080/$f
    b=$twin/1P_p1_0/snap-080/$f
    if [ ! -f "$a" ] || [ ! -f "$b" ]; then
        echo "skip: fiducial twin $f (missing $a or $b)"
    elif diff -q "$a" "$b" > /dev/null; then
        echo "ok:   fiducial twin agrees on $f"
    else
        rc=1
        echo "FAIL: 1P_p7_0 and 1P_p1_0 are the same simulation but differ in $f"
        diff "$a" "$b" | head -6 | sed 's/^/  /'
    fi
done

exit $rc
