#!/bin/bash
# ====================================================================
# Download the 2026-08-01 additions to the snapshot set.
#
# Redshifts MEASURED from the group-catalog Header/Redshift on
# 2026-08-01, not interpolated:
#   028 -> z = 3.4962      (the "z = 3.5" point)
#   038 -> z = 2.4647      (the "z = 2.5" point)
#   050 -> z = 1.6012
#   060 -> z = 1.0476      (snap 061 does NOT exist)
#   072 -> z = 0.5373      (snap 073 does NOT exist)
#   090 -> z = 0           (exactly; header reads 2.2e-16)
#
# CAMELS IllustrisTNG L25n256 only has snapshots 014, 018, 024, 028, 032
# and then EVEN numbers 034-090. Do not invent numbers between them.
#
# Existing set (already downloaded): 024 (z=4.00), 032 (z=3.00),
# 044 (z=2.00), 080 (z=0.27). 014 (z=6) and 018 (z=5) are dropped from
# the analysis -- TNG's UVB only switches on at z < 6.
#
# Run from repo root: bash shell_scripts/download_lowz_snapshots.sh
# ====================================================================
set -u

NEW_SNAPS=(28 38 50 60 72 90)
SIMS=(p1_n2 p1_n1 p1_0 p1_1 p1_2 p2_n2 p2_n1 p2_0 p2_1 p2_2)

for sim in "${SIMS[@]}"; do
  for s in "${NEW_SNAPS[@]}"; do
    dest=$(printf "data/IllustrisTNG/1P/1P_%s/snap_%03d.hdf5" "$sim" "$s")
    if [ -f "$dest" ]; then
      echo "SKIP (exists): $dest"
      continue
    fi
    echo ">>> 1P_${sim} snap ${s}"
    python downloader.py --suite IllustrisTNG --set 1P --sim "$sim" --snapshot "$s"
  done
done

echo "Done. Verify: expect 60 files"
find data/IllustrisTNG/1P \
  \( -name 'snap_028.hdf5' -o -name 'snap_038.hdf5' -o -name 'snap_050.hdf5' \
     -o -name 'snap_060.hdf5' -o -name 'snap_072.hdf5' -o -name 'snap_090.hdf5' \) | wc -l
