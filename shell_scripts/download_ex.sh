#!/bin/bash
# ====================================================================
# Download forest-core snapshots for the EX (extreme feedback) set.
# One tmux window per EX sim (EX_0..EX_3); one pane per snapshot.
# Snaps: 014 018 024 032 044 080  (z = 6, 5, 4, 3, 2, 0.27).
# downloader.py takes int snapshot, writes snap_0NN.hdf5.
# Run from repo root inside a tmux session: bash shell_scripts/download_ex.sh
# ====================================================================

snaps=(14 18 24 32 44 80)
sims=(0 1 2 3)

for sim in "${sims[@]}"; do
    tmux new-window -n "EX_$sim"
    for s in "${snaps[@]}"; do
        tmux split-window -h "source .venv/bin/activate && python downloader.py --set EX --sim $sim --snapshot $s; read -p \"Press Enter to continue...\""
        tmux select-layout tiled
    done
done
