#!/bin/bash

snaps=(24 28 32 38 44 50 60 72 80 90)
sims=(0 1 2 3)

for sim in "${sims[@]}"; do
    tmux new-window -n "EX_$sim"
    for s in "${snaps[@]}"; do
        tmux split-window -h "source .venv/bin/activate && python downloader.py --set EX --sim $sim --snapshot $s; read -p \"Press Enter to continue...\""
        tmux select-layout tiled
    done
done
