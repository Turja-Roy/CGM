#!/bin/bash
#SBATCH -J matter_pk_ex            # Job name
#SBATCH -o logs/job_%A.out         # Output file
#SBATCH -e logs/job_%A.err         # Error file
#SBATCH -p normal                  # Partition/queue (normal for Lonestar6)
#SBATCH -N 1                       # Number of nodes
#SBATCH -n 12                      # Number of tasks (cores)
#SBATCH -t 6:00:00                 # Time limit
#SBATCH --mail-type=END,FAIL       # Email notifications
#SBATCH --mail-user=turja.roy@uta.edu

# Matter P(k) for the EX (extreme-feedback) set.
#
# Total-matter field: gas + DM + stars + BH. Feedback moves baryons, so this is
# where AGN suppression at k >~ 1 h/Mpc actually shows. The tilt slope reported
# per variant is the same statistic the 1P run uses to call a shape change
# "Omega_0-like" -- a nonzero EX tilt means that discriminant is
# feedback-contaminated.
#
# 4 sims x 10 snapshots. All four share Omega_m=0.3, sigma_8=0.8 and IC seed
# 13560, so every difference here is feedback.

# Load modules
module load cmake
module load gcc/13.2.0
module load impi/19.0.9
module load eigen/3.4.0
module load fftw3/3.3.10

source .venv/bin/activate

source shell_scripts/timing.sh
timing_context matter_pk_ex all

step_start matter_pk_ex
python scripts/matter_pk_test.py \
    --data-root data/IllustrisTNG/EX \
    --cosmo-csv data/IllustrisTNG/EX/CosmoAstroSeed_IllustrisTNG_L25n256_EX.txt \
    --scans ex \
    --snaps 024,028,032,038,044,050,060,072,080,090 \
    --ngrid 256 \
    --field total \
    --out-dir plots/ex_matter_pk_test
rc=$?
step_end $rc
timing_total $rc
exit $rc
