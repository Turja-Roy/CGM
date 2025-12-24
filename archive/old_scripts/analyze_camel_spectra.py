#!/usr/bin/env python3
"""
Complete pipeline: Generate and analyze Lyman-alpha spectra from CAMEL data

This script:
1. Generates random sightlines through CAMEL snapshot
2. Uses fake_spectra to compute optical depths
3. Produces comprehensive analysis plots

Usage:
    python analyze_camel_spectra.py <snapshot_file> [options]
    
Examples:
    python analyze_camel_spectra.py data/IllustrisTNG/LH/LH_0/snap_080.hdf5
    python analyze_camel_spectra.py data/IllustrisTNG/LH/LH_0/snap_080.hdf5 --sightlines 200
    python analyze_camel_spectra.py data/IllustrisTNG/LH/LH_0/snap_080.hdf5 -n 50 --res 0.5
"""

import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
import os
import argparse

# ============================================================================
# Parse command line arguments
# ============================================================================
parser = argparse.ArgumentParser(
    description='Generate and analyze Lyman-alpha spectra from CAMEL snapshots',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  python analyze_camel_spectra.py data/IllustrisTNG/LH/LH_0/snap_080.hdf5
  python analyze_camel_spectra.py snap_080.hdf5 --sightlines 200 --res 0.05
  python analyze_camel_spectra.py snap_080.hdf5 -n 50 -r 0.5 --seed 123

Output files:
  - plots/camel_sample_spectra_snap_XXX.png
  - plots/camel_flux_statistics_snap_XXX.png
  - plots/camel_power_spectrum_snap_XXX.png
  - plots/camel_transmission_stats_snap_XXX.png
  - camel_lya_spectra_snap_XXX.hdf5
    """
)

parser.add_argument('snapshot_file', 
                    help='Path to CAMEL snapshot HDF5 file (e.g., snap_080.hdf5)')
parser.add_argument('-n', '--sightlines', type=int, default=100,
                    help='Number of random sightlines to generate (default: 100)')
parser.add_argument('-r', '--res', type=float, default=0.1,
                    help='Velocity resolution in km/s per pixel (default: 0.1)')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility (default: 42)')
parser.add_argument('--no-reload', action='store_true',
                    help='Use cached spectra file if it exists (faster)')

args = parser.parse_args()

snapshot_file = args.snapshot_file
num_sightlines = args.sightlines
resolution = args.res
random_seed = args.seed

# Check if file exists
if not os.path.exists(snapshot_file):
    print(f"Error: File not found: {snapshot_file}")
    print("\nSearching for available snapshots...")
    if os.path.exists('data'):
        found = False
        for root, dirs, files in os.walk('data'):
            for file in files:
                if file.startswith('snap_') and file.endswith('.hdf5'):
                    filepath = os.path.join(root, file)
                    print(f"  Found: {filepath}")
                    found = True
        if not found:
            print("  No snapshots found in data/ directory")
    sys.exit(1)

# Extract snapshot number and directory from file path
snapshot_dir = os.path.dirname(snapshot_file)
snapshot_name = os.path.basename(snapshot_file).replace('.hdf5', '')
snapshot_num = int(snapshot_name.replace('snap_', ''))

print("="*70)
print("CAMEL Lyman-alpha Spectra Analysis Pipeline")
print("="*70)
print(f"\nSnapshot file: {snapshot_file}")
print(f"Snapshot number: {snapshot_num}")
print(f"Sightlines: {num_sightlines}")
print(f"Resolution: {resolution} km/s/pixel")
print(f"Random seed: {random_seed}")

# ============================================================================
# STEP 1: Load snapshot info to prepare sightlines
# ============================================================================
print("\n[Step 1] Examining snapshot...")

with h5py.File(snapshot_file, 'r') as f:
    header = f['Header']
    redshift = header.attrs['Redshift']
    boxsize = header.attrs['BoxSize']  # ckpc/h
    hubble = header.attrs['HubbleParam']
    n_gas = header.attrs['NumPart_ThisFile'][0]

print(f"  Snapshot: {snapshot_name}")
print(f"  Redshift: z = {redshift:.3f}")
print(f"  Box size: {boxsize:.1f} ckpc/h = {boxsize/hubble/1000:.2f} Mpc/h")
print(f"  Gas particles: {n_gas:,}")

# ============================================================================
# STEP 2: Generate random sightlines
# ============================================================================
print("\n[Step 2] Generating random sightlines...")

np.random.seed(random_seed)

# Random positions in box (in ckpc/h)
cofm = np.random.uniform(0, boxsize, size=(num_sightlines, 3))

# Random axes (1=x, 2=y, 3=z)
axis = np.random.randint(1, 4, size=num_sightlines)

print(f"  Generated {num_sightlines} random sightlines")
print(f"  Box coverage: uniform random sampling")

# ============================================================================
# STEP 3: Run fake_spectra
# ============================================================================
print("\n[Step 3] Running fake_spectra (this may take a few minutes)...")
print("  Computing Lyman-alpha optical depths along sightlines...")

from fake_spectra import spectra
from fake_spectra import abstractsnapshot

# ============================================================================
# BUGFIX: Monkey-patch to fix uint32 overflow in fake_spectra with Python 3.13
# ============================================================================
# The library has a bug where 2**32 * uint32 causes overflow
# Fix: ensure calculation uses int64
def get_npart_fixed(self):
    """Get the total number of particles (fixed for uint32 overflow)."""
    npart_total = self.get_header_attr("NumPart_Total").astype(np.int64)
    npart_high = self.get_header_attr("NumPart_Total_HighWord").astype(np.int64)
    return npart_total + (2**32) * npart_high

# Apply the patch to all snapshot classes
abstractsnapshot.AbstractSnapshotFactory.get_npart = get_npart_fixed
abstractsnapshot.HDF5Snapshot.get_npart = get_npart_fixed
abstractsnapshot.BigFileSnapshot.get_npart = get_npart_fixed

# ============================================================================
# BUGFIX 2: Fix float32/float64 type casting for _Particle_Interpolate
# ============================================================================
# The C extension (_Particle_Interpolate) has strict type requirements:
#   - All scalar parameters must be float32
#   - Data arrays (pos, vel, elem_den, temp, hh) must be float32
#   - Sightline positions (cofm) MUST be float64 (not float32!)
#   - Axis indices must be int32
# In Python 3.13, numpy defaults may have changed, causing type mismatches
from fake_spectra._spectra_priv import _Particle_Interpolate as _PI_original

def _do_interpolation_work_fixed(self, pos, vel, elem_den, temp, hh, amumass, line, get_tau):
    """Run the interpolation with proper float32 casting (fixed for Python 3.13)"""
    # Factor of 10^-8 converts line width (lambda_X) from Angstrom to cm
    if self.turn_off_selfshield:
        gamma_X = 0
    else:
        gamma_X = line.gamma_X
    
    # Ensure all scalar parameters are float32
    box = np.float32(self.box)
    velfac = np.float32(self.velfac)
    atime = np.float32(self.atime)
    lambda_X = np.float32(line.lambda_X * 1e-8)
    gamma_X_f32 = np.float32(gamma_X)
    fosc_X = np.float32(line.fosc_X)
    amumass_f32 = np.float32(amumass)
    tautail = np.float32(self.tautail)
    
    # Ensure all array parameters are float32 (except cofm which needs float64)
    pos = np.asarray(pos, dtype=np.float32)
    vel = np.asarray(vel, dtype=np.float32)
    elem_den = np.asarray(elem_den, dtype=np.float32)
    temp = np.asarray(temp, dtype=np.float32)
    hh = np.asarray(hh, dtype=np.float32)
    axis = np.asarray(self.axis, dtype=np.int32)
    cofm = np.asarray(self.cofm, dtype=np.float64)  # cofm must be float64!
    
    return _PI_original(get_tau*1, self.nbins, self.kernel_int, box, velfac, atime, 
                        lambda_X, gamma_X_f32, fosc_X, amumass_f32, tautail, 
                        pos, vel, elem_den, temp, hh, axis, cofm)

# Apply the patch
spectra.Spectra._do_interpolation_work = _do_interpolation_work_fixed

print("Applied fake_spectra bugfixes for Python 3.13 compatibility")

# Output filename based on snapshot name
savefile = f'camel_lya_spectra_{snapshot_name}.hdf5'

# Create Spectra object
spec = spectra.Spectra(
    num=snapshot_num,          # Snapshot number
    base=snapshot_dir if snapshot_dir else '.',  # Base directory
    cofm=cofm,                 # Sightline positions
    axis=axis,                 # Sightline directions
    res=resolution,            # Velocity resolution
    savefile=savefile,
    reload_file=not args.no_reload  # Force recompute unless --no-reload specified
)

print(f"  Loaded snapshot successfully")
print(f"  Velocity range: {spec.vmax:.2f} km/s")
print(f"  Resolution: {spec.dvbin:.4f} km/s/pixel")

# Compute Lyman-alpha optical depth
spec.get_tau("H", 1, 1215)

# Extract tau array
tau = spec.tau[("H", 1, 1215)]
print(f"  ✓ Computed optical depths: {tau.shape}")

# Save to file
spec.save_file()
print(f"  ✓ Saved spectra to: {spec.savefile}")

# ============================================================================
# STEP 4: Analysis - Compute flux and statistics
# ============================================================================
print("\n[Step 4] Computing flux statistics...")

flux = np.exp(-tau)
mean_flux = np.mean(flux)
eff_tau = -np.log(mean_flux)

print(f"  Mean flux: {mean_flux:.4f}")
print(f"  Effective optical depth: {eff_tau:.4f}")
print(f"  Flux range: [{flux.min():.4f}, {flux.max():.4f}]")
print(f"  Tau range: [{tau.min():.4f}, {tau.max():.4f}]")

# Deep absorption statistics
deep = (flux < 0.1).sum()
total_pixels = flux.size
print(f"  Deep absorption (F < 0.1): {100*deep/total_pixels:.2f}% of pixels")

# ============================================================================
# STEP 5: Create comprehensive plots
# ============================================================================
print("\n[Step 5] Creating analysis plots...")

Path('plots').mkdir(exist_ok=True)

# Velocity axis
velocity = np.arange(tau.shape[1]) * spec.dvbin

# ===== PLOT 1: Sample Spectra =====
print("  (1/4) Sample spectra...")

fig1, axes = plt.subplots(5, 1, figsize=(14, 10))
indices = np.random.choice(num_sightlines, min(5, num_sightlines), replace=False)

for i, (ax, idx) in enumerate(zip(axes, indices)):
    ax.plot(velocity, flux[idx], 'k-', lw=0.8, alpha=0.8)
    ax.set_ylabel('Flux', fontsize=10)
    ax.set_ylim(-0.05, 1.15)
    ax.axhline(1.0, color='r', ls='--', alpha=0.3, lw=1)
    ax.axhline(0.0, color='gray', ls='--', alpha=0.3, lw=1)
    ax.grid(alpha=0.2)
    ax.set_title(f'Sightline {idx} (z={redshift:.2f})', fontsize=11)
    
    if i == 4:
        ax.set_xlabel('Velocity [km/s]', fontsize=11)

plt.suptitle(f'CAMEL Lyman-α Spectra (z={redshift:.2f})', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'plots/camel_sample_spectra_{snapshot_name}.png', dpi=150, bbox_inches='tight')
print(f"     → plots/camel_sample_spectra_{snapshot_name}.png")

# ===== PLOT 2: Flux Statistics =====
print("  (2/4) Flux statistics...")

fig2, axes = plt.subplots(2, 2, figsize=(12, 8))

# Mean flux profile
mean_flux_profile = np.mean(flux, axis=0)
std_flux = np.std(flux, axis=0)
axes[0, 0].plot(velocity, mean_flux_profile, 'b-', lw=2, label='Mean')
axes[0, 0].fill_between(velocity, 
                        mean_flux_profile - std_flux, 
                        mean_flux_profile + std_flux, 
                        alpha=0.3, label='±1σ')
axes[0, 0].set_xlabel('Velocity [km/s]')
axes[0, 0].set_ylabel('Flux')
axes[0, 0].set_title(f'Mean Flux Profile (z={redshift:.2f})')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.2)
axes[0, 0].axhline(mean_flux, color='r', ls='--', alpha=0.5, 
                   label=f'<F>={mean_flux:.3f}')

# Flux distribution
axes[0, 1].hist(flux.flatten(), bins=100, color='steelblue', 
                edgecolor='k', alpha=0.7, density=True)
axes[0, 1].axvline(mean_flux, color='r', ls='--', lw=2, 
                   label=f'Mean = {mean_flux:.3f}')
axes[0, 1].set_xlabel('Normalized Flux')
axes[0, 1].set_ylabel('PDF')
axes[0, 1].set_title('Flux Distribution')
axes[0, 1].set_yscale('log')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.2)

# Mean optical depth
mean_tau_profile = np.mean(tau, axis=0)
axes[1, 0].plot(velocity, mean_tau_profile, 'r-', lw=2)
axes[1, 0].set_xlabel('Velocity [km/s]')
axes[1, 0].set_ylabel('Optical Depth')
axes[1, 0].set_title('Mean Optical Depth')
axes[1, 0].grid(alpha=0.2)

# Effective optical depth per sightline
eff_tau_per_sightline = -np.log(np.mean(flux, axis=1))
axes[1, 1].hist(eff_tau_per_sightline, bins=30, color='coral', 
                edgecolor='k', alpha=0.7)
axes[1, 1].axvline(eff_tau, color='r', ls='--', lw=2, 
                   label=f'Mean = {eff_tau:.3f}')
axes[1, 1].set_xlabel('Effective Optical Depth')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title(f'τ_eff Distribution (z={redshift:.2f})')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.2)

plt.tight_layout()
plt.savefig(f'plots/camel_flux_statistics_{snapshot_name}.png', dpi=150, bbox_inches='tight')
print(f"     → plots/camel_flux_statistics_{snapshot_name}.png")

# ===== PLOT 3: Flux Power Spectrum =====
print("  (3/4) Flux power spectrum...")

# Fluctuation field
delta_F = (flux - mean_flux) / mean_flux

# Compute power spectrum for each sightline
n_spectra, n_pixels = delta_F.shape
P_k_all = []

for i in range(n_spectra):
    fft = np.fft.rfft(delta_F[i])
    power = np.abs(fft)**2 / n_pixels
    P_k_all.append(power)

P_k_mean = np.mean(P_k_all, axis=0)
P_k_std = np.std(P_k_all, axis=0)

# k values
k = np.fft.rfftfreq(n_pixels, d=spec.dvbin)  # 1/(km/s)

fig3, ax = plt.subplots(figsize=(10, 6))

# Plot (skip k=0)
ax.loglog(k[1:], P_k_mean[1:], 'o-', lw=2, markersize=3, label='Mean')
ax.fill_between(k[1:], P_k_mean[1:] - P_k_std[1:], 
                P_k_mean[1:] + P_k_std[1:], alpha=0.3)
ax.set_xlabel('k [s/km]', fontsize=12)
ax.set_ylabel('P(k) [dimensionless]', fontsize=12)
ax.set_title(f'Flux Power Spectrum (z={redshift:.2f}, <F>={mean_flux:.3f})', 
             fontsize=13, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3, which='both')

plt.tight_layout()
plt.savefig(f'plots/camel_power_spectrum_{snapshot_name}.png', dpi=150, bbox_inches='tight')
print(f"     → plots/camel_power_spectrum_{snapshot_name}.png")

# ===== PLOT 4: Transmission Statistics =====
print("  (4/4) Transmission statistics...")

fig4, axes = plt.subplots(2, 2, figsize=(12, 8))

# Transmission fraction distribution
transmission = np.mean(flux, axis=1)
axes[0, 0].hist(transmission, bins=30, color='green', 
                edgecolor='k', alpha=0.7)
axes[0, 0].axvline(mean_flux, color='r', ls='--', lw=2, 
                   label=f'Mean = {mean_flux:.3f}')
axes[0, 0].set_xlabel('Mean Transmission per Sightline')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('Transmission Distribution')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.2)

# Optical depth distribution
axes[0, 1].hist(tau.flatten(), bins=100, color='purple', 
                edgecolor='k', alpha=0.7, range=(0, 10))
axes[0, 1].set_xlabel('Optical Depth τ')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title('Optical Depth Distribution')
axes[0, 1].set_yscale('log')
axes[0, 1].grid(alpha=0.2)

# Cumulative distribution of flux
sorted_flux = np.sort(flux.flatten())
cumulative = np.arange(1, len(sorted_flux) + 1) / len(sorted_flux)
axes[1, 0].plot(sorted_flux, cumulative, 'b-', lw=2)
axes[1, 0].set_xlabel('Flux')
axes[1, 0].set_ylabel('Cumulative Probability')
axes[1, 0].set_title('Flux CDF')
axes[1, 0].grid(alpha=0.2)
axes[1, 0].axvline(mean_flux, color='r', ls='--', alpha=0.5)

# Statistics summary table
axes[1, 1].axis('off')
stats_text = f"""
CAMEL Lyman-α Forest Statistics
Snapshot: {snapshot_name} (z = {redshift:.3f})
═══════════════════════════════════════

Flux Statistics:
  Mean flux:              {mean_flux:.4f}
  Median flux:            {np.median(flux):.4f}
  Std dev:                {np.std(flux):.4f}

Optical Depth:
  Effective τ_eff:        {eff_tau:.4f}
  Mean τ:                 {np.mean(tau):.4f}
  Median τ:               {np.median(tau):.4f}

Absorption:
  Deep (F < 0.1):         {100*deep/total_pixels:.2f}%
  Moderate (0.1≤F<0.5):   {100*(((flux>=0.1)&(flux<0.5)).sum())/total_pixels:.2f}%
  Weak (F ≥ 0.5):         {100*((flux>=0.5).sum())/total_pixels:.2f}%

Sightlines:              {num_sightlines}
Pixels per sightline:    {n_pixels}
Resolution:              {spec.dvbin:.4f} km/s/pixel
"""
axes[1, 1].text(0.1, 0.5, stats_text, fontfamily='monospace', 
                fontsize=10, verticalalignment='center')

plt.tight_layout()
plt.savefig(f'plots/camel_transmission_stats_{snapshot_name}.png', dpi=150, bbox_inches='tight')
print(f"     → plots/camel_transmission_stats_{snapshot_name}.png")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
print(f"\nGenerated spectra: {num_sightlines} sightlines at z={redshift:.3f}")
print(f"Mean flux: <F> = {mean_flux:.4f}")
print(f"Effective optical depth: τ_eff = {eff_tau:.4f}")
print(f"\nPlots saved:")
print(f"  1. plots/camel_sample_spectra_{snapshot_name}.png")
print(f"  2. plots/camel_flux_statistics_{snapshot_name}.png")
print(f"  3. plots/camel_power_spectrum_{snapshot_name}.png")
print(f"  4. plots/camel_transmission_stats_{snapshot_name}.png")
print(f"\nSpectra data saved: {spec.savefile}")
print("\n" + "="*70)
