#!/usr/bin/env python3
"""
Analyze Lyman-Alpha Spectra
============================

This script demonstrates how to analyze Lyman-alpha forest spectra,
including computing the flux power spectrum mentioned in your meeting notes.

WHAT IS THE FLUX POWER SPECTRUM?
---------------------------------
The flux power spectrum P(k) measures fluctuations in the transmitted flux
as a function of wavenumber k (spatial frequency).

Why it's useful:
1. Sensitive to cosmological parameters (Omega_m, sigma_8)
2. Sensitive to astrophysical parameters (feedback)
3. Can be measured from observations
4. Allows model-to-observation comparison

Think of it like this:
- Low k (large scales): Measures large-scale density fluctuations
- High k (small scales): Sensitive to small-scale physics (feedback, temperature)

THE ANALYSIS PIPELINE:
---------------------
1. Load the spectra (flux vs. velocity/wavelength)
2. Normalize the flux (divide by mean)
3. Compute 1D Fourier transform
4. Calculate power spectrum P(k) = |FFT(flux)|^2
5. Average over multiple sightlines
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy import fft

def load_spectra(spectra_file):
    """
    Load previously generated spectra from NPZ file.
    """
    print("="*60)
    print("LOADING SPECTRA")
    print("="*60)
    
    data = np.load(spectra_file)
    
    print(f"Loaded: {spectra_file}")
    print(f"  Redshift: z = {data['redshift']:.4f}")
    print(f"  Number of sightlines: {data['flux'].shape[0]}")
    print(f"  Pixels per sightline: {data['flux'].shape[1]}")
    
    return data

def compute_flux_power_spectrum(flux, velocity):
    """
    Compute the 1D flux power spectrum.
    
    Args:
        flux: 2D array of flux values [n_sightlines, n_pixels]
        velocity: 1D array of velocity bins [n_pixels]
    
    Returns:
        k: Wavenumbers in s/km
        P_k: Power spectrum values
    """
    print("\n" + "="*60)
    print("COMPUTING FLUX POWER SPECTRUM")
    print("="*60)
    
    n_sightlines, n_pixels = flux.shape
    
    # Step 1: Normalize flux by mean
    print("\nStep 1: Normalizing flux...")
    mean_flux = np.mean(flux)
    print(f"  Mean flux: {mean_flux:.4f}")
    
    # Flux contrast: delta_F = F/<F> - 1
    delta_F = flux / mean_flux - 1.0
    print(f"  Flux contrast range: [{delta_F.min():.3f}, {delta_F.max():.3f}]")
    
    # Step 2: Compute Fourier transform for each sightline
    print("\nStep 2: Computing FFT for each sightline...")
    
    # Get wavenumbers from velocity spacing
    # velocity array is 3D: (n_sightlines, n_pixels, 3)
    # Extract 1D velocity grid from first sightline, first component (along sightline axis)
    if velocity.ndim == 3:
        velocity_1d = velocity[0, :, 0]  # Shape: (n_pixels,)
    elif velocity.ndim == 2:
        velocity_1d = velocity[0, :]  # Shape: (n_pixels,)
    else:
        velocity_1d = velocity  # Already 1D
    
    dv = velocity_1d[1] - velocity_1d[0]  # km/s (now a scalar)
    print(f"  Velocity spacing: {dv:.2f} km/s")
    
    # Wavenumber array (s/km)
    k = fft.fftfreq(n_pixels, d=dv)
    k = k[:n_pixels//2]  # Only positive frequencies
    
    # Compute FFT and power for each sightline
    power_spectra = []
    for i in range(n_sightlines):
        # FFT of flux contrast
        flux_fft = fft.fft(delta_F[i])
        
        # Power spectrum (only positive k)
        power = np.abs(flux_fft[:n_pixels//2])**2 / n_pixels
        power_spectra.append(power)
    
    # Step 3: Average over all sightlines
    print(f"\nStep 3: Averaging over {n_sightlines} sightlines...")
    P_k_mean = np.mean(power_spectra, axis=0)
    P_k_std = np.std(power_spectra, axis=0)
    
    # Multiply by velocity bin size to get proper normalization
    P_k_mean *= dv
    P_k_std *= dv
    
    print(f"  Power spectrum computed!")
    print(f"  k range: [{k[1]:.4e}, {k[-1]:.4e}] s/km")
    
    return k, P_k_mean, P_k_std

def plot_power_spectrum(k, P_k, P_k_std, output_dir="output"):
    """
    Plot the flux power spectrum.
    """
    print("\n" + "="*60)
    print("CREATING POWER SPECTRUM PLOT")
    print("="*60)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Skip first point (k=0, which is the mean)
    k_plot = k[1:]
    P_plot = P_k[1:]
    P_std_plot = P_k_std[1:]
    
    # Plot mean and standard deviation
    ax.loglog(k_plot, P_plot, 'b-', linewidth=2, label='Mean Power Spectrum')
    ax.fill_between(k_plot, P_plot - P_std_plot, P_plot + P_std_plot,
                     alpha=0.3, label='1-sigma variation')
    
    ax.set_xlabel('Wavenumber k (s/km)', fontsize=12)
    ax.set_ylabel('Power P(k) (km/s)', fontsize=12)
    ax.set_title('Lyman-Alpha Flux Power Spectrum', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "flux_power_spectrum.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Saved: {plot_path}")

def compare_statistics(flux):
    """
    Compute various statistics useful for comparison.
    """
    print("\n" + "="*60)
    print("STATISTICAL MEASURES")
    print("="*60)
    
    # Mean flux
    mean_flux = np.mean(flux)
    print(f"\nMean transmitted flux: {mean_flux:.4f}")
    print("  -> Lower = more absorption")
    print("  -> Sensitive to: redshift, UV background, IGM temperature")
    
    # Flux PDF (probability distribution function)
    print("\nFlux distribution:")
    flux_flat = flux.flatten()
    percentiles = [10, 25, 50, 75, 90]
    for p in percentiles:
        val = np.percentile(flux_flat, p)
        print(f"  {p}th percentile: {val:.4f}")
    
    # Effective optical depth
    tau_eff = -np.log(mean_flux)
    print(f"\nEffective optical depth: {tau_eff:.4f}")
    print("  -> This is what's often quoted in papers")
    print("  -> tau_eff = -ln(<F>)")
    
    # Flux variance
    flux_var = np.var(flux)
    print(f"\nFlux variance: {flux_var:.6f}")
    print("  -> Measures fluctuations in the forest")
    print("  -> Related to matter power spectrum")

def main():
    if len(sys.argv) < 2:
        print("Usage: python 04_analyze_spectra.py <spectra.npz>")
        print("\nExample:")
        print("  python 04_analyze_spectra.py output/lyman_alpha_spectra.npz")
        
        # Try to find generated spectra
        if os.path.exists("output/lyman_alpha_spectra.npz"):
            print("\nFound: output/lyman_alpha_spectra.npz")
            print("Run: python 04_analyze_spectra.py output/lyman_alpha_spectra.npz")
        sys.exit(1)
    
    spectra_file = sys.argv[1]
    
    if not os.path.exists(spectra_file):
        print(f"Error: File not found: {spectra_file}")
        print("\nMake sure you've generated spectra using:")
        print("  python 03_generate_spectra.py")
        sys.exit(1)
    
    # Load spectra
    data = load_spectra(spectra_file)
    
    # Compute statistics
    compare_statistics(data['flux'])
    
    # Compute power spectrum
    k, P_k, P_k_std = compute_flux_power_spectrum(data['flux'], data['velocity'])
    
    # Save power spectrum
    output_dir = os.path.dirname(spectra_file) or "output"
    np.savez(
        os.path.join(output_dir, "flux_power_spectrum.npz"),
        k=k,
        P_k=P_k,
        P_k_std=P_k_std
    )
    print(f"\nSaved power spectrum: {os.path.join(output_dir, 'flux_power_spectrum.npz')}")
    
    # Plot power spectrum
    plot_power_spectrum(k, P_k, P_k_std, output_dir)
    
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    print("""
The flux power spectrum tells us about structure in the IGM:

1. SHAPE of P(k):
   - Reflects the matter power spectrum
   - Sensitive to cosmological parameters
   
2. AMPLITUDE:
   - Affected by mean flux (ionization state)
   - Depends on UV background strength
   
3. SMALL-SCALE BEHAVIOR (high k):
   - Affected by IGM temperature
   - Sensitive to feedback mechanisms
   - This is where stellar vs AGN feedback differs!

NEXT STEPS FOR YOUR RESEARCH:
------------------------------
1. Generate spectra from multiple CAMELS simulations
   - Compare LH_0, LH_1, ..., LH_N (different parameter combinations)
   
2. Focus on variations in:
   - Stellar feedback parameter
   - AGN feedback parameter
   
3. Compare power spectra:
   - How does P(k) change with feedback strength?
   - Can you train an ML model to predict feedback from P(k)?
   
4. Compare with real data:
   - Download observed Lyman-alpha power spectra
   - Which simulation matches best?

This is exactly what's outlined in your meeting notes!
    """)

if __name__ == "__main__":
    main()
