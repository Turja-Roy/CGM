#!/usr/bin/env python3
"""
Generate Lyman-Alpha Spectra with Fake Spectra
===============================================

This script demonstrates how to use the fake_spectra package to generate
synthetic Lyman-alpha forest spectra from CAMELS simulations.

WHAT IS FAKE SPECTRA DOING?
----------------------------
1. Takes a simulation snapshot (gas particle data)
2. Shoots "sightlines" through the simulation box (like light from quasars)
3. For each sightline:
   - Samples gas properties along the line
   - Calculates neutral hydrogen (HI) column density
   - Computes optical depth tau for Lyman-alpha absorption
   - Converts to observed flux: F = exp(-tau)

THE PHYSICS:
-----------
Lyman-alpha absorption occurs when HI absorbs photons at 121.6 nm (rest frame).
Due to Hubble expansion, this is redshifted to longer wavelengths.

Optical depth: tau ~ nHI * cross_section * path_length
- High nHI -> low flux (strong absorption)
- Low nHI -> high flux (little absorption)

WHY SIGHTLINES?
---------------
In real observations, we observe light from distant quasars passing through
the IGM. Each quasar provides one "sightline" through the universe.
We simulate this by randomly sampling rays through our simulation box.

PARAMETERS TO EXPLORE:
----------------------
- Number of sightlines
- Sightline length
- Resolution (pixels per unit length)
- Redshift
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Apply dtype fix for fake_spectra (must be imported before using fake_spectra)
try:
    import fix_fake_spectra_dtype
    print("Applied dtype fix for fake_spectra")
except ImportError:
    print("Warning: Could not import fix_fake_spectra_dtype - may encounter dtype errors")

# Import fake_spectra
try:
    import fake_spectra.spectra as fsp
    import fake_spectra.griddedspectra as gsp
except ImportError:
    print("Error: fake_spectra not installed!")
    print("Install with: pip install fake_spectra")
    sys.exit(1)

def generate_spectra(snapshot_base, snapshot_num, output_dir="output", num_sightlines=5):
    """
    Generate Lyman-alpha forest spectra from a CAMELS snapshot.
    
    Args:
        snapshot_base: Base directory containing snapshot files
        snapshot_num: Snapshot number (e.g., 33 for snap_033.hdf5)
        output_dir: Directory to save output spectra and plots
        num_sightlines: Number of random sightlines to generate
    """
    
    print("="*60)
    print("GENERATING LYMAN-ALPHA SPECTRA")
    print("="*60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct expected filename
    snap_str = str(snapshot_num).rjust(3, '0')
    expected_file = os.path.join(snapshot_base, f"snap_{snap_str}.hdf5")
    
    print(f"\nBase directory: {snapshot_base}")
    print(f"Snapshot number: {snapshot_num} (snap_{snap_str}.hdf5)")
    print(f"Expected file: {expected_file}")
    print(f"Number of sightlines: {num_sightlines}")
    
    # Verify file exists
    if not os.path.exists(expected_file):
        print(f"\nError: Snapshot file not found!")
        print(f"Looking for: {expected_file}")
        print("\nMake sure you've created/downloaded the snapshot file.")
        return None
    
    # STEP 1: Load the snapshot
    # -------------------------
    print("\nSTEP 1: Loading snapshot with fake_spectra...")
    print("This reads the HDF5 file and prepares gas particle data")
    print(f"Note: fake_spectra expects files named snap_XXX.hdf5")
    
    # GriddedSpectra is the main class
    # It creates a grid of spectra at different positions
    try:
        spectra = gsp.GriddedSpectra(
            num=snapshot_num,  # Snapshot number
            base=snapshot_base,  # Base directory
            nspec=num_sightlines,  # Number of spectra to generate
            axis=2,  # Line of sight axis (0=x, 1=y, 2=z)
            res=1.0,    # Resolution in comoving kpc/h per pixel
            savefile=os.path.join(output_dir, "spectra_grid.hdf5")
        )
    except Exception as e:
        print(f"\nError loading snapshot: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure the snapshot file is named snap_XXX.hdf5")
        print("2. Pass the directory path and snapshot number separately")
        print("3. Verify you have enough memory to load the snapshot")
        import traceback
        traceback.print_exc()
        return None
    
    print(f"  Loaded! Redshift: z = {spectra.red:.4f}")
    print(f"  Box size: {spectra.box:.2f} comoving kpc/h")
    
    # STEP 2: Generate the spectra
    # ----------------------------
    print("\nSTEP 2: Generating optical depth spectra...")
    print("This computes tau (optical depth) along each sightline")
    print("  tau depends on: HI density, temperature, velocity")
    
    # Get the optical depth for HI Lyman-alpha (species 0 = HI, line 0 = Lyman-alpha)
    tau_lya = spectra.get_tau("H", 1, 1215)  # Hydrogen, ionization 1, wavelength 1215 Angstrom
    
    print(f"  tau array shape: {tau_lya.shape}")
    print(f"    -> {tau_lya.shape[0]} sightlines")
    print(f"    -> {tau_lya.shape[1]} pixels along each sightline")
    
    # STEP 3: Convert optical depth to flux
    # -------------------------------------
    print("\nSTEP 3: Converting optical depth to observed flux...")
    print("  Flux F = exp(-tau)")
    print("  F = 1 means no absorption (transmitted)")
    print("  F = 0 means complete absorption (absorbed)")
    
    flux = np.exp(-tau_lya)
    
    # STEP 4: Get wavelength/velocity arrays
    # --------------------------------------
    print("\nSTEP 4: Calculating wavelength/velocity arrays...")
    
    # Velocity along sightline (in km/s, relative to systemic redshift)
    # Get velocity for HI (Hydrogen, ion=1)
    vels = spectra.get_velocity("H", 1)
    
    print(f"  Velocity array shape: {vels.shape}")
    if len(vels.shape) > 1:
        print(f"  Velocity range (first sightline): [{vels[0].min():.1f}, {vels[0].max():.1f}] km/s")
    else:
        print(f"  Velocity range: [{vels.min():.1f}, {vels.max():.1f}] km/s")
    
    # STEP 5: Save the spectra
    # ------------------------
    print("\nSTEP 5: Saving spectra...")
    
    np.savez(
        os.path.join(output_dir, "lyman_alpha_spectra.npz"),
        tau=tau_lya,
        flux=flux,
        velocity=vels,
        redshift=float(spectra.red),  # type: ignore
        box_size=float(spectra.box)  # type: ignore
    )
    
    print(f"  Saved to: {os.path.join(output_dir, 'lyman_alpha_spectra.npz')}")
    
    return {
        'spectra': spectra,
        'tau': tau_lya,
        'flux': flux,
        'velocity': vels
    }

def plot_spectra(results, output_dir="output", num_to_plot=3):
    """
    Create plots to visualize the Lyman-alpha spectra.
    """
    
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    tau = results['tau']
    flux = results['flux']
    vels = results['velocity']
    
    # Handle velocity array shape - it can be (nsightlines, npixels, 3) or (npixels,)
    if len(vels.shape) == 3:
        vels = vels[:, :, 0]  # Use x-component of velocity
    
    num_sightlines = min(num_to_plot, tau.shape[0])
    
    # Plot 1: Optical depth and flux for a few sightlines
    # ---------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    for i in range(num_sightlines):
        axes[0].plot(vels[i], tau[i], alpha=0.7, label=f'Sightline {i+1}')
        axes[1].plot(vels[i], flux[i], alpha=0.7, label=f'Sightline {i+1}')
    
    axes[0].set_ylabel('Optical Depth (tau)', fontsize=12)
    axes[0].set_title('Lyman-Alpha Forest: Optical Depth and Flux', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    axes[1].set_xlabel('Velocity (km/s)', fontsize=12)
    axes[1].set_ylabel('Normalized Flux', fontsize=12)
    axes[1].axhline(y=1, color='k', linestyle='--', alpha=0.3, label='No absorption')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.1])
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "lyman_alpha_spectra.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot: {plot_path}")
    
    # Plot 2: Flux distribution
    # -------------------------
    plt.figure(figsize=(10, 6))
    
    flux_flat = flux.flatten()
    plt.hist(flux_flat, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Flux', fontsize=12)
    plt.ylabel('Number of pixels', fontsize=12)
    plt.title('Distribution of Flux Values\n(shows how much absorption we see)', fontsize=14)
    plt.axvline(x=1, color='r', linestyle='--', label='No absorption')
    plt.axvline(x=flux_flat.mean(), color='g', linestyle='--', label=f'Mean = {flux_flat.mean():.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    dist_path = os.path.join(output_dir, "flux_distribution.png")
    plt.savefig(dist_path, dpi=150)
    print(f"Saved plot: {dist_path}")
    
    print("\n" + "="*60)
    print("INTERPRETING THE RESULTS:")
    print("="*60)
    print(f"Mean flux: {flux_flat.mean():.4f}")
    print(f"  -> Lower mean flux = more absorption")
    print(f"  -> This depends on: redshift, cosmology, feedback")
    print(f"\nMedian optical depth: {np.median(tau):.4f}")
    print(f"  -> Related to mean flux by: <F> ~ exp(-<tau>)")
    print(f"\nNumber of absorption lines: {np.sum(flux < 0.8)}")
    print(f"  -> Regions where flux drops below 0.8")
    print(f"  -> These are the 'forest' of absorption lines!")

def main():
    if len(sys.argv) < 3:
        print("Usage: python 03_generate_spectra.py <base_directory> <snapshot_number> [num_sightlines]")
        print("\nExamples:")
        print("  python 03_generate_spectra.py data/mock/LH_MOCK 33 5")
        print("  python 03_generate_spectra.py data/IllustrisTNG/LH_0 33 10")
        print("\nNote: The script looks for snap_XXX.hdf5 in the base directory")
        print("      where XXX is the 3-digit snapshot number (e.g., 033)")
        
        # Try to find a downloaded snapshot
        if os.path.exists("data"):
            print("\nLooking for downloaded snapshots...")
            for root, dirs, files in os.walk("data"):
                for file in files:
                    if file.startswith("snap_") and file.endswith(".hdf5"):
                        # Extract snapshot number
                        snap_num = file.replace("snap_", "").replace(".hdf5", "")
                        print(f"Found: {os.path.join(root, file)}")
                        print(f"  -> Run: python 03_generate_spectra.py {root} {int(snap_num)}")
        sys.exit(1)
    
    snapshot_base = sys.argv[1]
    snapshot_num = int(sys.argv[2])
    num_sightlines = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    
    if not os.path.exists(snapshot_base):
        print(f"Error: Directory not found: {snapshot_base}")
        sys.exit(1)
    
    # Generate spectra
    results = generate_spectra(snapshot_base, snapshot_num, num_sightlines=num_sightlines)
    
    if results is not None:
        # Create plots
        plot_spectra(results)
        
        print("\n" + "="*60)
        print("SUCCESS!")
        print("="*60)
        print("You've successfully generated Lyman-alpha forest spectra!")
        print("\nNext steps:")
        print("1. Try different snapshots (different redshifts)")
        print("2. Compare different simulations (LH_0 vs LH_1, etc.)")
        print("3. Compute the flux power spectrum (mentioned in your meeting)")
        print("4. Run: python 04_analyze_spectra.py")

if __name__ == "__main__":
    main()
