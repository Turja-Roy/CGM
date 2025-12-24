import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# ============================================================================
# Parse command line arguments
# ============================================================================
if len(sys.argv) < 2:
    print("Error: No snapshot file provided")
    print("\nUsage: python Note1.py <snapshot_file>")
    print("\nExample:")
    print("  python Note1.py data/IllustrisTNG/LH/LH_0/snap_080.hdf5")
    print("\nSearching for available snapshots...")
    
    # Try to find snapshots in data directory
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

filepath = sys.argv[1]

# Check if file exists
if not os.path.exists(filepath):
    print(f"Error: File not found: {filepath}")
    sys.exit(1)

print("="*70)
print("TUTORIAL: Working with HDF5 Simulation Data")
print("="*70)
print(f"Analyzing: {filepath}\n")

# ============================================================================
# TECHNIQUE 1: Safe file opening and exploration
# ============================================================================
print("\n[1] Opening HDF5 files safely")
print("-" * 70)

with h5py.File(filepath, 'r') as f:  # 'r' = read-only mode
    print("✓ File opened successfully")
    print(f"  File path: {filepath}")
    print(f"  Top-level groups: {list(f.keys())}")

print("\n💡 Always use 'with' statement - it automatically closes the file!")

# ============================================================================
# TECHNIQUE 2: Inspecting metadata (attributes)
# ============================================================================
print("\n[2] Reading metadata from Header")
print("-" * 70)

with h5py.File(filepath, 'r') as f:
    header = f['Header']
    
    # Extract key cosmology info
    redshift = header.attrs['Redshift']
    boxsize = header.attrs['BoxSize']  # ckpc/h
    hubble = header.attrs['HubbleParam']
    omega0 = header.attrs['Omega0']
    
    print(f"  Redshift: z = {redshift:.3f}")
    print(f"  Box size: {boxsize:.1f} ckpc/h = {boxsize/1000:.1f} Mpc/h")
    print(f"  Hubble parameter: h = {hubble:.4f}")
    print(f"  Omega_matter: {omega0:.4f}")
    
    # Particle counts
    npart = header.attrs['NumPart_ThisFile']
    print(f"\n  Particle counts:")
    print(f"    Gas (Type 0): {npart[0]:,}")
    print(f"    DM (Type 1): {npart[1]:,}")
    print(f"    Stars (Type 4): {npart[4]:,}")

print("\n💡 Attributes contain metadata. Access with .attrs['key']")

# ============================================================================
# TECHNIQUE 3: Inspecting dataset properties WITHOUT loading data
# ============================================================================
print("\n[3] Checking dataset properties (no data loading)")
print("-" * 70)

with h5py.File(filepath, 'r') as f:
    gas = f['PartType0']
    
    print("  Available gas datasets:")
    for name in sorted(gas.keys()):
        dset = gas[name]
        # dset.shape and dset.dtype are fast - they don't load data!
        print(f"    {name:30s} shape={str(dset.shape):20s} dtype={dset.dtype}")

print("\n💡 .shape and .dtype don't load data - they're instant even for huge files!")

# ============================================================================
# TECHNIQUE 4: Loading data efficiently (slicing)
# ============================================================================
print("\n[4] Smart data loading with slicing")
print("-" * 70)

with h5py.File(filepath, 'r') as f:
    coords = f['PartType0/Coordinates']
    
    print(f"  Full dataset: {coords.shape} = {coords.shape[0]:,} particles")
    print(f"  Memory if loaded: ~{coords.nbytes / 1e9:.2f} GB")
    
    # Load only first 1000 particles
    sample = coords[:1000]  # This loads data!
    print(f"\n  Loaded sample: {sample.shape}")
    print(f"  Memory used: ~{sample.nbytes / 1e6:.2f} MB")
    print(f"  First particle position: [{sample[0,0]:.2f}, {sample[0,1]:.2f}, {sample[0,2]:.2f}] ckpc/h")

print("\n💡 Use slicing coords[:N] to load subsets. Full array [:] loads everything!")

# ============================================================================
# TECHNIQUE 5: Computing statistics on large datasets
# ============================================================================
print("\n[5] Computing statistics on large datasets")
print("-" * 70)

with h5py.File(filepath, 'r') as f:
    density = f['PartType0/Density']
    neutral_frac = f['PartType0/NeutralHydrogenAbundance']
    
    print(f"  Density array: {density.shape[0]:,} particles")
    
    # Load in chunks to avoid memory issues
    chunk_size = 100000
    n_particles = density.shape[0]
    
    # Calculate mean density using chunks
    density_sum = 0.0
    for i in range(0, n_particles, chunk_size):
        chunk = density[i:i+chunk_size]
        density_sum += np.sum(chunk)
    
    mean_density = density_sum / n_particles
    
    print(f"  Mean density: {mean_density:.6e} (10^10 Msun / (ckpc/h)^3)")
    
    # For neutral fraction, can load full array (it's just 16M floats ~ 64 MB)
    nH = neutral_frac[:]
    print(f"\n  Neutral hydrogen fraction:")
    print(f"    Min: {np.min(nH):.6e}")
    print(f"    Max: {np.max(nH):.6e}")
    print(f"    Mean: {np.mean(nH):.6e}")
    print(f"    Median: {np.median(nH):.6e}")

print("\n💡 For huge arrays, process in chunks. For smaller arrays (<1GB), load fully.")

# ============================================================================
# TECHNIQUE 6: Converting to physical units
# ============================================================================
print("\n[6] Converting to physical units")
print("-" * 70)

with h5py.File(filepath, 'r') as f:
    header = f['Header']
    
    # Unit conversions from header
    unit_length = header.attrs['UnitLength_in_cm']  # cm
    unit_mass = header.attrs['UnitMass_in_g']  # g
    unit_vel = header.attrs['UnitVelocity_in_cm_per_s']  # cm/s
    
    boxsize = header.attrs['BoxSize']  # ckpc/h
    hubble = header.attrs['HubbleParam']
    redshift = header.attrs['Redshift']
    
    # Convert box size to physical Mpc
    box_proper = boxsize / hubble / (1 + redshift) / 1000  # Mpc proper
    box_comoving = boxsize / hubble / 1000  # Mpc/h comoving
    
    print(f"  Box size:")
    print(f"    Code units: {boxsize:.1f} ckpc/h")
    print(f"    Comoving: {box_comoving:.2f} Mpc/h")
    print(f"    Proper (at z={redshift:.2f}): {box_proper:.2f} Mpc")
    
    # Load a small sample and convert velocities
    vel = f['PartType0/Velocities'][:10]  # km/s (physical peculiar velocities)
    print(f"\n  Sample particle velocities (first 3):")
    for i in range(3):
        v_mag = np.sqrt(np.sum(vel[i]**2))
        print(f"    Particle {i}: v = [{vel[i,0]:7.2f}, {vel[i,1]:7.2f}, {vel[i,2]:7.2f}] km/s, |v| = {v_mag:.2f} km/s")

print("\n💡 CAMEL uses: lengths in ckpc/h, velocities in km/s, masses in 10^10 Msun")

# ============================================================================
# TECHNIQUE 7: Selecting particles by properties
# ============================================================================
print("\n[7] Selecting particles by properties")
print("-" * 70)

with h5py.File(filepath, 'r') as f:
    # Load necessary fields
    density = f['PartType0/Density'][:]
    temp = f['PartType0/InternalEnergy'][:]  # (km/s)^2, need to convert
    nH = f['PartType0/NeutralHydrogenAbundance'][:]
    
    # Convert internal energy to temperature (rough approximation)
    # T = (gamma-1) * mu * m_p / k_B * u
    # For simplicity, T [K] ≈ u * 1.2e4 for ionized gas
    temp_K = temp * 1.2e4
    
    # Find high-density, cool, neutral gas (IGM absorbers)
    high_density = density > np.percentile(density, 90)
    cool = temp_K < 1e5  # < 100,000 K
    neutral = nH > 0.1  # > 10% neutral
    
    igm_absorbers = high_density & cool & neutral
    
    print(f"  Total particles: {len(density):,}")
    print(f"  High density (>90th percentile): {np.sum(high_density):,}")
    print(f"  Cool (T < 10^5 K): {np.sum(cool):,}")
    print(f"  Neutral (fHI > 0.1): {np.sum(neutral):,}")
    print(f"  IGM absorbers (all 3): {np.sum(igm_absorbers):,}")
    print(f"  Fraction: {100*np.sum(igm_absorbers)/len(density):.2f}%")

print("\n💡 Use boolean indexing to filter particles: arr[condition]")

# ============================================================================
# TECHNIQUE 8: Creating diagnostic plots
# ============================================================================
print("\n[8] Creating diagnostic plots")
print("-" * 70)

with h5py.File(filepath, 'r') as f:
    # Load positions (subsample for speed)
    stride = 100  # Use every 100th particle
    coords = f['PartType0/Coordinates'][::stride]  # Shape: (N/100, 3)
    nH = f['PartType0/NeutralHydrogenAbundance'][::stride]
    
    print(f"  Loaded {coords.shape[0]:,} particles (every {stride}th)")
    
    # Create projection plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 2D histogram (density projection)
    h1 = axes[0].hist2d(coords[:, 0], coords[:, 1], bins=200,
                        cmap='viridis', cmin=0)
    axes[0].set_xlabel('X [ckpc/h]')
    axes[0].set_ylabel('Y [ckpc/h]')
    axes[0].set_title('Gas Density Projection (xy plane)')
    plt.colorbar(h1[3], ax=axes[0], label='N particles per bin')
    
    # Neutral hydrogen distribution
    axes[1].hist(np.log10(nH + 1e-10), bins=100, color='steelblue',
                 edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('log10(Neutral Fraction)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Neutral Hydrogen Distribution')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Extract snapshot name for output file
    snapshot_name = os.path.basename(filepath).replace('.hdf5', '')
    output = f'plots/camel_snapshot_exploration_{snapshot_name}.png'
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved plot to {output}")

print("\n💡 Always subsample large datasets for visualization!")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("SUMMARY: Key Techniques for HDF5 Data")
print("="*70)
print("""
1. ✓ Use 'with h5py.File()' for safe file handling
2. ✓ Access metadata with .attrs['key']
3. ✓ Check .shape and .dtype before loading (fast!)
4. ✓ Use slicing [start:end] to load subsets
5. ✓ Process huge arrays in chunks
6. ✓ Understand units (check Header attributes)
7. ✓ Filter with boolean indexing: arr[condition]
8. ✓ Subsample for visualization (use stride [::N])

Next step:
- Run analyze_camel_spectra.py to generate Lyman-alpha spectra!
""")
