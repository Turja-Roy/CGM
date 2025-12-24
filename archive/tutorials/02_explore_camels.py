#!/usr/bin/env python3
"""
Explore CAMELS Snapshot Data
=============================

This script teaches you how to read and understand CAMELS HDF5 files.

HDF5 FORMAT:
------------
HDF5 is a hierarchical file format (like a file system within a file).
CAMELS snapshots use the Gadget/AREPO format:
- Group 0: Gas particles
- Group 1: Dark matter particles  
- Group 4: Star particles
- Group 5: Black hole particles

Each group contains arrays for particle properties:
- Coordinates: 3D positions [x, y, z] in comoving kpc/h
- Velocities: 3D velocities in km/s
- Masses: Particle masses
- Gas-specific: Density, Temperature, Electron abundance, etc.

UNDERSTANDING THE DATA:
-----------------------
1. Cosmological simulations track particles through cosmic time
2. Each snapshot = one moment in time (characterized by redshift z)
3. Box size: typically 25 Mpc/h (comoving)
4. Periodic boundary conditions (like a torus)

FOR LYMAN-ALPHA FOREST:
-----------------------
We need GAS particles because:
- Lyman-alpha absorption comes from neutral hydrogen (HI)
- HI traces the intergalactic medium (IGM)
- Gas density and temperature determine HI fraction
"""

import h5py
import numpy as np
import sys
import os

def explore_hdf5_structure(filename):
    """
    Print the structure of an HDF5 file to understand its organization.
    """
    print("\n" + "="*60)
    print(f"EXPLORING: {filename}")
    print("="*60)
    
    with h5py.File(filename, 'r') as f:
        print("\nTOP-LEVEL GROUPS:")
        print("-" * 40)
        for key in f.keys():
            print(f"  {key}")
        
        # Header contains simulation metadata
        if 'Header' in f:
            print("\nHEADER INFORMATION:")
            print("-" * 40)
            header = f['Header']
            
            # Important attributes
            attrs_to_show = [
                'Time', 'Redshift', 'BoxSize', 'NumPart_Total',
                'Omega0', 'OmegaLambda', 'HubbleParam'
            ]
            
            for attr in attrs_to_show:
                if attr in header.attrs:
                    value = header.attrs[attr]
                    if attr == 'Redshift':
                        print(f"  {attr}: {value:.4f} (z)")
                        print("\t-> This is when the snapshot was taken")
                        print("\t-> For Lyman-alpha, we want z ~ 2-4")
                    elif attr == 'BoxSize':
                        print(f"  {attr}: {value:.2f} (comoving kpc/h)")
                        print(f"\t-> Simulation box side length: {value/1000:.2f} Mpc/h")
                    elif attr == 'NumPart_Total':
                        print(f"  {attr}: {value}")
                        print("\t-> [Gas, DM, ..., Stars, BH]")
                    else:
                        print("{attr}: {value}")
        
        # Explore particle groups
        for group_name in ['PartType0', 'PartType1', 'PartType4', 'PartType5']:
            if group_name in f:
                print(f"\n{group_name} (", end="")
                if group_name == 'PartType0':
                    print("GAS PARTICLES - MOST IMPORTANT FOR LYMAN-ALPHA!):")
                elif group_name == 'PartType1':
                    print("DARK MATTER):")
                elif group_name == 'PartType4':
                    print("STARS):")
                elif group_name == 'PartType5':
                    print("BLACK HOLES - AGN!):")
                
                print("-" * 40)
                
                group = f[group_name]
                for dataset in group.keys():
                    data = group[dataset]
                    print(f"  {dataset}: shape {data.shape}, dtype {data.dtype}")
                    
                    # Add explanations for gas properties
                    if group_name == 'PartType0':
                        if dataset == 'Coordinates':
                            print("    -> 3D positions [x, y, z] of each gas particle")
                        elif dataset == 'Density':
                            print("    -> Gas density (affects HI absorption strength)")
                        elif dataset == 'InternalEnergy' or dataset == 'Temperature':
                            print("    -> Temperature (determines ionization state)")
                        elif dataset == 'ElectronAbundance':
                            print("    -> Ionization state (needed for HI fraction)")
                        elif dataset == 'NeutralHydrogenAbundance':
                            print("    -> Direct HI measurement!")

def analyze_gas_properties(filename):
    """
    Analyze gas properties relevant to Lyman-alpha forest.
    """
    print("\n" + "="*60)
    print("GAS PROPERTIES ANALYSIS")
    print("="*60)
    
    with h5py.File(filename, 'r') as f:
        if 'PartType0' not in f:
            print("No gas particles in this snapshot!")
            return
        
        gas = f['PartType0']
        
        # Get box size and redshift
        boxsize = f['Header'].attrs['BoxSize']
        redshift = f['Header'].attrs['Redshift']
        
        print(f"\nRedshift: z = {redshift:.4f}")
        print(f"Box size: {boxsize/1000:.2f} Mpc/h")
        print(f"Number of gas particles: {len(gas['Coordinates'])}")
        
        # Coordinates
        coords = gas['Coordinates'][:]
        print(f"\nCoordinate ranges:")
        print(f"  X: [{coords[:,0].min():.2f}, {coords[:,0].max():.2f}] kpc/h")
        print(f"  Y: [{coords[:,1].min():.2f}, {coords[:,1].max():.2f}] kpc/h")
        print(f"  Z: [{coords[:,2].min():.2f}, {coords[:,2].max():.2f}] kpc/h")
        
        # Density
        if 'Density' in gas:
            density = gas['Density'][:]
            print(f"\nDensity statistics:")
            print(f"  Min:    {density.min():.2e}")
            print(f"  Median: {density[len(density)//2]:.2e}")
            print(f"  Max:    {density.max():.2e}")
            print(f"  -> High density regions = galaxies/halos")
            print(f"  -> Low density regions = IGM (where we see Lyman-alpha forest)")
        
        # Temperature or Internal Energy
        if 'Temperature' in gas:
            temp = gas['Temperature'][:]
            print(f"\nTemperature statistics (K):")
            print(f"  Min:    {temp.min():.2e}")
            print(f"  Median: {np.median(temp):.2e}")
            print(f"  Max:    {temp.max():.2e}")
            print(f"  -> IGM is typically ~10^4 K")
            print(f"  -> Hot gas (~10^6 K) is highly ionized -> less HI -> less absorption")
        
        # Neutral hydrogen
        if 'NeutralHydrogenAbundance' in gas:
            nH = gas['NeutralHydrogenAbundance'][:]
            print(f"\nNeutral Hydrogen Abundance:")
            print(f"  Min:    {nH.min():.2e}")
            print(f"  Median: {np.median(nH):.2e}")
            print(f"  Max:    {nH.max():.2e}")
            print(f"  -> This directly tells us HI content!")
            print(f"  -> Fake Spectra will use this to compute absorption")

def main():
    if len(sys.argv) < 2:
        print("Usage: python 02_explore_camels.py <path_to_snapshot.hdf5>")
        print("\nExample: python 02_explore_camels.py data/IllustrisTNG/LH_0/snap_033.hdf5")
        
        # Try to find a downloaded snapshot
        if os.path.exists("data"):
            print("\nLooking for downloaded snapshots...")
            for root, dirs, files in os.walk("data"):
                for file in files:
                    if file.endswith(".hdf5"):
                        snapshot = os.path.join(root, file)
                        print(f"Found: {snapshot}")
                        print(f"Run: python 02_explore_camels.py {snapshot}")
        sys.exit(1)
    
    snapshot_file = sys.argv[1]
    
    if not os.path.exists(snapshot_file):
        print(f"Error: File not found: {snapshot_file}")
        print("\nMake sure you've downloaded data using:")
        print("  python 01_setup_and_download.py")
        sys.exit(1)
    
    # Explore the file structure
    explore_hdf5_structure(snapshot_file)
    
    # Analyze gas properties
    analyze_gas_properties(snapshot_file)
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("Now that you understand the data structure, we can:")
    print("1. Generate Lyman-alpha spectra using Fake Spectra")
    print("2. This involves shooting 'sightlines' through the gas")
    print("3. Computing HI optical depth along each sightline")
    print("4. Converting optical depth to observed flux")
    print("\nRun: python 03_generate_spectra.py")

if __name__ == "__main__":
    main()
