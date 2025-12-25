import os
import numpy as np
import h5py

import scripts.config as config
from scripts.hdf5_io import explore_hdf5_structure


# List all available simulation data and spectra files
def cmd_list(args):
    config.list_all_available_data()
    return 0


# Explore HDF5 file structure and contents
def cmd_explore(args):
    spectra_file = args.spectra_file

    if not os.path.exists(spectra_file):
        print(f"Error: File not found: {spectra_file}")
        return 1

    print("=" * 70)
    print(f"EXPLORING: {spectra_file}")
    print("=" * 70)

    # Use utils function to explore structure
    explore_hdf5_structure(spectra_file)

    # If it's a spectra file, show some sample data
    with h5py.File(spectra_file, 'r') as f:
        # Check for tau data (supports both old and new formats)
        tau_datasets = []

        # New format: tau/Element/Ion/Wavelength
        if 'tau' in f and isinstance(f['tau'], h5py.Group):
            print("\n" + "=" * 70)
            print("SAMPLE SPECTRA DATA")
            print("=" * 70)

            # Find all tau datasets
            def find_tau_datasets(group, path=''):
                datasets = []
                for key in group.keys():
                    item = group[key]
                    if isinstance(item, h5py.Dataset):
                        datasets.append((path + '/' + key, item))
                    elif isinstance(item, h5py.Group):
                        datasets.extend(find_tau_datasets(
                            item, path + '/' + key))
                return datasets

            tau_datasets = find_tau_datasets(f['tau'], 'tau')

            # Show first tau dataset found
            if tau_datasets:
                for path, dataset in tau_datasets[:1]:  # Show first one
                    print(f"\nDataset: {path}")
                    print(f"Shape: {dataset.shape}")

                    # Show statistics for first 3 sightlines
                    n_show = min(3, dataset.shape[0])
                    for i in range(n_show):
                        tau_i = dataset[i, :]
                        flux_i = np.exp(-tau_i)
                        print(f"\nSightline {i}:")
                        print(f"  tau:  {tau_i.min():.3e} to {
                              tau_i.max():.3e}, mean={tau_i.mean():.3f}")
                        print(f"  flux: {flux_i.min():.3f} to {
                              flux_i.max():.3f}, mean={flux_i.mean():.3f}")
            else:
                print("  No tau datasets found")

        # Old format: tau as direct dataset
        elif 'tau' in f and isinstance(f['tau'], h5py.Dataset):
            print("\n" + "=" * 70)
            print("SAMPLE SPECTRA DATA (Old Format)")
            print("=" * 70)

            tau = f['tau']
            print(f"Optical depth array shape: {tau.shape}")

            # Show statistics for first 3 sightlines
            n_show = min(3, tau.shape[0])
            for i in range(n_show):
                tau_i = tau[i, :]
                flux_i = np.exp(-tau_i)
                print(f"\nSightline {i}:")
                print(f"  tau:  min={tau_i.min():.3f}, max={
                      tau_i.max():.3f}, mean={tau_i.mean():.3f}")
                print(f"  flux: min={flux_i.min():.3f}, max={
                      flux_i.max():.3f}, mean={flux_i.mean():.3f}")

    return 0
