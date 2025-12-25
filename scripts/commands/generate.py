import os
import time
import numpy as np

import scripts.config as config
from scripts.hdf5_io import load_snapshot_metadata
from scripts.fake_spectra_fix import apply_fake_spectra_bugfixes


# Generate Lyman-alpha (or other) spectra from simulation snapshot
def cmd_generate(args):
    snapshot_file = args.snapshot
    num_sightlines = args.sightlines
    resolution = args.res
    random_seed = args.seed
    output_file = args.output
    line_arg = args.line if hasattr(args, 'line') else 'lya'

    # Parse line specification
    lines_to_compute = config.parse_line_list(line_arg)
    if lines_to_compute is None:
        valid_lines = ', '.join(config.SPECTRAL_LINES.keys())
        print(f"Error: Invalid line specification '{line_arg}'")
        print(f"Valid lines: {valid_lines}")
        return 1

    # Check if file exists
    if not os.path.exists(snapshot_file):
        print(f"Error: File not found: {snapshot_file}")
        print("\nUse 'python analyze_spectra.py list' to see available data")
        return 1

    # Extract snapshot directory and number
    snapshot_dir = os.path.dirname(snapshot_file)
    snapshot_name = os.path.basename(snapshot_file).replace('.hdf5', '')

    try:
        snapshot_num = int(snapshot_name.replace('snap_', ''))
    except ValueError:
        print(f"Error: Could not extract snapshot number from '{
              snapshot_name}'")
        print("Expected format: snap_NNN.hdf5")
        return 1

    print("=" * 70)
    print("GENERATING SPECTRA FROM SIMULATION SNAPSHOT")
    print("=" * 70)
    print(f"Snapshot:    {snapshot_file}")
    print(f"Snap number: {snapshot_num}")
    print(f"Sightlines:  {num_sightlines}")
    print(f"Resolution:  {resolution} km/s/pixel")
    print(f"Random seed: {random_seed}")

    # Display lines to compute
    print("Lines:\t\t", end='')
    for line_code in lines_to_compute:
        elem, ion, wave, name = config.get_line_info(line_code)
        print(f"{name} ({elem}{ion} {wave}Å)", end='  ')
    print()

    # Load snapshot metadata
    print("\n[1/5] Loading snapshot metadata...")
    metadata = load_snapshot_metadata(snapshot_file)

    print(f"Redshift:      z = {metadata['redshift']:.3f}")
    print(f"Box size:      {metadata['boxsize_proper']:.2f} Mpc/h")
    print(f"Gas particles: {metadata['num_gas']:,}")

    # Generate random sightlines
    print(f"\n[2/5] Generating {num_sightlines} random sightlines...")
    np.random.seed(random_seed)

    boxsize = metadata['boxsize']  # comoving kpc/h
    cofm = np.random.uniform(0, boxsize, size=(num_sightlines, 3))
    axis = np.random.randint(1, 4, size=num_sightlines)

    print("Positions: random uniform across box")
    print("Axes: random (x, y, or z)")

    # Apply fake_spectra bugfixes
    print("\n[3/5] Initializing fake_spectra (applying Python 3.13 bugfixes)...")
    apply_fake_spectra_bugfixes()

    # Import after bugfixes applied
    from fake_spectra import spectra

    # Determine output file path
    if output_file:
        savefile = os.path.abspath(output_file)
    else:
        savefile = config.get_snapshot_output_name(
            snapshot_file,
            lines=lines_to_compute,
            num_sightlines=num_sightlines
        )
        # Save in same directory as snapshot - use absolute path
        if not os.path.isabs(savefile):
            savefile = os.path.abspath(os.path.join(
                snapshot_dir, savefile) if snapshot_dir else savefile)

    print(f"Output will be saved to: {savefile}")

    # Create output directory if needed
    output_dir = os.path.dirname(savefile)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Generate spectra
    print("\n[4/5] Initializing spectra object...")

    try:
        spec = spectra.Spectra(
            num=snapshot_num,                    # Snapshot number (e.g., 86)
            base=snapshot_dir or '.',            # Directory containing snapshot
            savefile=savefile,
            savedir='',                          # Don't append subdirectory to savefile
            res=resolution,
            reload_file=True,                    # Set to True to generate new spectra
            # Sightline positions (determines count)
            cofm=cofm,
            axis=axis,
            load_halo=False,
            quiet=False
        )
        print(f"Loaded snapshot {snapshot_num}")
        print(f"Initialized {cofm.shape[0]} sightlines")
        print(f"Velocity range: {spec.vmax:.2f} km/s")
        print(f"Resolution: {spec.dvbin:.4f} km/s/pixel")

    except Exception as e:
        print(f"\nError initializing spectra object: {e}")
        print("\nTroubleshooting:")
        print("- Check that snapshot file is valid HDF5")
        print("- Try reducing -n (number of sightlines)")
        print("- Ensure sufficient memory available")
        return 1

    # Compute optical depths for each line
    print("\n[5/5] Computing optical depths...")
    start_time = time.time()

    for i, line_code in enumerate(lines_to_compute):
        elem, ion, wave, name = config.get_line_info(line_code)
        print(f"[{i+1}/{len(lines_to_compute)
                          }] {name} ({elem} {ion} {wave}Å)...", end=' ', flush=True)

        line_start = time.time()

        try:
            spec.get_tau(elem, ion, wave)

            tau = spec.tau.get((elem, ion, wave))
            if tau is None:
                print(f"FAILED (tau not computed)")
                return 1

            line_elapsed = time.time() - line_start
            print(f"OK ({line_elapsed:.1f}s, shape={tau.shape})")

        except MemoryError:
            print(f"\nError: Out of memory computing {name}")
            return 1

        except Exception as e:
            print(f"\nError computing {name}: {e}")
            return 1

    total_elapsed = time.time() - start_time
    print(f"  Total computation time: {total_elapsed:.1f}s")

    # Compute temperature and density for first line (needed for T-ρ analysis)
    print("\nComputing temperature and density along sightlines...")
    first_line_code = lines_to_compute[0]
    elem, ion, wave, name = config.get_line_info(first_line_code)

    try:
        print(f"Computing temperature for {
              elem} {ion}...", end=' ', flush=True)
        spec.get_temp(elem, ion)

        print(
            f"Computing density-weighted density for {elem} {ion}...", end=' ', flush=True)
        spec.get_dens_weighted_density(elem, ion)
    except Exception as e:
        print(f"\nWarning: Could not compute temperature/density: {e}")
        print("(Temperature-density analysis will not be available)")

    # Save to HDF5
    print("\nSaving to HDF5...")
    try:
        spec.save_file()
        print(f"Saved to: {spec.savefile}")
    except Exception as e:
        print(f"Error saving file: {e}")
        return 1

    output_path = spec.savefile
    print(f"\n{'=' * 70}")
    print(f"Spectra saved to:")
    print(f"  {output_path}")
    print(f"{'=' * 70}")

    return 0
