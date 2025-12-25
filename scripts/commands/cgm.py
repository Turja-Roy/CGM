import os
import time
import numpy as np
from pathlib import Path

import scripts.config as config
from scripts.hdf5_io import load_snapshot_metadata
from scripts.fake_spectra_fix import apply_fake_spectra_bugfixes
from scripts.cgm import (
    load_subfind_catalog,
    filter_halos_by_mass,
    filter_isolated_halos,
    generate_cgm_sightlines,
    save_cgm_metadata,
)


# 1. Loads halo catalog and filters by mass/isolation
# 2. Generates sightline positions at specified impact parameters
# 3. Runs fake_spectra to generate synthetic spectra
# 4. Saves CGM metadata to output HDF5 file
def cmd_cgm(args):
    snapshot_file = args.snapshot
    mass_range = args.mass_range
    n_halos = args.n_halos
    impact_params_str = args.impact_params
    n_per_bin = args.n_per_bin
    azimuthal = args.azimuthal
    line_arg = args.line
    output_file = args.output
    isolated_only = args.isolated_only
    
    # Parse impact parameters
    try:
        impact_params = [float(x.strip()) for x in impact_params_str.split(',')]
    except ValueError:
        print(f"Error: Invalid impact parameters: {impact_params_str}")
        print("Expected format: comma-separated floats (e.g., '0.5,1.0,1.5')")
        return 1
    
    # Parse spectral lines
    lines_to_compute = config.parse_line_list(line_arg)
    if lines_to_compute is None:
        valid_lines = ', '.join(config.SPECTRAL_LINES.keys())
        print(f"Error: Invalid line specification '{line_arg}'")
        print(f"Valid lines: {valid_lines}")
        return 1
    
    # Check snapshot exists
    if not os.path.exists(snapshot_file):
        print(f"Error: Snapshot file not found: {snapshot_file}")
        return 1
    
    print("=" * 70)
    print("CGM-TARGETED SPECTRA GENERATION")
    print("=" * 70)
    print(f"Snapshot:        {snapshot_file}")
    print(f"Mass range:      [{mass_range[0]:.1f}, {mass_range[1]:.1f}] log M_sun")
    if n_halos is not None:
        print(f"Number of halos: {n_halos} (top most massive)")
    print(f"Impact params:   {impact_params} (R_vir)")
    print(f"Per bin:         {n_per_bin} sightlines")
    print(f"Azimuthal:       {azimuthal} samples")
    print(f"Isolated only:   {isolated_only}")
    
    # Display lines to compute
    print("Lines:           ", end='')
    for line_code in lines_to_compute:
        elem, ion, wave, name = config.get_line_info(line_code)
        print(f"{name} ({elem}{ion} {wave}Å)", end='  ')
    print()
    
    # Load snapshot metadata
    print("\n[1/6] Loading snapshot metadata...")
    metadata = load_snapshot_metadata(snapshot_file)
    print(f"Redshift:        z = {metadata['redshift']:.3f}")
    print(f"Box size:        {metadata['boxsize_proper']:.2f} Mpc (proper)")
    print(f"Gas particles:   {metadata['num_gas']:,}")
    
    # Load halo catalog
    print("\n[2/6] Loading halo catalog...")
    try:
        catalog = load_subfind_catalog(snapshot_file)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nHalo catalogs not found. You need to download group catalogs.")
        return 1
    except Exception as e:
        print(f"Error loading halo catalog: {e}")
        return 1
    
    # Filter halos
    print("\n[3/6] Filtering halos...")
    filtered_catalog = filter_halos_by_mass(catalog, mass_range)
    
    if isolated_only:
        filtered_catalog = filter_isolated_halos(filtered_catalog)
    
    if len(filtered_catalog) == 0:
        print("Error: No halos match the selection criteria")
        return 1
    
    # Limit number of halos if requested
    if n_halos is not None and n_halos < len(filtered_catalog):
        filtered_catalog = filtered_catalog.sort_values('mass_total', ascending=False).head(n_halos)
        print(f"\nLimited to top {n_halos} most massive halos")
    
    print(f"\nSelected {len(filtered_catalog)} halos for CGM analysis")
    
    # Generate CGM sightlines
    print("\n[4/6] Generating CGM sightline positions...")
    cgm_data = generate_cgm_sightlines(
        filtered_catalog,
        impact_params=impact_params,
        n_per_bin=n_per_bin,
        azimuthal_samples=azimuthal,
        axis_direction='z',
        radius_type='radius_vir'
    )
    
    cofm = cgm_data['cofm']
    axis = cgm_data['axis']
    n_sightlines = cgm_data['n_sightlines']
    
    print(f"\nGenerated {n_sightlines:,} sightline positions")
    
    # Determine output filename
    if output_file is None:
        snapshot_path = Path(snapshot_file)
        snap_num = snapshot_path.stem.split('_')[-1]
        line_str = '_'.join(lines_to_compute)
        output_file = str(config.OUTPUT_DIR / 
                         f"cgm_{line_str}_spectra_snap_{snap_num}_n{n_sightlines}.hdf5")
    
    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Output file:     {output_file}")
    
    # Apply fake_spectra bugfixes
    print("\n[5/6] Generating spectra with fake_spectra...")
    apply_fake_spectra_bugfixes()
    
    # Import fake_spectra (after bugfixes applied)
    try:
        from fake_spectra import spectra
    except ImportError as e:
        print(f"Error: Could not import fake_spectra: {e}")
        print("Please ensure fake_spectra is installed:")
        print("  pip install fake_spectra")
        return 1
    
    start_time = time.time()
    
    # Create Spectra object
    print(f"Creating spectra for {n_sightlines} sightlines...")
    
    # Extract snapshot number and base directory for fake_spectra
    snapshot_path = Path(snapshot_file)
    snap_num = int(snapshot_path.stem.split('_')[-1])  # Extract number from snap_080
    snap_base = str(snapshot_path.parent)  # Directory containing snapshot
    
    print(f"Snapshot directory: {snap_base}")
    print(f"Snapshot number: {snap_num}")
    
    spec = spectra.Spectra(
        num=snap_num,  # Snapshot number (NOT number of sightlines!)
        base=snap_base,  # Directory containing snapshot files
        cofm=cofm,
        axis=axis,
        res=config.DEFAULT_PIXEL_RESOLUTION,
        savefile=output_file,
        reload_file=True,  # Create new file or reload existing
    )
    
    print("Computing optical depths for requested lines...")
    
    # Compute optical depth for each line
    for line_code in lines_to_compute:
        elem, ion, wave, name = config.get_line_info(line_code)
        print(f"  Computing {name} ({elem} {ion}, {wave} Å)...")
        spec.get_tau(elem, ion, wave)
    
    # Extract temperature and density data for the first line
    first_line = lines_to_compute[0]
    elem, ion, wave, name = config.get_line_info(first_line)
    print(f"\nExtracting temperature and density data for {name}...")
    spec.get_temp(elem, ion)
    spec.get_density(elem, ion)
    spec.get_dens_weighted_density(elem, ion)
    
    # Save spectra
    print("Saving spectra to HDF5...")
    spec.save_file()
    
    elapsed = time.time() - start_time
    print(f"Spectra generation completed in {elapsed:.1f} seconds")
    
    # Save CGM metadata
    print("\n[6/6] Saving CGM metadata...")
    result = save_cgm_metadata(output_file, cgm_data, filtered_catalog)
    
    if result != 0:
        print("Warning: Failed to save CGM metadata")
        return 1
    
    # Summary
    print("\n" + "=" * 70)
    print("CGM SPECTRA GENERATION COMPLETE")
    print("=" * 70)
    print(f"Output file:     {output_file}")
    print(f"Sightlines:      {n_sightlines:,}")
    print(f"Halos:           {len(filtered_catalog)}")
    print(f"Impact params:   {impact_params} (R_vir)")
    print(f"Spectral lines:  {', '.join(lines_to_compute)}")
    print("=" * 70)
    
    return 0
