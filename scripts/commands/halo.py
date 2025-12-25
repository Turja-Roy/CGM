import os
import numpy as np
from pathlib import Path

import scripts.config as config
from scripts.cgm import (
    load_subfind_catalog,
    filter_halos_by_mass,
    filter_isolated_halos,
    plot_halo_projection,
    plot_temperature_slices,
    plot_radial_profiles,
    plot_halo_summary,
)


# Analyze individual galaxy halos from simulation snapshot.
def cmd_halo(args):
    snapshot_file = args.snapshot
    mass_range = args.mass_range
    halo_id_target = args.halo_id
    n_halos = args.n_halos
    isolated_only = args.isolated_only
    output_dir = args.output_dir
    slice_thickness = args.slice_thickness
    plot_type = args.plot_type
    
    # Check if snapshot exists
    if not os.path.exists(snapshot_file):
        print(f"Error: Snapshot file not found: {snapshot_file}")
        print("\nUse 'python analyze_spectra.py list' to see available data")
        return 1
    
    print("=" * 70)
    print("HALO ANALYSIS")
    print("=" * 70)
    print(f"Snapshot:       {snapshot_file}")
    print(f"Mass range:     [{mass_range[0]:.1f}, {mass_range[1]:.1f}] log M_sun")
    print(f"Isolated only:  {isolated_only}")
    print(f"Plot type:      {plot_type}")
    
    # Setup output directory
    if output_dir is None:
        snapshot_path = Path(snapshot_file)
        # Extract simulation name from path
        if len(snapshot_path.parts) >= 2:
            sim_name = snapshot_path.parts[-2]
        else:
            sim_name = 'unknown'
        output_dir = config.PLOTS_DIR / sim_name / 'halos'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir:     {output_dir}")
    
    # Load halo catalog
    print("\n[1/4] Loading halo catalog...")
    try:
        catalog = load_subfind_catalog(snapshot_file)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nHalo catalogs not found.")
        return 1
    except Exception as e:
        print(f"Error loading halo catalog: {e}")
        return 1
    
    # Filter halos
    print("\n[2/4] Filtering halos...")
    filtered_catalog = filter_halos_by_mass(catalog, mass_range)
    
    if isolated_only:
        filtered_catalog = filter_isolated_halos(filtered_catalog)
    
    if len(filtered_catalog) == 0:
        print("Error: No halos match the selection criteria")
        return 1
    
    # Select halos to analyze
    if halo_id_target is not None:
        selected = filtered_catalog[filtered_catalog['halo_id'] == halo_id_target]
        if len(selected) == 0:
            print(f"Error: Halo ID {halo_id_target} not found in filtered catalog")
            return 1
        print(f"Analyzing specific halo: {halo_id_target}")
    else:
        # Analyze top N massive halos
        sorted_catalog = filtered_catalog.sort_values('mass_total', ascending=False)
        selected = sorted_catalog.head(n_halos)
        print(f"Analyzing top {len(selected)} massive halos")
    
    # Display selected halos
    print("\nSelected halos:")
    print(f"{'ID':>8} {'Mass (M_sun)':>15} {'Rvir (ckpc/h)':>15} {'Redshift':>10}")
    print("-" * 70)
    for _, halo in selected.iterrows():
        print(f"{int(halo['halo_id']):8d} {halo['mass_total']:15.2e} "
              f"{halo['radius_vir']:15.1f} {halo['redshift']:10.3f}")
    
    # Generate plots for each halo
    print(f"\n[3/4] Generating {plot_type} plots...")
    
    n_success = 0
    n_failed = 0
    
    for idx, halo in selected.iterrows():
        halo_id = int(halo['halo_id'])
        print(f"\nProcessing halo {halo_id} ({n_success + n_failed + 1}/{len(selected)})...")
        
        # Determine output filename
        snap_num = Path(snapshot_file).stem.split('_')[-1]
        output_file = output_dir / f"halo_{halo_id:06d}_{plot_type}_snap_{snap_num}.png"
        
        try:
            # Generate requested plot type
            if plot_type == 'projection':
                result = plot_halo_projection(
                    snapshot_file, halo, str(output_file),
                    slice_thickness=slice_thickness,
                    properties=['density', 'temperature'],
                    show_virial_circle=True,
                    axes_to_plot=['xy']
                )
            elif plot_type == 'temperature':
                result = plot_temperature_slices(
                    snapshot_file, halo, str(output_file),
                    temp_bins=[(3.5, 4), (4, 4.5), (4.5, 7)],
                    slice_thickness=slice_thickness
                )
            elif plot_type == 'radial':
                result = plot_radial_profiles(
                    snapshot_file, halo, str(output_file),
                    properties=['density', 'temperature', 'neutral_fraction']
                )
            elif plot_type == 'summary':
                result = plot_halo_summary(
                    snapshot_file, halo, str(output_file)
                )
            else:
                print(f"Unknown plot type: {plot_type}")
                result = 1
            
            if result == 0:
                n_success += 1
            else:
                n_failed += 1
                print(f"Warning: Plot generation returned error code {result}")
                
        except Exception as e:
            print(f"Error generating plot: {e}")
            n_failed += 1
            continue
    
    # Summary
    print("\n[4/4] Summary")
    print("=" * 70)
    print(f"Successfully processed: {n_success}/{len(selected)} halos")
    if n_failed > 0:
        print(f"Failed: {n_failed} halos")
    print(f"Plots saved to: {output_dir}")
    print("=" * 70)
    
    return 0 if n_failed == 0 else 1
