import os
import re
import numpy as np
import h5py
import matplotlib.pyplot as plt

import scripts.config as config
import scripts.utils as utils
from scripts.analysis import (
    compute_flux_statistics,
    compute_effective_optical_depth,
    compute_power_spectrum,
    compute_column_density_distribution,
    compute_line_width_distribution,
    compute_temperature_density_relation,
    compute_metal_line_statistics,
    format_stats_table,
)
from scripts.plotting import (
    setup_plot_style,
    create_sample_spectra_plot,
    plot_flux_power_spectrum,
    plot_column_density_distribution,
    plot_line_width_distribution,
    plot_temperature_density_relation,
    plot_multi_line_comparison,
)


def cmd_analyze(args):
    spectra_file = args.spectra_file

    if not os.path.exists(spectra_file):
        print(f"Error: File not found: {spectra_file}")
        return 1

    print("=" * 70)
    print("ANALYZING LYMAN-ALPHA SPECTRA")
    print("=" * 70)
    print(f"Input file: {spectra_file}")

    # Load spectra data
    print("\n[1/5] Loading spectra data...")

    line_to_analyze = args.line if hasattr(args, 'line') else None

    with h5py.File(spectra_file, 'r') as f:
        tau = None
        tau_path = None

        # Try to find tau data
        # Option 1: User specified line
        if line_to_analyze:
            line_info = config.get_line_info(line_to_analyze)
            if line_info is None:
                print(f"Error: Unknown line '{line_to_analyze}'")
                valid_lines = ', '.join(config.SPECTRAL_LINES.keys())
                print(f"Valid lines: {valid_lines}")
                return 1

            elem, ion, wave, name = line_info
            tau_path = f'tau/{elem}/{ion}/{wave}'

            if tau_path in f:
                tau = f[tau_path][:]
                print(f"Loading {name} ({elem} {ion} {wave}Å)")
            else:
                print(f"Error: {name} not found in file at {tau_path}")
                return 1

        # Option 2: Auto-detect (try common lines)
        else:
            # Try new format: tau/H/1/1215 (Lyman-alpha)
            if 'tau/H/1/1215' in f:
                tau = f['tau/H/1/1215'][:]
                tau_path = 'tau/H/1/1215'
                print("Auto-detected: Lyman-alpha (H I 1215Å)")

            # Try old format: direct tau dataset
            elif 'tau' in f and isinstance(f['tau'], h5py.Dataset):
                tau = f['tau'][:]
                tau_path = 'tau'
                print("Old format detected: tau dataset")

            # Search for any tau dataset in new format
            elif 'tau' in f and isinstance(f['tau'], h5py.Group):
                # Find first available tau dataset
                def find_first_tau(group):
                    for key in group.keys():
                        item = group[key]
                        if isinstance(item, h5py.Dataset):
                            return item
                        elif isinstance(item, h5py.Group):
                            result = find_first_tau(item)
                            if result is not None:
                                return result
                    return None

                tau_dataset = find_first_tau(f['tau'])
                if tau_dataset is not None:
                    tau = tau_dataset[:]
                    tau_path = tau_dataset.name
                    print(f"  Auto-detected: {tau_path}")
                else:
                    print("Error: No tau datasets found in file")
                    return 1
            else:
                print("Error: Cannot find tau data in file")
                print("Use the 'explore' command to inspect file structure")
                return 1

        # Always compute flux from tau
        flux = np.exp(-tau)

        # Load metadata if available
        redshift = None
        if 'Header' in f:
            header = f['Header'].attrs
            # Try both 'redshift' and 'Redshift'
            redshift = header.get('redshift', header.get('Redshift', None))

    n_sightlines, n_pixels = tau.shape
    print(f"Sightlines: {n_sightlines}")
    print(f"Pixels: {n_pixels}")
    if redshift:
        print(f"Redshift: z = {redshift:.3f}")

    # ========== COMPREHENSIVE ANALYSIS ==========
    print("\n" + "=" * 70)
    print("COMPREHENSIVE ANALYSIS")
    print("=" * 70)

    # [1/8] Basic flux statistics
    print("\n[1/8] Computing basic flux statistics...")
    stats = compute_flux_statistics(tau)
    print(format_stats_table(stats))

    # [2/8] Effective optical depth
    print("\n[2/8] Computing effective optical depth tau_eff...")
    tau_eff_dict = compute_effective_optical_depth(tau)
    print(f"tau_eff = {tau_eff_dict['tau_eff']:.4f} ± {
          tau_eff_dict['tau_eff_err']:.4f}")
    print(f"Mean transmitted flux <F> = {tau_eff_dict['mean_flux']:.4f}")

    # Generate velocity and wavelength arrays
    # Velocity spacing from resolution (assume 0.1 km/s per pixel if not specified)
    n_sightlines, n_pixels = tau.shape
    velocity_spacing = 0.1  # km/s, assuming default resolution
    velocity = np.arange(n_pixels) * velocity_spacing

    # Create wavelength array for VPFIT (centered on Lyman-alpha at given redshift)
    lambda_lya = 1215.67  # Angstroms
    lambda_rest = lambda_lya * (1 + redshift) if redshift else lambda_lya
    wavelength = lambda_rest * (1 + velocity / 299792.458)  # Doppler shift

    # [3/8] Flux power spectrum
    print("\n[3/8] Computing flux power spectrum P_F(k)...")
    power_dict = compute_power_spectrum(flux, velocity_spacing)
    print(f"Computed power spectrum with {len(power_dict['k'])} k-modes")
    print(f"k range: {power_dict['k'][1]:.4f} to {
          power_dict['k'][-1]:.2f} s/km")
    print(f"Mean flux used: {power_dict['mean_flux']:.4f}")

    # [4/8] Column density distribution
    cd_method = getattr(args, 'cd_method', 'simple')
    print(
        f"\n[4/8] Computing column density distribution f(N_HI) using {cd_method} method...")

    if cd_method == 'simple':
        cddf_dict = compute_column_density_distribution(
            tau, velocity_spacing, threshold=0.5)
        print(f"Simple pixel optical depth method")
        print(f"Identified {cddf_dict['n_absorbers']} absorbers")
        if not np.isnan(cddf_dict['beta_fit']):
            print(f"Power law index β = {cddf_dict['beta_fit']:.2f}")
        else:
            print("Power law fit: insufficient data")

    elif cd_method == 'vpfit':
        cddf_dict = utils.compute_column_density_distribution_vpfit(
            flux, wavelength, redshift, threshold=0.05
        )
        if 'error' not in cddf_dict:
            print("VoigtFit method")
            print("Fitted {cddf_dict['n_absorbers']} absorbers")
            if cddf_dict['n_absorbers'] > 0:
                print(f"  Column density range: {
                      cddf_dict['N_HI'].min():.1e} - {cddf_dict['N_HI'].max():.1e} cm^-2")
                print(
                    f"  Mean b-parameter: {cddf_dict['b_params'].mean():.1f} km/s")
        else:
            print(f"  Error: {cddf_dict['error']}")
            print("  Falling back to simple method...")
            cddf_dict = compute_column_density_distribution(
                tau, velocity_spacing, threshold=0.5)

    elif cd_method == 'hybrid':
        cddf_dict = utils.compute_column_density_distribution_hybrid(
            tau, wavelength, redshift
        )
        print(f"Hybrid method (detection + fitting)")
        print(f"Identified {cddf_dict.get('n_absorbers', 0)} absorbers")
        if 'simple_backup' in cddf_dict:
            print(f"VPFIT refined {cddf_dict['n_absorbers']} absorbers")

    else:
        print(f"Unknown method '{cd_method}', using simple")
        cddf_dict = compute_column_density_distribution(
            tau, velocity_spacing, threshold=0.5)

    # [4b/8] Line width distribution (b-parameter analysis)
    print("\n[4b/8] Computing line width distribution b(N_HI)...")
    try:
        lwd_dict = compute_line_width_distribution(
            tau, velocity_spacing, threshold=0.5)
        print(f"Identified {
              lwd_dict['n_absorbers']} absorbers with b-parameters")
        if lwd_dict['n_absorbers'] > 0:
            print(f"Median b-parameter: {lwd_dict['b_median']:.1f} km/s")
            print(
                f"Mean b-parameter: {lwd_dict['b_mean']:.1f} ± {lwd_dict['b_std']:.1f} km/s")
            T_mean = 1.28e4 * lwd_dict['b_mean']**2
            print(f"Implied temperature: {T_mean/1e3:.0f} × 10³ K")
        else:
            print("Warning: No absorbers found for line width analysis")
            lwd_dict = None
    except Exception as e:
        print(f"Warning: Line width analysis failed: {e}")
        lwd_dict = None

    # [4c/8] Temperature-density relation (if data available)
    print("\n[4c/8] Checking for temperature-density data...")
    tdens_dict = None
    try:
        with h5py.File(spectra_file, 'r') as f:
            # Get element and ion from tau_path (e.g., 'tau/H/1/1215')
            if '/' in tau_path:
                parts = tau_path.split('/')
                if len(parts) >= 3:
                    temp_elem = parts[1]  # 'H'
                    temp_ion = parts[2]   # '1'
                else:
                    temp_elem = 'H'
                    temp_ion = '1'
            else:
                temp_elem = 'H'
                temp_ion = '1'

            # Check if temperature and density data exist
            has_temp = ('temperature' in f and
                        temp_elem in f['temperature'] and
                        temp_ion in f['temperature'][temp_elem])
            has_dens = ('density_weight_density' in f and
                        temp_elem in f['density_weight_density'] and
                        temp_ion in f['density_weight_density'][temp_elem])

            if has_temp and has_dens:
                print("Found temperature and density data - computing T-ρ relation...")
                temperature = f['temperature'][temp_elem][temp_ion][:]
                density = f['density_weight_density'][temp_elem][temp_ion][:]

                tdens_dict = compute_temperature_density_relation(
                    temperature, density, tau, min_tau=0.1
                )

                print(f"Valid pixels: {tdens_dict['n_pixels']:,}")
                if np.isfinite(tdens_dict['T0']):
                    print(f"T_0 (at mean density): {tdens_dict['T0']:.0f} K")
                    print(f"gamma (polytropic index): {tdens_dict['gamma']:.3f} ± {
                          tdens_dict['gamma_err']:.3f}")
                else:
                    print(f"Warning: T-ρ fit failed")
            else:
                print("Temperature/density data not available")
                print("(Regenerate spectra to include T-ρ analysis)")
    except Exception as e:
        print(f"Warning: Could not load temperature-density data: {e}")
        tdens_dict = None

    # [4d/8] Multi-line analysis (if multiple lines available)
    print("\n[4d/8] Checking for multi-line data...")
    metal_line_stats = []
    try:
        with h5py.File(spectra_file, 'r') as f:
            # Scan for all available tau data
            available_lines = []
            if 'tau' in f and isinstance(f['tau'], h5py.Group):
                for elem in f['tau'].keys():
                    for ion in f['tau'][elem].keys():
                        for wave in f['tau'][elem][ion].keys():
                            tau_group_path = f'tau/{elem}/{ion}/{wave}'
                            available_lines.append(
                                (tau_group_path, elem, ion, wave))

            if len(available_lines) > 1:
                print(f"Found {
                      len(available_lines)} spectral lines - performing multi-line analysis...")

                for tau_group_path, elem, ion, wave in available_lines:
                    # Load tau for this line
                    line_tau = np.array(f[tau_group_path])

                    # Create descriptive ion name
                    # Map element symbols to common names
                    elem_map = {'H': 'HI', 'C': 'CIV',
                                'O': 'OVI', 'Mg': 'MgII', 'Si': 'SiIV'}
                    if elem in elem_map:
                        ion_name = f"{elem_map[elem]} {wave}Å"
                    else:
                        ion_name = f"{elem}{ion}+ {wave}Å"

                    # Use lower threshold for metal lines
                    threshold = 0.5 if elem == 'H' else 0.05

                    print(f"Analyzing {
                          ion_name} (threshold={threshold})...")
                    stats = compute_metal_line_statistics(
                        line_tau,
                        velocity_spacing=velocity_spacing,
                        ion_name=ion_name,
                        threshold=threshold
                    )
                    metal_line_stats.append(stats)

                    print(f"Absorbers: {stats['n_absorbers']}, dN/dz: {stats['dN_dz']:.2f}, "
                          f"covering: {stats['covering_fraction']*100:.1f}%")

                print(f"Multi-line analysis complete")
            else:
                print("Only single line available - skipping multi-line analysis")
    except Exception as e:
        print(f"Warning: Multi-line analysis failed: {e}")
        metal_line_stats = []

    # Setup plotting
    print("\n[5/8] Setting up plots...")
    setup_plot_style()

    # [6/8] Create comprehensive plots
    print("\n[6/8] Creating comprehensive analysis plots...")

    # 6a. Sample spectra
    plot_file = config.get_plot_output_name(spectra_file, 'sample_spectra')
    create_sample_spectra_plot(
        velocity=velocity,
        flux=flux,
        redshift=redshift,
        n_samples=min(5, n_sightlines),
        output_path=plot_file
    )
    print(f"[a] Sample spectra: {plot_file}")

    # 6b. Flux power spectrum
    plot_file = config.get_plot_output_name(spectra_file, 'power_spectrum')
    plot_flux_power_spectrum(power_dict, redshift, plot_file)
    print(f"[b] Power spectrum: {plot_file}")

    # 6c. Column density distribution
    plot_file = config.get_plot_output_name(spectra_file, 'cddf')
    plot_column_density_distribution(cddf_dict, redshift, plot_file)
    print(f"[c] CDDF: {plot_file}")

    # 6d. Line width distribution (if available)
    if lwd_dict is not None and lwd_dict['n_absorbers'] > 0:
        plot_file = config.get_plot_output_name(spectra_file, 'line_widths')
        plot_line_width_distribution(lwd_dict, redshift, plot_file)
        print(f"[d] Line widths: {plot_file}")

    # 6e. Temperature-density relation (if available)
    if tdens_dict is not None and tdens_dict['n_pixels'] >= 100:
        plot_file = config.get_plot_output_name(spectra_file, 'temp_density')
        plot_temperature_density_relation(
            tdens_dict, redshift, plot_file)
        print(f"[e] T-ρ relation: {plot_file}")

    # 6f. Multi-line comparison (if available)
    if len(metal_line_stats) > 1:
        plot_file = config.get_plot_output_name(
            spectra_file, 'multi_line_comparison')
        plot_multi_line_comparison(metal_line_stats, redshift, plot_file)
        print(f"[f] Multi-line comparison: {plot_file}")

    # [7/8] Create detailed statistics plots
    print("\n[7/8] Creating detailed statistics plots...")

    try:
        import matplotlib.pyplot as plt

        # Flux distribution
        fig, axes = plt.subplots(2, 2, figsize=config.FIGSIZE_QUAD)

        # Subsample large arrays for faster plotting
        max_samples = 100000
        if tau.size > max_samples:
            sample_indices = np.random.choice(
                tau.size, max_samples, replace=False)
            flux_sample = flux.flatten()[sample_indices]
            tau_sample = tau.flatten()[sample_indices]
        else:
            flux_sample = flux.flatten()
            tau_sample = tau.flatten()

        # Panel 1: Flux histogram
        ax = axes[0, 0]
        ax.hist(flux_sample, bins=50, density=True, alpha=0.7,
                color='steelblue', edgecolor='black')
        ax.axvline(stats['mean_flux'], color='red', linestyle='--',
                   label=f"Mean = {stats['mean_flux']:.3f}")
        ax.axvline(stats['median_flux'], color='orange', linestyle='--',
                   label=f"Median = {stats['median_flux']:.3f}")
        ax.set_xlabel('Flux')
        ax.set_ylabel('Probability Density')
        ax.set_title('Flux Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel 2: Optical depth histogram
        ax = axes[0, 1]
        # Clip tau for visualization (very large values can cause issues)
        tau_clipped = np.clip(tau_sample, 0, 10)
        ax.hist(tau_clipped, bins=50, density=True,
                alpha=0.7, color='coral', edgecolor='black')
        ax.axvline(min(stats['mean_tau'], 10), color='red', linestyle='--',
                   label=f"Mean = {stats['mean_tau']:.3f}")
        ax.set_xlabel(r'Optical Depth $\tau$ (clipped at 10)')
        ax.set_ylabel('Probability Density')
        ax.set_title('Optical Depth Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel 3: Mean flux per sightline
        ax = axes[1, 0]
        mean_flux_per_los = flux.mean(axis=1)
        ax.plot(mean_flux_per_los, marker='o',
                linestyle='-', markersize=3, alpha=0.6)
        ax.axhline(stats['mean_flux'], color='red',
                   linestyle='--', label='Overall mean')
        ax.set_xlabel('Sightline Index')
        ax.set_ylabel('Mean Flux')
        ax.set_title('Mean Flux per Sightline')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel 4: Transmission statistics
        ax = axes[1, 1]
        transmitted_frac = (flux > 0.1).sum(axis=1) / \
            n_pixels  # Fraction with F > 0.1
        saturated_frac = (tau > 5.0).sum(axis=1) / \
            n_pixels     # Fraction with tau > 5

        ax.scatter(transmitted_frac, saturated_frac, alpha=0.5, s=20)
        ax.set_xlabel('Fraction with F > 0.1')
        ax.set_ylabel('Fraction with tau > 5 (saturated)')
        ax.set_title('Transmission vs Saturation')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        stats_file = config.get_plot_output_name(spectra_file, 'statistics')
        plt.savefig(stats_file, dpi=config.PLOT_DPI, bbox_inches='tight')
        plt.close()
        print(f"  [d] Statistics: {stats_file}")

    except Exception as e:
        print(f"  Warning: Could not create detailed statistics plot: {e}")

    # [8/8] Create diagnostic plots with particle data
    print("\n[8] Creating diagnostic plots from snapshot")
    print("-" * 70)
    # Try to load the original snapshot file to create diagnostic plots
    try:
        # Extract the snapshot filename from the spectra filename
        # Typical spectra file: camel_lya_spectra_snap_080.hdf5
        # We need to find: snap_080.hdf5
        import re
        match = re.search(r'snap[_-](\d+)', os.path.basename(spectra_file))
        if match:
            snap_num = match.group(1)
            # Look for snapshot file in common locations
            snapshot_dir = os.path.dirname(spectra_file)
            possible_snapshot_paths = [
                os.path.join(snapshot_dir, f'snap_{snap_num}.hdf5'),
                os.path.join(snapshot_dir, f'snap-{snap_num}.hdf5'),
                os.path.join(config.DATA_DIR, f'snap_{snap_num}.hdf5'),
            ]

            snapshot_filepath = None
            for path in possible_snapshot_paths:
                if os.path.exists(path):
                    snapshot_filepath = path
                    break

            if snapshot_filepath:
                print(f"Found snapshot: {snapshot_filepath}")

                with h5py.File(snapshot_filepath, 'r') as f:
                    # Load positions (subsample for speed)
                    stride = 100  # Use every 100th particle
                    coords = f['PartType0/Coordinates'][::stride]  # Shape: (N/100, 3)
                    nH = f['PartType0/NeutralHydrogenAbundance'][::stride]
                    print(f"Loaded {coords.shape[0]:,} particles (every {stride}th)")
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
                    # Save to appropriate subfolder
                    diagnostic_file = config.get_plot_output_name(spectra_file, 'snapshot_diagnostic')
                    plt.savefig(diagnostic_file, dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f"Saved plot to {diagnostic_file}")
            else:
                print(f"Snapshot file not found (tried snap_{snap_num}.hdf5)")
                print("Skipping diagnostic plots")
        else:
            print("Could not determine snapshot number from filename")
            print("Skipping diagnostic plots")
    except Exception as e:
        print(f"Warning: Could not create diagnostic plots: {e}")

    # ========== SUMMARY ==========
    print(f"\n{'=' * 70}")
    print("COMPREHENSIVE ANALYSIS COMPLETE")
    print(f"{'=' * 70}")
    print(f"\nKey Results:")
    print(f"Mean flux <F>:           {tau_eff_dict['mean_flux']:.4f}")
    print(f"Effective tau tau_eff:     {tau_eff_dict['tau_eff']:.4f} ± {
          tau_eff_dict['tau_eff_err']:.4f}")
    print(f"Number of absorbers:     {cddf_dict['n_absorbers']}")
    if not np.isnan(cddf_dict['beta_fit']):
        print(f"CDDF power law β:        {cddf_dict['beta_fit']:.2f}")

    if lwd_dict is not None and lwd_dict['n_absorbers'] > 0:
        print(
            f"Mean b-parameter:        {lwd_dict['b_mean']:.1f} ± {lwd_dict['b_std']:.1f} km/s")
        T_mean = 1.28e4 * lwd_dict['b_mean']**2
        print(f"Implied temperature:     {T_mean/1e3:.0f} × 10³ K")

    if tdens_dict is not None and np.isfinite(tdens_dict['T0']):
        print(f"T_0 (at mean density):    {tdens_dict['T0']:.0f} K")
        print(f"gamma (polytropic index):    {tdens_dict['gamma']:.3f}")

    if len(metal_line_stats) > 1:
        print(
            f"Multi-line analysis:     {len(metal_line_stats)} lines detected")
        for stats in metal_line_stats:
            print(f"{stats['ion_name']:15s}: {stats['n_absorbers']:4d} absorbers, "
                f"dN/dz={stats['dN_dz']:5.1f}, covering={stats['covering_fraction']*100:4.1f}%")

    print(f"\nPlots saved to: {config.PLOTS_DIR}")
    print("- Sample spectra")
    print("- Flux power spectrum P_F(k)")
    print("- Column density distribution f(N_HI)")
    if lwd_dict is not None and lwd_dict['n_absorbers'] > 0:
        print("- Line width distribution b(N_HI)")
    if tdens_dict is not None and tdens_dict['n_pixels'] >= 100:
        print("- Temperature-density relation T(ρ)")
    if len(metal_line_stats) > 1:
        print("- Multi-line comparison")
    print("- Detailed statistics")
    print(f"{'=' * 70}")

    return 0
