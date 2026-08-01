import os
import re
import numpy as np
import h5py
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

import scripts.config as config
from scripts.analysis import (
    compute_flux_statistics,
    compute_effective_optical_depth,
    compute_power_spectrum,
    compute_column_density_distribution,
    compute_line_width_distribution,
    compute_temperature_density_relation,
    compute_temperature_density_chunked,
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

from scripts.analysis import compute_column_density_distribution_vpfit


_ABSORBER_LABELS = {
    0: 'per-feature deblending (tau > threshold runs)',
    1: 'whole sightline',
    2: 'fixed 50 km/s cells (fake_spectra line=False)',
}


def _cddf_method_label():
    mode = config.CDDF_OPTIONS['absorber_mode']
    reduction = 'summed' if config.CDDF_OPTIONS['colden_mode'] == 1 else 'max'
    label = _ABSORBER_LABELS.get(mode, f'mode {mode}')
    if mode == 2:
        label = f"fixed {config.CDDF_OPTIONS['cell_dv']:.0f} km/s cells (fake_spectra line=False)"
    return f"Absorbers: {label}; N per absorber: {reduction} pixel colden"


def _format_dX(cddf_dict):
    """dx_mode = 1 returns the dimensionless absorption distance, not Mpc."""
    if config.CDDF_OPTIONS['dx_mode'] == 1:
        return f"X = {cddf_dict['dX']:.6f} (absorption distance, dimensionless)"
    return f"dX = {cddf_dict['dX']:.2f} Mpc (comoving)"


def _cddf_simple(tau, velocity_spacing, threshold, colden, redshift,
                 box_size_ckpc_h, hubble, omega_m):
    """Production CDDF: the C++ kernel under config.CDDF_OPTIONS.

    Module level, not a closure, so ProcessPoolExecutor can pickle it. Above
    config.CDDF_SATURATED_REDSHIFT the fitted slope is suppressed to NaN, with
    the raw value kept as beta_fit_raw so the suppression stays auditable.

    `threshold` is forwarded for signature compatibility; absorber_mode = 2
    ignores it.
    """
    result = compute_column_density_distribution(
        tau, velocity_spacing, threshold=threshold, colden=colden,
        redshift=redshift, box_size_ckpc_h=box_size_ckpc_h,
        hubble=hubble, omega_m=omega_m, **config.CDDF_OPTIONS)

    saturated = redshift is not None and redshift > config.CDDF_SATURATED_REDSHIFT
    result['saturated'] = saturated
    if saturated:
        result['beta_fit_raw'] = result['beta_fit']
        result['beta_fit'] = float('nan')
    return result


def cmd_analyze(args):
    spectra_file = args.spectra_file
    max_sightlines = args.max_sightlines if hasattr(args, 'max_sightlines') else None
    num_workers = args.workers if hasattr(args, 'workers') else 1

    if not os.path.exists(spectra_file):
        print(f"Error: File not found: {spectra_file}")
        return 1

    print("=" * 70)
    print("ANALYZING LYMAN-ALPHA SPECTRA")
    print("=" * 70)
    print(f"Input file: {spectra_file}")
    print(f"Workers: {num_workers}")

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
        
        # Try to load column density data (for improved N_HI calculations)
        colden = None
        try:
            # Determine colden path based on tau_path
            # tau_path format: 'tau/H/1/1215' -> colden path: 'colden/H/1'
            if tau_path.startswith('tau/'):
                parts = tau_path.split('/')
                if len(parts) >= 3:
                    colden_path = f'colden/{parts[1]}/{parts[2]}'
                    if colden_path in f:
                        colden = f[colden_path][:]
                        print(f"  Loaded column density data from {colden_path}")
                        print(f"  Using fake_spectra's pre-computed column densities for accuracy")
        except Exception as e:
            print(f"  Note: Could not load column density data: {e}")
            print(f"  Will use fallback tau-based method for N_HI calculation")

        # Load metadata if available
        redshift = None
        box_size_ckpc_h = None
        hubble = 0.6774  # Default for TNG/SIMBA
        omega_m = 0.3089  # Default for TNG/SIMBA
        
        if 'Header' in f:
            header = f['Header'].attrs
            # Try both 'redshift' and 'Redshift'
            redshift = header.get('redshift', header.get('Redshift', None))
            # Load box size (ckpc/h)
            box_size_ckpc_h = header.get('box', header.get('BoxSize', None))
            # Load cosmology parameters if available
            hubble = header.get('hubble', header.get('HubbleParam', 0.6774))
            omega_m = header.get('omegam', header.get('Omega0', 0.3089))

            velocity_spacing = None
            if 'dvbin' in header:
                velocity_spacing = float(header['dvbin'])

    n_sightlines, n_pixels = tau.shape
    
    # Subsample if requested to reduce memory usage
    if max_sightlines is not None and n_sightlines > max_sightlines:
        print(f"\nSubsampling to {max_sightlines} sightlines (original: {n_sightlines})")
        indices = np.random.choice(n_sightlines, max_sightlines, replace=False)
        indices.sort()  # Keep in order
        tau = tau[indices]
        flux = flux[indices]
        if colden is not None:
            colden = colden[indices]
        n_sightlines = max_sightlines
    
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
    # Load or compute velocity spacing from header
    n_sightlines, n_pixels = tau.shape
    
    # If velocity_spacing not set from header, try to load/compute it
    if velocity_spacing is None:
        try:
            if 'dvbin' in f['Header'].attrs:
                velocity_spacing = float(f['Header'].attrs['dvbin'])
                print(f"Loaded velocity spacing from header: {velocity_spacing:.4f} km/s/pixel")
            else:
                # Compute from header attributes (backward compatible)
                header = f['Header'].attrs
                nbins = header['nbins']
                box = header['box']  # ckpc/h
                Hz = header['Hz']    # km/s/Mpc
                hubble = header['hubble']  # h parameter
                
                # Compute vmax: convert box to cMpc/h, then to velocity
                vmax = (box / 1000.0) * Hz / hubble  # km/s
                
                # Compute velocity spacing
                velocity_spacing = 2.0 * vmax / nbins
                print(f"Computed velocity spacing from header: {velocity_spacing:.4f} km/s/pixel")
                print(f"  (vmax={vmax:.2f} km/s, nbins={nbins})")
        except Exception as e:
            print(f"Warning: Could not load/compute velocity spacing: {e}")
            print(f"Falling back to default: 0.1 km/s/pixel (may be inaccurate!)")
            velocity_spacing = 0.1  # km/s, fallback
    else:
        # velocity_spacing already set from header earlier
        print(f"Using velocity spacing from header: {velocity_spacing:.4f} km/s/pixel")
    
    velocity = np.arange(n_pixels) * velocity_spacing

    # Create wavelength array for VPFIT (centered on Lyman-alpha at given redshift)
    lambda_lya = 1215.67  # Angstroms
    lambda_rest = lambda_lya * (1 + redshift) if redshift else lambda_lya
    wavelength = lambda_rest * (1 + velocity / 299792.458)  # Doppler shift

    # [3/8] Flux power spectrum
    # [4/8] Column density distribution
    # [4b/8] Line width distribution
    skip_power = getattr(args, 'skip_power_spectrum', False)
    skip_cddf = getattr(args, 'skip_cddf', False)
    skip_lwd = getattr(args, 'skip_line_width', False)

    # Optical depth above which a pixel counts as absorbing. Overridable with
    # --tau-threshold; see config.TAU_THRESHOLD_HI for why this value is a
    # systematic and not a neutral default.
    tau_threshold = getattr(args, 'tau_threshold', None)
    if tau_threshold is None:
        tau_threshold = config.TAU_THRESHOLD_HI
    if tau_threshold != config.TAU_THRESHOLD_HI:
        print(f"\nUsing non-default HI absorber threshold: tau > {tau_threshold} "
              f"(default {config.TAU_THRESHOLD_HI})")

    # Run expensive computations in parallel when workers > 1
    if num_workers > 1:
        print(f"\n[3-5/8] Running expensive analyses in parallel with {num_workers} workers...")
        print("-" * 70)

        cd_method = getattr(args, 'cd_method', 'simple')

        # Prepare arguments for each function
        tasks = []
        if not skip_power:
            tasks.append(('power', compute_power_spectrum, flux, velocity_spacing))
        
        if not skip_cddf:
            if cd_method == 'simple':
                tasks.append(('cddf', _cddf_simple,
                            tau, velocity_spacing, tau_threshold, colden, redshift, box_size_ckpc_h, hubble, omega_m))
            elif cd_method == 'vpfit':
                tasks.append(('cddf_vpfit', compute_column_density_distribution_vpfit,
                            flux, wavelength, redshift, config.TAU_THRESHOLD_VPFIT))
            else:
                tasks.append(('cddf', _cddf_simple,
                            tau, velocity_spacing, tau_threshold, colden, redshift, box_size_ckpc_h, hubble, omega_m))

        if not skip_lwd:
            tasks.append(('lwd', compute_line_width_distribution,
                        tau, velocity_spacing, tau_threshold, colden))

        # Execute in parallel
        results = {}
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_task = {}
            for task_name, func, *args in tasks:
                future = executor.submit(func, *args)
                future_to_task[future] = task_name

            for future in as_completed(future_to_task):
                task_name = future_to_task[future]
                try:
                    results[task_name] = future.result()
                    print(f"  {task_name}: completed")
                except Exception as e:
                    print(f"  {task_name}: FAILED - {e}")
                    results[task_name] = None

        # Extract results
        power_dict = results.get('power')
        cddf_dict = results.get('cddf') or results.get('cddf_vpfit')
        lwd_dict = results.get('lwd')

        if power_dict:
            print(f"Computed power spectrum with {len(power_dict['k'])} k-modes")
            print(f"k range: {power_dict['k'][1]:.4f} to {power_dict['k'][-1]:.2f} s/km")
            print(f"Mean flux used: {power_dict['mean_flux']:.4f}")

        if cddf_dict and 'error' not in cddf_dict:
            if cd_method == 'simple':
                print(f"Identified {cddf_dict['n_absorbers']} absorbers")
                if redshift and box_size_ckpc_h:
                    print(f"Absorption path length: {_format_dX(cddf_dict)}")
                if not np.isnan(cddf_dict.get('beta_fit', np.nan)):
                    print(f"Power law index β = {cddf_dict['beta_fit']:.2f}")
                elif cddf_dict.get('saturated'):
                    print(f"Power law fit: SUPPRESSED (z = {redshift:.2f} > "
                          f"{config.CDDF_SATURATED_REDSHIFT}, forest saturated)")
            elif cd_method == 'vpfit':
                print(f"Fitted {cddf_dict['n_absorbers']} absorbers")
        else:
            print(f"Column density computation failed or not available")

        if lwd_dict and lwd_dict.get('n_absorbers', 0) > 0:
            print(f"Line widths: {lwd_dict['n_absorbers']} absorbers, median b={lwd_dict['b_median']:.1f} km/s")
    else:
        # Sequential execution (original code)
        power_dict = None
        cddf_dict = None
        lwd_dict = None

        if not skip_power:
            print("\n[3/8] Computing flux power spectrum P_F(k)...")
            power_dict = compute_power_spectrum(flux, velocity_spacing)
            print(f"Computed power spectrum with {len(power_dict['k'])} k-modes")
            print(f"k range: {power_dict['k'][1]:.4f} to {power_dict['k'][-1]:.2f} s/km")
            print(f"Mean flux used: {power_dict['mean_flux']:.4f}")
        else:
            print("\n[3/8] Skipping power spectrum (--skip-power-spectrum)")

        cd_method = getattr(args, 'cd_method', 'simple')
        if not skip_cddf:
            print(f"\n[4/8] Computing column density distribution f(N_HI) using {cd_method} method...")

            if cd_method == 'simple':
                cddf_dict = _cddf_simple(
                    tau, velocity_spacing, tau_threshold, colden,
                    redshift, box_size_ckpc_h, hubble, omega_m)
                print(f"{_cddf_method_label()}")
                print(f"Identified {cddf_dict['n_absorbers']} absorbers")
                if redshift and box_size_ckpc_h:
                    print(f"Absorption path length: {_format_dX(cddf_dict)}")
                if not np.isnan(cddf_dict.get('beta_fit', np.nan)):
                    print(f"Power law index β = {cddf_dict['beta_fit']:.2f}")
                elif cddf_dict.get('saturated'):
                    print(f"Power law fit: SUPPRESSED (z = {redshift:.2f} > "
                          f"{config.CDDF_SATURATED_REDSHIFT}, forest saturated; "
                          f"raw value would be {cddf_dict.get('beta_fit_raw', float('nan')):.2f})")
                else:
                    print("Power law fit: insufficient data")

            elif cd_method == 'vpfit':
                cddf_dict = compute_column_density_distribution_vpfit(
                    flux, wavelength, redshift, threshold=config.TAU_THRESHOLD_VPFIT
                )
                if 'error' not in cddf_dict:
                    print("VoigtFit method")
                    print(f"Fitted {cddf_dict['n_absorbers']} absorbers")
                    if cddf_dict['n_absorbers'] > 0:
                        print(f"  Column density range: {cddf_dict['N_HI'].min():.1e} - {cddf_dict['N_HI'].max():.1e} cm^-2")
                        print(f"  Mean b-parameter: {cddf_dict['b_params'].mean():.1f} km/s")
                else:
                    print(f"  Error: {cddf_dict['error']}")
                    print("  Falling back to simple method...")
                    cddf_dict = _cddf_simple(
                        tau, velocity_spacing, tau_threshold, colden,
                        redshift, box_size_ckpc_h, hubble, omega_m)

            else:
                print(f"Unknown method '{cd_method}', using simple")
                cddf_dict = _cddf_simple(
                    tau, velocity_spacing, tau_threshold, colden,
                    redshift, box_size_ckpc_h, hubble, omega_m)
        else:
            print(f"\n[4/8] Skipping column density distribution (--skip-cddf)")

        if not skip_lwd:
            print("\n[4b/8] Computing line width distribution b(N_HI)...")
            try:
                lwd_dict = compute_line_width_distribution(
                    tau, velocity_spacing, threshold=tau_threshold, colden=colden)
                print(f"Identified {lwd_dict['n_absorbers']} absorbers with b-parameters")
                if lwd_dict['n_absorbers'] > 0:
                    print(f"Median b-parameter: {lwd_dict['b_median']:.1f} km/s")
                    print(f"Mean b-parameter: {lwd_dict['b_mean']:.1f} ± {lwd_dict['b_std']:.1f} km/s")
                    T_mean = 1.28e4 * lwd_dict['b_mean']**2
                    print(f"Implied temperature: {T_mean/1e3:.0f} × 10³ K")
                else:
                    print("Warning: No absorbers found for line width analysis")
                    lwd_dict = None
            except Exception as e:
                print(f"Warning: Line width analysis failed: {e}")
                lwd_dict = None
        else:
            print("\n[4b/8] Skipping line width distribution (--skip-line-width)")

    # [4c/8] Temperature-density relation (if data available)
    skip_tdens = getattr(args, 'skip_temp_density', False)
    if skip_tdens:
        print("\n[4c/8] Skipping temperature-density relation (--skip-temp-density)")
        tdens_dict = None
    else:
        print("\n[4c/8] Checking for temperature-density data...")
        tdens_dict = None
        try:
            with h5py.File(spectra_file, 'r') as f:
                temp_elem = 'H'
                temp_ion = '1'
                if '/' in tau_path:
                    parts = tau_path.split('/')
                    if len(parts) >= 3:
                        temp_elem = parts[1]
                        temp_ion = parts[2]

                has_temp = ('temperature' in f and
                            temp_elem in f['temperature'] and
                            temp_ion in f['temperature'][temp_elem])
                has_dens = ('density_weight_density' in f and
                            temp_elem in f['density_weight_density'] and
                            temp_ion in f['density_weight_density'][temp_elem])

                if has_temp and has_dens:
                    print("Found temperature and density data - using chunked processing...")
                    tdens_dict = compute_temperature_density_chunked(
                        spectra_file, tau_path, min_tau=0.1, chunk_size=1000
                    )

                    if 'error' in tdens_dict:
                        print(f"  Error: {tdens_dict['error']}")
                        tdens_dict = None
                    else:
                        print(f"  Valid pixels: {tdens_dict['n_pixels']:,}")
                        if np.isfinite(tdens_dict['T0']):
                            print(f"  T_0 (at mean density): {tdens_dict['T0']:.0f} K")
                            print(f"  gamma (polytropic index): {tdens_dict['gamma']:.3f}")
                        else:
                            print(f"  Warning: T-ρ fit failed")
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
                    
                    # Try to load colden for this line
                    line_colden = None
                    try:
                        colden_path = f'colden/{elem}/{ion}'
                        if colden_path in f:
                            line_colden = np.array(f[colden_path])
                            # Subsample if needed to match line_tau
                            if max_sightlines is not None and line_colden.shape[0] > max_sightlines:
                                line_colden = line_colden[indices]
                    except Exception:
                        pass  # colden not available for this ion

                    # Create descriptive ion name
                    # Map element symbols to common names
                    elem_map = {'H': 'HI', 'C': 'CIV',
                                'O': 'OVI', 'Mg': 'MgII', 'Si': 'SiIV'}
                    if elem in elem_map:
                        ion_name = f"{elem_map[elem]} {wave}Å"
                    else:
                        ion_name = f"{elem}{ion}+ {wave}Å"

                    # Use lower threshold for metal lines
                    threshold = tau_threshold if elem == 'H' else config.TAU_THRESHOLD_METAL

                    print(f"Analyzing {
                          ion_name} (threshold={threshold})...")
                    stats = compute_metal_line_statistics(
                        line_tau,
                        velocity_spacing=velocity_spacing,
                        ion_name=ion_name,
                        threshold=threshold,
                        colden=line_colden
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

    # [9/8] Export analysis data
    print("\n[9/8] Exporting analysis results...")
    try:
        from scripts.data_export import save_analysis_results, get_analysis_output_dir

        # Persist the tau_eff uncertainty into flux_stats (effective_tau is already
        # there). It was computed but never exported, which blocked CSV-only reload
        # of the tau_eff error bars in evolve/compare. Per-sightline scatter still
        # needs raw tau, so we only carry the two scalars.
        stats['tau_eff_err'] = tau_eff_dict.get('tau_eff_err', float('nan'))
        stats['tau_eff_std'] = tau_eff_dict.get('tau_eff_std', float('nan'))

        # Prepare results dictionary
        results_dict = {
            'metadata': {
                'spectra_file': spectra_file,
                'redshift': redshift,
                'n_sightlines': n_sightlines,
                'n_pixels': n_pixels,
                'cd_method': cd_method,
            },
            'flux_stats': stats,
            'tau_eff': tau_eff_dict,
            'power_spectrum': power_dict,
            'cddf': cddf_dict,
            'line_widths': lwd_dict,
            'temp_density': tdens_dict,
            'metal_lines': metal_line_stats if len(metal_line_stats) > 0 else None,
        }
        
        # Get output directory
        output_dir = get_analysis_output_dir(spectra_file)
        
        # Save results
        created_files = save_analysis_results(results_dict, output_dir)
        
        print(f"Exported analysis data to: {output_dir}")
        
    except Exception as e:
        print(f"Warning: Could not export analysis data: {e}")
        print("(Analysis completed successfully, but data export failed)")

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
    
    try:
        print(f"\nData exported to: {output_dir}")
        print("- power_spectrum.csv")
        print("- cddf.csv")
        print("- flux_stats.csv")
        if lwd_dict is not None and lwd_dict['n_absorbers'] > 0:
            print("- line_widths.csv")
        if tdens_dict is not None:
            print("- temp_density.csv")
        if len(metal_line_stats) > 0:
            print("- metal_lines.csv")
    except:
        pass
    
    print(f"{'=' * 70}")

    return 0
