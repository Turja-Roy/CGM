import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
from pathlib import Path
from .analysis import (
    compute_flux_statistics,
    compute_effective_optical_depth,
    compute_power_spectrum,
    compute_column_density_distribution,
    compute_line_width_distribution,
    compute_temperature_density_relation,
)
from .plotting import save_plot
from .statistical_tests import comprehensive_comparison, pairwise_comparison_matrix, format_test_results_table
from .exploratory import (
    extract_spectral_features,
    compare_features,
    compare_distributions,
    compute_correlation_matrix,
    spectra_clustering_analysis,
    physics_regime_analysis
)


def _parse_csv_comment_header(path):
    """Parse leading '# key = value' comment lines into a dict.

    Values are coerced to float when possible; trailing units (e.g.
    '1.234 Mpc (comoving)') are dropped, keeping only the first token.
    """
    meta = {}
    with open(path, 'r') as fh:
        for line in fh:
            if not line.startswith('#'):
                break
            body = line.lstrip('#').strip()
            if '=' not in body:
                continue
            key, _, val = body.partition('=')
            key = key.strip()
            tokens = val.strip().split()
            val = tokens[0] if tokens else ''
            try:
                meta[key] = float(val)
            except ValueError:
                meta[key] = val
    return meta


def load_results_from_csv(spectra_file):
    """Build a load_spectra_results-shaped dict from the analysis CSVs.

    The physics is identical to the raw-tau path: every value here is read back
    verbatim from what `analyze` wrote (effective_tau = -ln<F>, P_F(k), CDDF,
    T-rho). The spectra_file path is used only as a key to locate the analysis
    output directory, so this works even after the raw spectra HDF5 is deleted.

    Fields that were never persisted (per-sightline / per-pixel arrays) are set
    to None; tau_eff_err falls back to NaN for CSVs written before it was
    exported (plots render NaN error bars as no bar, not a crash).

    Returns a dict with success=False (and 'error') if the core CSVs are absent,
    signalling the caller to try the raw-tau path instead.
    """
    import pandas as pd
    from scripts.data_export import get_analysis_output_dir

    out = {'filepath': str(spectra_file), 'success': False, 'loaded_from': 'csv'}
    output_dir = Path(get_analysis_output_dir(spectra_file))

    ps_path = output_dir / 'power_spectrum.csv'
    cddf_path = output_dir / 'cddf.csv'
    stats_path = output_dir / 'flux_stats.csv'
    if not (ps_path.exists() and cddf_path.exists() and stats_path.exists()):
        out['error'] = f'analysis CSVs not found in {output_dir}'
        return out

    try:
        # --- flux statistics (two-column statistic,value) ---
        fs_df = pd.read_csv(stats_path)
        flux_stats = dict(zip(fs_df['statistic'],
                              pd.to_numeric(fs_df['value'], errors='coerce')))
        out['flux_stats'] = flux_stats

        # --- tau_eff: effective_tau == -ln<F>, identical to the raw-tau value ---
        out['tau_eff'] = {
            'tau_eff': flux_stats.get('effective_tau'),
            'mean_flux': flux_stats.get('mean_flux'),
            'tau_eff_err': flux_stats.get('tau_eff_err', np.nan),
            'tau_eff_std': flux_stats.get('tau_eff_std', np.nan),
            'tau_eff_per_sightline': None,  # needs raw tau; not persisted
        }

        # --- power spectrum ---
        ps_df = pd.read_csv(ps_path)
        k = ps_df['k_s_per_km'].values
        ps = {
            'k': k,
            'P_k_mean': ps_df['P_k_mean_km_per_s'].values,
            'P_k_std': ps_df['P_k_std'].values if 'P_k_std' in ps_df else np.zeros_like(k),
            'P_k_err': ps_df['P_k_err'].values if 'P_k_err' in ps_df else np.zeros_like(k),
            'mean_flux': flux_stats.get('mean_flux'),
        }
        out['power_spectrum'] = ps

        # --- CDDF (+ metadata from the comment header) ---
        cmeta = _parse_csv_comment_header(cddf_path)
        cddf_df = pd.read_csv(cddf_path, comment='#')
        log10_N = cddf_df['log10_N_HI'].values
        bin_centers = (cddf_df['bin_center'].values
                       if 'bin_center' in cddf_df else 10.0 ** log10_N)
        if 'delta_log_N' in cddf_df:
            delta_log_N = cddf_df['delta_log_N'].values
        else:
            d = np.diff(log10_N)
            delta_log_N = np.full_like(log10_N, np.median(d) if len(d) else np.nan)
        f_N = cddf_df['f_N_HI'].values
        out['cddf'] = {
            'log10_N_HI': log10_N,
            'bin_centers': bin_centers,
            'bins': 10.0 ** np.append(log10_N - delta_log_N / 2.0,
                                      log10_N[-1] + delta_log_N[-1] / 2.0) if len(log10_N) else np.array([]),
            'counts': cddf_df['counts'].values if 'counts' in cddf_df else np.zeros_like(log10_N),
            'f_N': f_N,
            'f_N_HI': f_N,
            'delta_log_N': delta_log_N,
            'beta_fit': cmeta.get('beta_fit', np.nan),
            'n_absorbers': int(cmeta.get('n_absorbers', 0)),
            'n_sightlines': int(cmeta['n_sightlines']) if 'n_sightlines' in cmeta else None,
            # dx_mode = 1 writes the header key as "X", not "dX".
            'dX': cmeta.get('dX', cmeta.get('X', np.nan)),
            'redshift': cmeta.get('redshift'),
            # Carry the config echo through so plots pick the right units and so
            # files written under different CDDF settings stay distinguishable.
            # None for CSVs with no echo, i.e. the per-dex Mpc^-1 quantity.
            'norm_mode': cmeta.get('norm_mode'),
            'dx_mode': cmeta.get('dx_mode'),
            'absorber_mode': cmeta.get('absorber_mode'),
            'fit_log_N_min': cmeta.get('fit_log_N_min'),
            'fit_log_N_max': cmeta.get('fit_log_N_max'),
            'saturated': str(cmeta.get('saturated', '')).lower().startswith('true'),
        }

        # --- line widths (recompute summary stats from the b array) ---
        out['line_widths'] = None
        lw_path = output_dir / 'line_widths.csv'
        if lw_path.exists():
            lw_df = pd.read_csv(lw_path)
            if 'b_param_km_s' in lw_df and len(lw_df) > 0:
                b = lw_df['b_param_km_s'].values
                out['line_widths'] = {
                    'N_HI': lw_df['N_HI'].values if 'N_HI' in lw_df else np.array([]),
                    'b_params': b,
                    'b_median': float(np.median(b)),
                    'b_mean': float(np.mean(b)),
                    'b_std': float(np.std(b)),
                    'n_absorbers': int(len(b)),
                }

        # --- temperature-density (fit params from header, scatter from columns) ---
        out['temp_density'] = None
        td_path = output_dir / 'temp_density.csv'
        if td_path.exists():
            tmeta = _parse_csv_comment_header(td_path)
            try:
                td_df = pd.read_csv(td_path, comment='#')
            except Exception:
                td_df = None
            has_cols = td_df is not None and 'log_temperature' in td_df and len(td_df) > 0
            out['temp_density'] = {
                'T0': tmeta.get('T0', np.nan),
                'gamma': tmeta.get('gamma', np.nan),
                'gamma_err': tmeta.get('gamma_err', np.nan),
                'n_pixels': int(tmeta['n_pixels']) if 'n_pixels' in tmeta and str(tmeta['n_pixels']).replace('.', '', 1).isdigit() else None,
                'log_T': td_df['log_temperature'].values if has_cols else np.array([]),
                'log_rho': td_df['log_density'].values if has_cols else np.array([]),
            }

        # --- top-level metadata ---
        out['redshift'] = (float(out['cddf']['redshift'])
                           if out['cddf']['redshift'] is not None else None)
        out['n_sightlines'] = out['cddf']['n_sightlines']
        out['n_pixels'] = 2 * (len(k) - 1) if len(k) else None
        out['success'] = True
    except Exception as e:
        out['success'] = False
        out['error'] = f'CSV parse failed in {output_dir}: {e}'
    return out


def load_spectra_results(spectra_file, velocity_spacing=0.1, prefer_csv=True):
    """Load Lya analysis results for one sim/snapshot, CSV-first.

    Reads the analysis CSVs (output/analysis/.../snap-XXX/) by default — they
    carry identical physics to the raw-tau computation and survive deletion of
    the spectra HDF5. Falls back to recomputing from the raw tau only when the
    CSVs are absent and the spectra file still exists.
    """
    if prefer_csv:
        csv_res = load_results_from_csv(spectra_file)
        if csv_res.get('success'):
            return csv_res
        csv_err = csv_res.get('error')
    else:
        csv_err = None

    if os.path.exists(spectra_file):
        return _load_results_from_spectra(spectra_file, velocity_spacing)

    return {'filepath': str(spectra_file), 'success': False,
            'error': csv_err or f'no analysis CSVs and spectra file missing: {spectra_file}'}


def _load_results_from_spectra(spectra_file, velocity_spacing=0.1):
    """Recompute analysis results from the raw tau in the spectra HDF5.

    Legacy / fallback path used only when the analysis CSVs are unavailable
    (see load_spectra_results). Heavier: reads the full per-sightline tau array.
    """
    results = {
        'filepath': str(spectra_file),
        'success': False,
    }

    try:
        with h5py.File(spectra_file, 'r') as f:
            # Get metadata
            redshift = None
            box_size_ckpc_h = None
            hubble = 0.6774
            omega_m = 0.3089
            
            if 'Header' in f:
                header = f['Header'].attrs
                redshift = header.get('redshift', header.get('Redshift', None))
                box_size_ckpc_h = header.get('box', header.get('BoxSize', None))
                hubble = header.get('hubble', header.get('HubbleParam', 0.6774))
                omega_m = header.get('omegam', header.get('Omega0', 0.3089))
                
            results['redshift'] = float(redshift) if redshift is not None else None
            
            # Auto-detect tau data (try HI Lya first)
            tau = None
            tau_path = None
            if 'tau/H/1/1215' in f:
                tau = np.array(f['tau/H/1/1215'])
                tau_path = 'tau/H/1/1215'
            elif 'tau' in f and isinstance(f['tau'], h5py.Dataset):
                tau = np.array(f['tau'])
                tau_path = 'tau'
            
            if tau is None:
                results['error'] = 'No tau data found'
                return results
            
            flux = np.exp(-tau)
            n_sightlines, n_pixels = tau.shape
            
            results['n_sightlines'] = n_sightlines
            results['n_pixels'] = n_pixels
            
            # Compute all analyses with proper normalization
            results['flux_stats'] = compute_flux_statistics(tau)
            results['tau_eff'] = compute_effective_optical_depth(tau)
            results['power_spectrum'] = compute_power_spectrum(flux, velocity_spacing)
            results['cddf'] = compute_column_density_distribution(
                tau, velocity_spacing, threshold=0.5, colden=None,
                redshift=redshift, box_size_ckpc_h=box_size_ckpc_h,
                hubble=hubble, omega_m=omega_m)
            
            # Try line width analysis
            try:
                results['line_widths'] = compute_line_width_distribution(tau, velocity_spacing)
            except:
                results['line_widths'] = None
            
            # Try T-ρ analysis
            try:
                if tau_path and '/' in tau_path:
                    parts = tau_path.split('/')
                    temp_elem = parts[1] if len(parts) >= 2 else 'H'
                    temp_ion = parts[2] if len(parts) >= 3 else '1'
                else:
                    temp_elem, temp_ion = 'H', '1'
                
                has_temp = ('temperature' in f and temp_elem in f['temperature'] and 
                           temp_ion in f['temperature'][temp_elem])
                has_dens = ('density_weight_density' in f and temp_elem in f['density_weight_density'] and 
                           temp_ion in f['density_weight_density'][temp_elem])
                
                if has_temp and has_dens:
                    temperature = np.array(f['temperature'][temp_elem][temp_ion])
                    density = np.array(f['density_weight_density'][temp_elem][temp_ion])
                    results['temp_density'] = compute_temperature_density_relation(
                        temperature, density, tau, min_tau=0.1
                    )
                else:
                    results['temp_density'] = None
            except:
                results['temp_density'] = None
            
            results['success'] = True
            
    except Exception as e:
        results['error'] = str(e)
        return results
    
    return results


def compare_simulations(spectra_files, labels=None, output_path=None):
    if labels is None:
        labels = [f"Sim {i}" for i in range(len(spectra_files))]
    
    # Load all results
    print(f"Loading {len(spectra_files)} simulation results...")
    all_results = []
    for i, (fpath, label) in enumerate(zip(spectra_files, labels)):
        print(f"  [{i+1}/{len(spectra_files)}] {label}: {fpath}")
        results = load_spectra_results(fpath)
        if results['success']:
            results['label'] = label
            all_results.append(results)
            print(f"      OK z={results['redshift']:.3f}, N={results['n_sightlines']}")
        else:
            print(f"      Failed: {results.get('error', 'unknown error')}")
    
    if len(all_results) == 0:
        print("Error: No valid results to compare")
        return None
    
    # Create comparison figure
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Color scheme
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Panel 1: Flux power spectrum
    ax = fig.add_subplot(gs[0, :])
    for i, res in enumerate(all_results):
        ps = res['power_spectrum']
        k = ps['k']
        P_k = ps['P_k_mean']
        P_k_err = ps['P_k_err']
        
        mask = k > 0
        k = k[mask]
        P_k = P_k[mask]
        P_k_err = P_k_err[mask]
        
        color = colors[i % len(colors)]
        ax.loglog(k, P_k, 'o-', color=color, linewidth=2, markersize=3,
                 label=f"{res['label']} (z={res['redshift']:.2f})", alpha=0.8)
        ax.fill_between(k, P_k - P_k_err, P_k + P_k_err, alpha=0.2, color=color)
    
    ax.set_xlabel(r'Wavenumber $k$ [s/km]', fontsize=13)
    ax.set_ylabel(r'Power Spectrum $P_F(k)$ [km/s]', fontsize=13)
    ax.set_title('Flux Power Spectrum Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=10, loc='best')
    
    # Panel 2: Effective optical depth
    ax = fig.add_subplot(gs[1, 0])
    tau_effs = [res['tau_eff']['tau_eff'] for res in all_results]
    tau_errs = [res['tau_eff']['tau_eff_err'] for res in all_results]
    x_pos = np.arange(len(all_results))
    
    bars = ax.bar(x_pos, tau_effs, yerr=tau_errs, capsize=5,
                  color=[colors[i % len(colors)] for i in range(len(all_results))],
                  alpha=0.7, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([res['label'] for res in all_results], rotation=45, ha='right')
    ax.set_ylabel(r'Effective Optical Depth $\tau_{\rm eff}$', fontsize=12)
    ax.set_title('Effective Optical Depth', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: Mean flux
    ax = fig.add_subplot(gs[1, 1])
    mean_fluxes = [res['flux_stats']['mean_flux'] for res in all_results]
    
    bars = ax.bar(x_pos, mean_fluxes,
                  color=[colors[i % len(colors)] for i in range(len(all_results))],
                  alpha=0.7, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([res['label'] for res in all_results], rotation=45, ha='right')
    ax.set_ylabel(r'Mean Transmitted Flux $\langle F \rangle$', fontsize=12)
    ax.set_title('Mean Flux', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: Column density distribution
    ax = fig.add_subplot(gs[2, 0])
    for i, res in enumerate(all_results):
        cddf = res['cddf']
        if cddf['n_absorbers'] > 0 and len(cddf['counts']) > 0:
            color = colors[i % len(colors)]
            
            bin_centers = cddf['bin_centers']
            log_bin_centers = np.log10(bin_centers)

            # Use the f(N) the analysis computed and wrote to cddf.csv. The
            # fallback normalises by absorber count rather than by sightlines
            # and path length, so it is not f(N) and is only a last resort.
            f_N = cddf.get('f_N')
            if f_N is None or len(f_N) != len(bin_centers):
                if 'delta_log_N' in cddf:
                    delta_log_N = cddf['delta_log_N'][0]
                else:
                    delta_log_N = np.mean(np.diff(np.log10(cddf['bins'])))
                f_N = cddf['counts'] / (cddf['n_absorbers'] * delta_log_N)
            f_N = np.asarray(f_N)
            
            # Only plot non-zero bins
            mask = f_N > 0
            if np.any(mask):
                ax.scatter(log_bin_centers[mask], f_N[mask],
                          s=30, alpha=0.6, color=color, label=res['label'])
                
                # Plot the fit only over the range it was fitted on
                if not np.isnan(cddf['beta_fit']):
                    lo = cddf.get('fit_log_N_min') or 12.0
                    hi = cddf.get('fit_log_N_max') or 16.0
                    N_fit = np.logspace(lo, hi, 100)
                    # Power law: f(N) = A * N^(-beta)
                    A_norm = f_N[mask].max() / (bin_centers[mask][np.argmax(f_N[mask])]**(-cddf['beta_fit']))
                    f_fit = A_norm * N_fit**(-cddf['beta_fit'])
                    ax.plot(np.log10(N_fit), f_fit, '--', color=color, alpha=0.5, linewidth=1.5)

    # Units follow norm_mode (see config.CDDF_OPTIONS).
    per_dN = any((r.get('cddf') or {}).get('norm_mode') == 1 for r in all_results)
    ax.set_xlabel(r'$\log_{10}(N_{\rm HI} / {\rm cm}^{-2})$', fontsize=12)
    ax.set_ylabel(r'$f(N_{\rm HI})$ [cm$^{2}$]' if per_dN
                  else r'$f(N_{\rm HI})$ [dN/d log$_{10}$ N]', fontsize=12)
    ax.set_yscale('log')
    ax.set_title('Column Density Distribution', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    
    # Panel 5: Temperature-density parameters (if available)
    ax = fig.add_subplot(gs[2, 1])
    T0_values = []
    gamma_values = []
    valid_labels = []
    
    for i, res in enumerate(all_results):
        if res['temp_density'] is not None:
            td = res['temp_density']
            if np.isfinite(td['T0']) and np.isfinite(td['gamma']):
                T0_values.append(td['T0'])
                gamma_values.append(td['gamma'])
                valid_labels.append(res['label'])
    
    if len(T0_values) > 0:
        x_pos_td = np.arange(len(T0_values))
        
        ax2 = ax.twinx()
        
        bars1 = ax.bar(x_pos_td - 0.2, T0_values, width=0.4, 
                      color='steelblue', alpha=0.7, label=r'$T_0$ [K]')
        bars2 = ax2.bar(x_pos_td + 0.2, gamma_values, width=0.4,
                       color='coral', alpha=0.7, label=r'$\gamma$')
        
        ax.set_xticks(x_pos_td)
        ax.set_xticklabels(valid_labels, rotation=45, ha='right')
        ax.set_ylabel(r'$T_0$ at Mean Density [K]', fontsize=12, color='steelblue')
        ax2.set_ylabel(r'Polytropic Index $\gamma$', fontsize=12, color='coral')
        ax.tick_params(axis='y', labelcolor='steelblue')
        ax2.tick_params(axis='y', labelcolor='coral')
        ax.set_title('IGM Equation of State', fontsize=13, fontweight='bold')
        
        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left')
    else:
        ax.text(0.5, 0.5, 'No temperature-density\ndata available',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Overall title
    fig.suptitle('Simulation Comparison', fontsize=16, fontweight='bold', y=0.995)
    
    if output_path:
        save_plot(fig, output_path)
        print(f"\nComparison plot saved: {output_path}")
    
    plt.close()
    
    # Return summary
    comparison = {
        'n_simulations': len(all_results),
        'labels': [res['label'] for res in all_results],
        'results': all_results
    }
    
    return comparison


def track_redshift_evolution(spectra_files, labels=None, output_path=None):
    if labels is None:
        labels = [f"Snap {i}" for i in range(len(spectra_files))]
    
    # Load all results
    print(f"Loading {len(spectra_files)} snapshots for evolution tracking...")
    all_results = []
    for i, (fpath, label) in enumerate(zip(spectra_files, labels)):
        print(f"  [{i+1}/{len(spectra_files)}] {label}: {fpath}")
        results = load_spectra_results(fpath)
        if results['success'] and results['redshift'] is not None:
            results['label'] = label
            all_results.append(results)
            print(f"      OK z={results['redshift']:.3f}")
        else:
            print(f"      Failed or no redshift")
    
    if len(all_results) == 0:
        print("Error: No valid results for evolution tracking")
        return None
    
    # Sort by redshift
    all_results.sort(key=lambda x: x['redshift'])
    
    # Extract evolution data
    redshifts = np.array([res['redshift'] for res in all_results])
    tau_effs = np.array([res['tau_eff']['tau_eff'] for res in all_results])
    tau_errs = np.array([res['tau_eff']['tau_eff_err'] for res in all_results])
    mean_fluxes = np.array([res['flux_stats']['mean_flux'] for res in all_results])
    n_absorbers = np.array([res['cddf']['n_absorbers'] for res in all_results])
    
    # T-ρ parameters (if available)
    T0_values = []
    gamma_values = []
    z_with_T = []
    
    for res in all_results:
        if res['temp_density'] is not None:
            td = res['temp_density']
            if np.isfinite(td['T0']) and np.isfinite(td['gamma']):
                z_with_T.append(res['redshift'])
                T0_values.append(td['T0'])
                gamma_values.append(td['gamma'])
    
    z_with_T = np.array(z_with_T)
    T0_values = np.array(T0_values)
    gamma_values = np.array(gamma_values)
    
    # Create evolution figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel 1: Effective optical depth evolution
    ax = axes[0, 0]
    ax.errorbar(redshifts, tau_effs, yerr=tau_errs, fmt='o-', 
               color='steelblue', linewidth=2, markersize=6, capsize=4, alpha=0.8)
    ax.set_xlabel('Redshift $z$', fontsize=13)
    ax.set_ylabel(r'Effective Optical Depth $\tau_{\rm eff}$', fontsize=13)
    ax.set_title(r'$\tau_{\rm eff}(z)$ Evolution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # Higher z on left
    
    # Panel 2: Mean flux evolution
    ax = axes[0, 1]
    ax.plot(redshifts, mean_fluxes, 'o-', color='coral', 
           linewidth=2, markersize=6, alpha=0.8)
    ax.set_xlabel('Redshift $z$', fontsize=13)
    ax.set_ylabel(r'Mean Transmitted Flux $\langle F \rangle$', fontsize=13)
    ax.set_title(r'$\langle F \rangle(z)$ Evolution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    
    # Panel 3: Number of absorbers
    ax = axes[1, 0]
    ax.plot(redshifts, n_absorbers, 'o-', color='green',
           linewidth=2, markersize=6, alpha=0.8)
    ax.set_xlabel('Redshift $z$', fontsize=13)
    ax.set_ylabel('Number of Absorbers', fontsize=13)
    ax.set_title('Absorber Count Evolution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    
    # Panel 4: Temperature-density evolution
    ax = axes[1, 1]
    if len(z_with_T) > 0:
        ax2 = ax.twinx()
        
        line1 = ax.plot(z_with_T, T0_values, 'o-', color='steelblue',
                       linewidth=2, markersize=6, alpha=0.8, label=r'$T_0$')
        line2 = ax2.plot(z_with_T, gamma_values, 's-', color='coral',
                        linewidth=2, markersize=6, alpha=0.8, label=r'$\gamma$')
        
        ax.set_xlabel('Redshift $z$', fontsize=13)
        ax.set_ylabel(r'$T_0$ at Mean Density [K]', fontsize=13, color='steelblue')
        ax2.set_ylabel(r'Polytropic Index $\gamma$', fontsize=13, color='coral')
        ax.tick_params(axis='y', labelcolor='steelblue')
        ax2.tick_params(axis='y', labelcolor='coral')
        ax.set_title('IGM Equation of State Evolution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        
        # Combined legend
        lines = line1 + line2
        labels_leg = [l.get_label() for l in lines]
        ax.legend(lines, labels_leg, fontsize=11, loc='best')
    else:
        ax.text(0.5, 0.5, 'No temperature-density\ndata available',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Overall title
    z_min, z_max = redshifts.min(), redshifts.max()
    fig.suptitle(f'Redshift Evolution (z = {z_min:.2f} - {z_max:.2f})', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        save_plot(fig, output_path)
        print(f"\nEvolution plot saved: {output_path}")
    
    plt.close()
    
    # Return summary
    evolution = {
        'n_snapshots': len(all_results),
        'redshift_range': (float(redshifts.min()), float(redshifts.max())),
        'redshifts': redshifts.tolist(),
        'tau_eff': tau_effs.tolist(),
        'mean_flux': mean_fluxes.tolist(),
        'n_absorbers': n_absorbers.tolist(),
        'results': all_results
    }
    
    if len(z_with_T) > 0:
        evolution['T0'] = T0_values.tolist()
        evolution['gamma'] = gamma_values.tolist()
        evolution['z_with_T'] = z_with_T.tolist()
    
    return evolution


def compare_simulations_comprehensive(spectra_files, labels=None, output_dir=None, mode='detailed'):
    """
    Comprehensive comparison with multiple analysis modes.
    mode: 'quick' (basic plots), 'detailed' (enhanced plots), 'full' (all analyses)
    
    Uses memory-efficient lazy loading to handle large datasets.
    """
    if labels is None:
        labels = [f"Sim {i}" for i in range(len(spectra_files))]
    
    print(f"[COMPREHENSIVE COMPARISON - Mode: {mode}]")
    print(f"Loading {len(spectra_files)} simulations...")
    
    all_results = []
    valid_files = []
    valid_labels = []
    
    for i, (fpath, label) in enumerate(zip(spectra_files, labels)):
        results = load_spectra_results(fpath)
        if results['success']:
            results['label'] = label
            all_results.append(results)
            valid_files.append(fpath)
            valid_labels.append(label)
        else:
            print(f"  Skipping {label}: {results.get('error', 'unknown error')}")
    
    if len(all_results) == 0:
        return None

    # detailed/full modes do per-pixel / per-sightline work (flux distributions,
    # KS tests, clustering, feature extraction) that is NOT stored in the CSVs.
    # If the raw spectra HDF5 files are gone, transparently downgrade to 'quick'
    # (fully CSV-based) instead of crashing.
    if mode in ('detailed', 'full'):
        missing = [fp for fp in valid_files if not os.path.exists(fp)]
        if missing:
            print(f"  [warn] {len(missing)}/{len(valid_files)} raw spectra file(s) "
                  f"absent; per-pixel analyses need them. Downgrading mode "
                  f"'{mode}' -> 'quick' (CSV-only).")
            mode = 'quick'

    if output_dir is None:
        output_dir = Path('plots/comparisons')
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Basic comparison plot (always generated)
    compare_simulations(valid_files, valid_labels, output_path=output_dir / 'comparison_basic.png')
    
    if mode in ['detailed', 'full']:
        print("\n[ENHANCED COMPARISON PLOTS]")
        
        # Enhanced comparison with box plots and sample spectra (lazy loaded)
        _create_enhanced_comparison_plot_lazy(all_results, valid_labels, valid_files, 
                                             output_dir / 'comparison_enhanced.png')
        
        # Power spectrum ratios
        _create_power_spectrum_ratio_plot(all_results, valid_labels,
                                         output_dir / 'power_spectrum_ratios.png')
        
        # Distribution comparison for flux (chunked processing)
        print("Comparing flux distributions (chunked processing)...")
        compare_distributions_lazy(valid_files, valid_labels, 
                                  output_dir / 'flux_distributions.png', 'Flux')
        
        # Statistical tests (streaming)
        print("\n[STATISTICAL TESTS]")
        test_results = []
        for i in range(len(all_results)):
            for j in range(i + 1, len(all_results)):
                from scripts.statistical_tests import comprehensive_comparison as comp_test
                # Load flux data in chunks for statistical tests
                flux_i = _load_flux_chunked(valid_files[i], max_samples=100000)
                flux_j = _load_flux_chunked(valid_files[j], max_samples=100000)
                result = comp_test(flux_i, flux_j, 
                                 valid_labels[i], valid_labels[j])
                test_results.append(result)
        
        if test_results:
            from scripts.statistical_tests import format_test_results_table
            test_summary = format_test_results_table(test_results, correction='bonferroni')
            with open(output_dir / 'statistical_tests.txt', 'w') as f:
                f.write(test_summary)
            print(test_summary[:500] + "...\n(Full results in statistical_tests.txt)")
        
        # Correlation matrices
        compute_correlation_matrix(all_results, valid_labels,
                                  output_dir / 'correlation_matrices.png')
    
    if mode == 'full':
        print("\n[FULL EXPLORATORY ANALYSIS]")
        
        # Feature extraction and comparison (chunked)
        print("Extracting spectral features (chunked processing)...")
        features_list = [extract_spectral_features_chunked(fpath) for fpath in valid_files]
        compare_features(features_list, valid_labels,
                        output_dir / 'feature_comparison.png')
        
        # Physics regime analysis
        physics_regime_analysis(all_results, valid_labels,
                              output_dir / 'physics_regimes.png')
        
        # Clustering analysis
        print("Running PCA/t-SNE clustering...")
        spectra_clustering_analysis(valid_files, valid_labels,
                                   output_dir / 'spectra_clustering.png',
                                   n_samples=500)
        
        # Pairwise comparison matrix (streaming)
        print("Computing pairwise comparison matrix...")
        from scripts.statistical_tests import pairwise_comparison_matrix
        flux_data_samples = [_load_flux_chunked(f, max_samples=50000) for f in valid_files]
        matrix_result = pairwise_comparison_matrix(flux_data_samples, valid_labels, metric='ks')
        _plot_pairwise_matrix(matrix_result, output_dir / 'pairwise_ks_matrix.png')
    
    print(f"\n[COMPLETE] All plots saved to {output_dir}/")
    
    return {
        'n_simulations': len(all_results),
        'labels': labels,
        'results': all_results,
        'output_dir': str(output_dir)
    }


def _create_enhanced_comparison_plot(all_results, labels, all_tau_data, output_path):
    """Enhanced comparison with box plots, ratios, sample spectra, significance."""
    n_sims = len(all_results)
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.4)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Panel 1: Power spectrum (same as before)
    ax = fig.add_subplot(gs[0, :])
    for i, res in enumerate(all_results):
        ps = res['power_spectrum']
        k, P_k = ps['k'], ps['P_k_mean']
        mask = k > 0
        ax.loglog(k[mask], P_k[mask], 'o-', color=colors[i % len(colors)], 
                 linewidth=2, markersize=3, label=f"{res['label']}", alpha=0.8)
    ax.set_xlabel(r'$k$ [s/km]', fontsize=12)
    ax.set_ylabel(r'$P_F(k)$ [km/s]', fontsize=12)
    ax.set_title('Flux Power Spectrum', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    
    # Panel 2: τ_eff box plot
    ax = fig.add_subplot(gs[1, 0])
    tau_eff_data = [res['tau_eff']['tau_eff_per_sightline'] for res in all_results]
    bp = ax.boxplot(tau_eff_data, labels=labels, patch_artist=True, showfliers=False)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel(r'$\tau_{\rm eff}$', fontsize=12)
    ax.set_title('Optical Depth Distribution', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # Panel 3: Flux box plot
    ax = fig.add_subplot(gs[1, 1])
    flux_data = [np.exp(-tau) for tau in all_tau_data]
    bp = ax.boxplot([f.flatten()[:10000] for f in flux_data], labels=labels, patch_artist=True, showfliers=False)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Flux', fontsize=12)
    ax.set_title('Flux Distribution', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # Panel 4: Sample spectra (3 random per sim)
    ax = fig.add_subplot(gs[1, 2])
    for i, (tau, label) in enumerate(zip(all_tau_data, labels)):
        n_sightlines = tau.shape[0]
        n_pixels = tau.shape[1]
        sample_idx = np.random.choice(n_sightlines, min(3, n_sightlines), replace=False)
        
        for idx in sample_idx:
            flux = np.exp(-tau[idx])
            v = np.arange(len(flux)) * 0.1
            ax.plot(v[:2000], flux[:2000], alpha=0.4, linewidth=0.8, 
                   color=colors[i % len(colors)])
    
    ax.set_xlabel('Velocity [km/s]', fontsize=10)
    ax.set_ylabel('Flux', fontsize=10)
    ax.set_title('Sample Spectra', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 200)
    ax.grid(alpha=0.3)
    
    # Panel 5: Power spectrum ratio (ref = first sim)
    ax = fig.add_subplot(gs[2, 0])
    ref_ps = all_results[0]['power_spectrum']
    for i in range(1, n_sims):
        ps = all_results[i]['power_spectrum']
        k = ps['k']
        mask = (k > 0) & (ref_ps['P_k_mean'] > 0)
        ratio = ps['P_k_mean'][mask] / ref_ps['P_k_mean'][mask]
        ax.semilogx(k[mask], ratio, 'o-', label=f"{labels[i]}/{labels[0]}", 
                   markersize=3, linewidth=1.5, color=colors[i % len(colors)])
    ax.axhline(1, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel(r'$k$ [s/km]', fontsize=11)
    ax.set_ylabel('Power Ratio', fontsize=11)
    ax.set_title('Power Spectrum Ratios', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # Panel 6: CDDF comparison
    ax = fig.add_subplot(gs[2, 1])
    for i, res in enumerate(all_results):
        cddf = res['cddf']
        if cddf['n_absorbers'] > 0:
            log_N = np.log10(cddf['bin_centers'])
            counts = cddf['counts']
            mask = counts > 0
            if np.any(mask):
                ax.plot(log_N[mask], counts[mask], 'o-', label=labels[i],
                       color=colors[i % len(colors)], alpha=0.7, markersize=4)
    ax.set_xlabel(r'$\log_{10}(N_{\rm HI})$', fontsize=11)
    ax.set_ylabel('Counts', fontsize=11)
    ax.set_yscale('log')
    ax.set_title('Column Density Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # Panel 7: Absorption fractions
    ax = fig.add_subplot(gs[2, 2])
    x = np.arange(n_sims)
    sat_fracs = [res['flux_stats']['deep_absorption_frac'] * 100 for res in all_results]
    mod_fracs = [res['flux_stats']['moderate_absorption_frac'] * 100 for res in all_results]
    weak_fracs = [res['flux_stats']['weak_absorption_frac'] * 100 for res in all_results]
    
    width = 0.25
    ax.bar(x - width, sat_fracs, width, label='Deep', color='darkred', alpha=0.7)
    ax.bar(x, mod_fracs, width, label='Moderate', color='orange', alpha=0.7)
    ax.bar(x + width, weak_fracs, width, label='Weak', color='green', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Pixel Fraction [%]', fontsize=11)
    ax.set_title('Absorption Regimes', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')
    
    # Panel 8: Mean statistics comparison
    ax = fig.add_subplot(gs[3, 0])
    mean_tau = [res['flux_stats']['mean_tau'] for res in all_results]
    mean_flux = [res['flux_stats']['mean_flux'] for res in all_results]
    ax2 = ax.twinx()
    
    ax.bar(x - 0.2, mean_tau, 0.4, label=r'$\langle\tau\rangle$', color='steelblue', alpha=0.7)
    ax2.bar(x + 0.2, mean_flux, 0.4, label=r'$\langle F\rangle$', color='coral', alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(r'$\langle\tau\rangle$', color='steelblue', fontsize=11)
    ax2.set_ylabel(r'$\langle F\rangle$', color='coral', fontsize=11)
    ax.tick_params(axis='y', labelcolor='steelblue')
    ax2.tick_params(axis='y', labelcolor='coral')
    ax.set_title('Mean Statistics', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # Panel 9: Variance comparison
    ax = fig.add_subplot(gs[3, 1])
    std_tau = [res['tau_eff']['tau_eff_std'] for res in all_results]
    std_flux = [res['flux_stats']['std_flux'] for res in all_results]
    ax2 = ax.twinx()
    
    ax.bar(x - 0.2, std_tau, 0.4, label=r'$\sigma_\tau$', color='steelblue', alpha=0.7)
    ax2.bar(x + 0.2, std_flux, 0.4, label=r'$\sigma_F$', color='coral', alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(r'$\sigma_\tau$', color='steelblue', fontsize=11)
    ax2.set_ylabel(r'$\sigma_F$', color='coral', fontsize=11)
    ax.tick_params(axis='y', labelcolor='steelblue')
    ax2.tick_params(axis='y', labelcolor='coral')
    ax.set_title('Variability', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # Panel 10: Summary table
    ax = fig.add_subplot(gs[3, 2])
    ax.axis('off')
    
    summary_text = "Key Statistics:\n" + "=" * 35 + "\n"
    for i, (res, label) in enumerate(zip(all_results, labels)):
        summary_text += f"\n{label}:\n"
        summary_text += f"  τ_eff: {res['tau_eff']['tau_eff']:.4f}\n"
        summary_text += f"  <F>: {res['flux_stats']['mean_flux']:.4f}\n"
        summary_text += f"  N_abs: {res['cddf']['n_absorbers']}\n"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
           fontsize=8, verticalalignment='top', fontfamily='monospace')
    
    fig.suptitle('Enhanced Simulation Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def _create_power_spectrum_ratio_plot(all_results, labels, output_path):
    """Create detailed power spectrum ratio plot."""
    n_sims = len(all_results)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Absolute power spectra
    ax = axes[0]
    for i, res in enumerate(all_results):
        ps = res['power_spectrum']
        k, P_k, P_k_err = ps['k'], ps['P_k_mean'], ps['P_k_err']
        mask = k > 0
        ax.loglog(k[mask], P_k[mask], 'o-', color=colors[i % len(colors)],
                 linewidth=2, markersize=3, label=labels[i], alpha=0.8)
        ax.fill_between(k[mask], P_k[mask] - P_k_err[mask], 
                       P_k[mask] + P_k_err[mask], 
                       alpha=0.15, color=colors[i % len(colors)])
    ax.set_xlabel(r'$k$ [s/km]', fontsize=12)
    ax.set_ylabel(r'$P_F(k)$ [km/s]', fontsize=12)
    ax.set_title('Power Spectra', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # Ratios relative to first
    ax = axes[1]
    ref_ps = all_results[0]['power_spectrum']
    for i in range(1, n_sims):
        ps = all_results[i]['power_spectrum']
        k = ps['k']
        mask = (k > 0) & (ref_ps['P_k_mean'] > 0)
        ratio = ps['P_k_mean'][mask] / ref_ps['P_k_mean'][mask]
        ax.semilogx(k[mask], ratio, 'o-', label=f"{labels[i]}/{labels[0]}",
                   markersize=4, linewidth=2, color=colors[i % len(colors)])
    ax.axhline(1, color='k', linestyle='--', alpha=0.5, linewidth=2)
    ax.set_xlabel(r'$k$ [s/km]', fontsize=12)
    ax.set_ylabel(r'$P_F(k) / P_F^{\rm ref}(k)$', fontsize=12)
    ax.set_title(f'Power Ratios (ref: {labels[0]})', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def _plot_pairwise_matrix(matrix_result, output_path):
    """Plot pairwise comparison matrix."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    matrix = matrix_result['matrix']
    pvalue_matrix = matrix_result['pvalue_matrix']
    labels = matrix_result['labels']
    
    # Statistic matrix
    im1 = axes[0].imshow(matrix, cmap='viridis', aspect='auto')
    axes[0].set_xticks(np.arange(len(labels)))
    axes[0].set_yticks(np.arange(len(labels)))
    axes[0].set_xticklabels(labels, rotation=45, ha='right')
    axes[0].set_yticklabels(labels)
    axes[0].set_title(f'{matrix_result["metric"].upper()} Test Statistic')
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = axes[0].text(j, i, f'{matrix[i, j]:.3f}',
                              ha='center', va='center', fontsize=9)
    
    plt.colorbar(im1, ax=axes[0])
    
    # P-value matrix
    im2 = axes[1].imshow(np.log10(pvalue_matrix + 1e-100), cmap='RdYlGn', aspect='auto', vmin=-10, vmax=0)
    axes[1].set_xticks(np.arange(len(labels)))
    axes[1].set_yticks(np.arange(len(labels)))
    axes[1].set_xticklabels(labels, rotation=45, ha='right')
    axes[1].set_yticklabels(labels)
    axes[1].set_title('p-values (log10 scale)')
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j:
                sig = '***' if pvalue_matrix[i, j] < 0.001 else ('**' if pvalue_matrix[i, j] < 0.01 else ('*' if pvalue_matrix[i, j] < 0.05 else ''))
                text = axes[1].text(j, i, sig, ha='center', va='center', 
                                  fontsize=14, fontweight='bold', color='red')
    
    plt.colorbar(im2, ax=axes[1], label='log10(p-value)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()



def _load_flux_chunked(filepath, max_samples=100000, chunk_size=1000):
    """
    Load flux data in chunks to avoid memory issues.
    Returns a flattened sample of flux values (max_samples total).
    """
    with h5py.File(filepath, 'r') as f:
        if 'tau/H/1/1215' in f:
            tau_dataset = f['tau/H/1/1215']
            n_sightlines, n_pixels = tau_dataset.shape
            total_pixels = n_sightlines * n_pixels
            
            # Calculate stride to get approximately max_samples
            stride = max(1, total_pixels // max_samples)
            
            flux_samples = []
            for i in range(0, n_sightlines, chunk_size):
                chunk_end = min(i + chunk_size, n_sightlines)
                tau_chunk = tau_dataset[i:chunk_end, ::stride]
                flux_chunk = np.exp(-tau_chunk)
                flux_samples.append(flux_chunk.flatten())
                
                # Stop if we have enough samples
                if len(flux_samples) * flux_chunk.size >= max_samples:
                    break
            
            flux = np.concatenate(flux_samples)
            # Randomly sample to exact size if we have more
            if len(flux) > max_samples:
                indices = np.random.choice(len(flux), max_samples, replace=False)
                flux = flux[indices]
            
            return flux
    
    return np.array([])


def _load_tau_chunked(filepath, chunk_size=1000):
    """
    Generator that yields chunks of tau data.
    Use this for processing that can be done incrementally.
    """
    with h5py.File(filepath, 'r') as f:
        if 'tau/H/1/1215' in f:
            tau_dataset = f['tau/H/1/1215']
            n_sightlines = tau_dataset.shape[0]
            
            for i in range(0, n_sightlines, chunk_size):
                chunk_end = min(i + chunk_size, n_sightlines)
                yield tau_dataset[i:chunk_end]


def extract_spectral_features_chunked(filepath, chunk_size=1000):
    """
    Extract spectral features using chunked processing to save memory.
    """
    from scripts.exploratory import extract_spectral_features
    
    with h5py.File(filepath, 'r') as f:
        if 'tau/H/1/1215' not in f:
            return {}
        
        tau_dataset = f['tau/H/1/1215']
        n_sightlines, n_pixels = tau_dataset.shape
        
        # Process in chunks and accumulate features
        all_features = []
        for i in range(0, n_sightlines, chunk_size):
            chunk_end = min(i + chunk_size, n_sightlines)
            tau_chunk = np.array(tau_dataset[i:chunk_end])
            features_chunk = extract_spectral_features(tau_chunk)
            all_features.append(features_chunk)
        
        # Merge features from all chunks
        merged = {}
        
        # For arrays, concatenate
        for key in ['void_sizes', 'line_widths', 'absorber_separations']:
            if key in all_features[0]:
                merged[key] = np.concatenate([f[key] for f in all_features if key in f and len(f[key]) > 0])
        
        # For scalars, take weighted average or sum
        scalar_keys = ['saturation_fraction', 'deep_absorption_fraction', 'transmission_fraction',
                      'flux_mean', 'flux_variance', 'flux_skewness', 'flux_kurtosis']
        
        for key in scalar_keys:
            if key in all_features[0]:
                merged[key] = float(np.mean([f[key] for f in all_features if key in f]))
        
        # Recompute mean/median from merged arrays
        if 'void_sizes' in merged and len(merged['void_sizes']) > 0:
            merged['mean_void_size'] = float(np.mean(merged['void_sizes']))
            merged['median_void_size'] = float(np.median(merged['void_sizes']))
        
        if 'line_widths' in merged and len(merged['line_widths']) > 0:
            merged['mean_line_width'] = float(np.mean(merged['line_widths']))
            merged['median_line_width'] = float(np.median(merged['line_widths']))
        
        if 'absorber_separations' in merged and len(merged['absorber_separations']) > 0:
            merged['mean_absorber_separation'] = float(np.mean(merged['absorber_separations']))
        else:
            merged['mean_absorber_separation'] = 0.0
        
        return merged


def _create_enhanced_comparison_plot_lazy(all_results, labels, filepaths, output_path):
    """
    Enhanced comparison plot using lazy loading to avoid loading all data at once.
    Loads only what's needed for each panel.
    """
    n_sims = len(all_results)
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.4)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Panel 1: Power spectrum
    ax = fig.add_subplot(gs[0, :])
    for i, res in enumerate(all_results):
        ps = res['power_spectrum']
        k, P_k = ps['k'], ps['P_k_mean']
        mask = k > 0
        ax.loglog(k[mask], P_k[mask], 'o-', color=colors[i % len(colors)], 
                 linewidth=2, markersize=3, label=f"{res['label']}", alpha=0.8)
    ax.set_xlabel(r'$k$ [s/km]', fontsize=12)
    ax.set_ylabel(r'$P_F(k)$ [km/s]', fontsize=12)
    ax.set_title('Flux Power Spectrum', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    
    # Panel 2: τ_eff box plot
    ax = fig.add_subplot(gs[1, 0])
    tau_eff_data = [res['tau_eff']['tau_eff_per_sightline'] for res in all_results]
    bp = ax.boxplot(tau_eff_data, labels=labels, patch_artist=True, showfliers=False)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel(r'$\tau_{\rm eff}$', fontsize=12)
    ax.set_title('Optical Depth Distribution', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # Panel 3: Flux box plot (load only sample for visualization)
    ax = fig.add_subplot(gs[1, 1])
    flux_samples = []
    for fpath in filepaths:
        flux_sample = _load_flux_chunked(fpath, max_samples=10000)
        flux_samples.append(flux_sample)
    
    bp = ax.boxplot(flux_samples, labels=labels, patch_artist=True, showfliers=False)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Flux', fontsize=12)
    ax.set_title('Flux Distribution', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # Panel 4: CDDF
    ax = fig.add_subplot(gs[1, 2])
    for i, res in enumerate(all_results):
        cddf = res['cddf']
        if cddf['n_absorbers'] > 0:
            log_N = np.log10(cddf['bin_centers'])
            counts = cddf['counts']
            mask = counts > 0
            if np.any(mask):
                ax.semilogy(log_N[mask], counts[mask], 'o-', color=colors[i % len(colors)],
                           linewidth=2, markersize=4, label=labels[i], alpha=0.8)
    ax.set_xlabel(r'$\log_{10}(N_{\rm HI}$ [cm$^{-2}$])', fontsize=12)
    ax.set_ylabel('Counts', fontsize=12)
    ax.set_title('Column Density Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # Panel 5: Sample spectra (load one spectrum per simulation)
    ax = fig.add_subplot(gs[2, :])
    for i, fpath in enumerate(filepaths):
        with h5py.File(fpath, 'r') as f:
            if 'tau/H/1/1215' in f:
                # Load just one spectrum
                tau_single = np.array(f['tau/H/1/1215'][0, :])
                flux_single = np.exp(-tau_single)
                vel = np.arange(len(flux_single)) * 0.1  # km/s
                ax.plot(vel, flux_single, color=colors[i % len(colors)],
                       linewidth=1.5, label=labels[i], alpha=0.8)
    
    ax.set_xlabel('Velocity [km/s]', fontsize=12)
    ax.set_ylabel('Flux', fontsize=12)
    ax.set_title('Sample Spectra (first sightline)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.1)
    
    # Panel 6: Mean flux vs z (if available)
    ax = fig.add_subplot(gs[3, 0])
    mean_fluxes = [res['flux_stats']['mean_flux'] for res in all_results]
    redshifts = [res['redshift'] for res in all_results]
    x = np.arange(n_sims)
    ax.bar(x, mean_fluxes, color=[colors[i % len(colors)] for i in range(n_sims)], alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Mean Flux', fontsize=12)
    ax.set_title('Mean Flux Comparison', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # Panel 7: τ_eff comparison
    ax = fig.add_subplot(gs[3, 1])
    tau_eff_means = [res['tau_eff']['tau_eff'] for res in all_results]
    ax.bar(x, tau_eff_means, color=[colors[i % len(colors)] for i in range(n_sims)], alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel(r'$\langle\tau_{\rm eff}\rangle$', fontsize=12)
    ax.set_title('Mean Effective Optical Depth', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # Panel 8: Summary statistics text
    ax = fig.add_subplot(gs[3, 2])
    ax.axis('off')
    summary_text = "Summary Statistics\n" + "="*30 + "\n"
    for i, res in enumerate(all_results):
        summary_text += f"\n{labels[i]}:\n"
        summary_text += f"  z: {res['redshift']:.3f}\n"
        summary_text += f"  N_los: {res['n_sightlines']}\n"
        summary_text += f"  <F>: {res['flux_stats']['mean_flux']:.4f}\n"
        summary_text += f"  τ_eff: {res['tau_eff']['tau_eff']:.3f}\n"
        summary_text += f"  N_abs: {res['cddf']['n_absorbers']}\n"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
           fontsize=8, verticalalignment='top', fontfamily='monospace')
    
    fig.suptitle('Enhanced Simulation Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def compare_distributions_lazy(filepaths, labels, output_path=None, quantity_name='Flux', 
                               max_samples_per_sim=50000):
    """
    Compare distributions using lazy loading to avoid memory issues.
    Loads samples from each file rather than entire datasets.
    """
    from scripts.exploratory import compare_distributions
    
    # Load sampled data
    flux_samples = []
    for fpath in filepaths:
        flux = _load_flux_chunked(fpath, max_samples=max_samples_per_sim)
        flux_samples.append(flux.reshape(-1, 1))  # Reshape to 2D for compare_distributions
    
    # Use existing compare_distributions function
    compare_distributions(flux_samples, labels, output_path, quantity_name)
