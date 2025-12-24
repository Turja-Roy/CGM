"""
Simulation comparison and redshift evolution tracking.

Functions for loading spectra results, comparing multiple simulations,
and tracking how observables evolve with redshift.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from .analysis import (
    compute_flux_statistics,
    compute_effective_optical_depth,
    compute_power_spectrum,
    compute_column_density_distribution,
    compute_line_width_distribution,
    compute_temperature_density_relation,
)
from .plotting import save_plot


def load_spectra_results(spectra_file, velocity_spacing=0.1):
    """
    Load and compute all analysis results from a spectra file.
    
    Parameters
    ----------
    spectra_file : str or Path
        Path to HDF5 spectra file
    velocity_spacing : float
        Velocity spacing in km/s (default: 0.1)
    
    Returns
    -------
    results : dict
        Dictionary containing all analysis results:
        - 'success': bool
        - 'filepath': str
        - 'redshift': float
        - 'n_sightlines': int
        - 'n_pixels': int
        - 'flux_stats': dict
        - 'tau_eff': dict
        - 'power_spectrum': dict
        - 'cddf': dict
        - 'line_widths': dict or None
        - 'temp_density': dict or None
        - 'error': str (if failed)
    """
    results = {
        'filepath': str(spectra_file),
        'success': False,
    }
    
    try:
        with h5py.File(spectra_file, 'r') as f:
            # Get redshift
            redshift = None
            if 'Header' in f:
                header = f['Header'].attrs
                redshift = header.get('redshift', header.get('Redshift', None))
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
            
            # Compute all analyses
            results['flux_stats'] = compute_flux_statistics(tau)
            results['tau_eff'] = compute_effective_optical_depth(tau)
            results['power_spectrum'] = compute_power_spectrum(flux, velocity_spacing)
            results['cddf'] = compute_column_density_distribution(tau, velocity_spacing)
            
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
    """
    Compare results from multiple simulation runs.
    
    Parameters
    ----------
    spectra_files : list of str
        List of paths to spectra HDF5 files
    labels : list of str, optional
        Labels for each simulation (default: "Sim 0", "Sim 1", ...)
    output_path : str or Path, optional
        Where to save comparison plot
    
    Returns
    -------
    comparison : dict
        Dictionary containing comparison results:
        - 'n_simulations': int
        - 'labels': list of str
        - 'results': list of dicts (all loaded results)
    """
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
            # Normalize counts to get f(N_HI) - number density per bin
            log_N = cddf['bin_centers']
            # f(N) in units of dN/dlog(N)/dz (absorbers per unit log column density per unit redshift)
            dN_dz = cddf['n_absorbers'] / res['n_sightlines'] / 0.1  # Rough dz estimate
            delta_log_N = log_N[1] - log_N[0] if len(log_N) > 1 else 1.0
            f_N = cddf['counts'] / (cddf['n_absorbers'] if cddf['n_absorbers'] > 0 else 1) / delta_log_N
            
            # Only plot non-zero bins
            mask = f_N > 0
            if np.any(mask):
                ax.scatter(log_N[mask], f_N[mask],
                          s=30, alpha=0.6, color=color, label=res['label'])
                
                # Plot fit if available
                if not np.isnan(cddf['beta_fit']):
                    N_fit = np.logspace(12, 16, 100)
                    # Simple power law for visualization
                    f_fit = f_N[mask].max() * (N_fit / 10**log_N[mask][np.argmax(f_N[mask])]) ** cddf['beta_fit']
                    ax.plot(np.log10(N_fit), f_fit, '--', color=color, alpha=0.5, linewidth=1.5)
    
    ax.set_xlabel(r'$\log_{10}(N_{\rm HI} / {\rm cm}^{-2})$', fontsize=12)
    ax.set_ylabel(r'$f(N_{\rm HI})$', fontsize=12)
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
    """
    Track how observables evolve with redshift across multiple snapshots.
    
    Parameters
    ----------
    spectra_files : list of str
        List of paths to spectra HDF5 files (from different redshifts)
    labels : list of str, optional
        Labels for each snapshot (default: "Snap 0", "Snap 1", ...)
    output_path : str or Path, optional
        Where to save evolution plot
    
    Returns
    -------
    evolution : dict
        Dictionary containing evolution results:
        - 'n_snapshots': int
        - 'redshift_range': tuple (z_min, z_max)
        - 'redshifts': list
        - 'tau_eff': list
        - 'mean_flux': list
        - 'n_absorbers': list
        - 'T0': list (if available)
        - 'gamma': list (if available)
        - 'z_with_T': list (if available)
        - 'results': list of dicts
    """
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
