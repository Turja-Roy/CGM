import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


def save_analysis_results(results_dict, output_dir, formats=['csv']):
    """Save comprehensive analysis results to JSON and/or CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    created_files = {}
    
    # Add export timestamp
    results_dict['export_info'] = {
        'timestamp': datetime.now().isoformat(),
        'formats': formats
    }
    
    # Export full results to JSON
    if 'json' in formats:
        json_path = output_dir / 'analysis_results.json'
        save_results_json(results_dict, json_path)
        created_files['json'] = json_path
        print(f"Saved analysis results (JSON): {json_path}")
    
    # Export individual data tables to CSV
    if 'csv' in formats:
        csv_files = {}
        
        # Power spectrum
        if 'power_spectrum' in results_dict:
            csv_path = output_dir / 'power_spectrum.csv'
            save_power_spectrum_csv(results_dict['power_spectrum'], csv_path)
            csv_files['power_spectrum'] = csv_path
        
        # CDDF
        if 'cddf' in results_dict:
            csv_path = output_dir / 'cddf.csv'
            save_cddf_csv(results_dict['cddf'], csv_path)
            csv_files['cddf'] = csv_path
        
        # Flux statistics
        if 'flux_stats' in results_dict:
            csv_path = output_dir / 'flux_stats.csv'
            save_flux_stats_csv(results_dict['flux_stats'], csv_path)
            csv_files['flux_stats'] = csv_path
        
        # Line widths
        if 'line_widths' in results_dict and results_dict['line_widths'] is not None:
            csv_path = output_dir / 'line_widths.csv'
            save_line_widths_csv(results_dict['line_widths'], csv_path)
            csv_files['line_widths'] = csv_path
        
        # Temperature-density
        if 'temp_density' in results_dict and results_dict['temp_density'] is not None:
            csv_path = output_dir / 'temp_density.csv'
            save_temp_density_csv(results_dict['temp_density'], csv_path)
            csv_files['temp_density'] = csv_path
        
        # Metal lines
        if results_dict.get('metal_lines') and len(results_dict['metal_lines']) > 0:
            csv_path = output_dir / 'metal_lines.csv'
            save_metal_lines_csv(results_dict['metal_lines'], csv_path)
            csv_files['metal_lines'] = csv_path
        
        created_files['csv'] = csv_files
        print(f"Saved {len(csv_files)} CSV data files to: {output_dir}")
    
    return created_files


def save_results_json(results_dict, output_path):
    """Save analysis results to JSON."""
    json_dict = convert_for_json(results_dict)
    
    with open(output_path, 'w') as f:
        json.dump(json_dict, f, indent=2)


def save_power_spectrum_csv(power_dict, output_path):
    """Save power spectrum data to CSV."""
    data = {
        'k_s_per_km': power_dict['k'],
        'P_k_mean_km_per_s': power_dict['P_k_mean'],
    }
    
    # Add optional columns if available
    if 'P_k_std' in power_dict:
        data['P_k_std'] = power_dict['P_k_std']
    if 'P_k_err' in power_dict:
        data['P_k_err'] = power_dict['P_k_err']
    
    # Add k*P(k)/pi if available
    if 'kPk_pi' in power_dict:
        data['kPk_pi'] = power_dict['kPk_pi']
    if 'kPk_pi_err' in power_dict:
        data['kPk_pi_err'] = power_dict['kPk_pi_err']
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, float_format='%.6e')


def save_cddf_csv(cddf_dict, output_path):
    """Save column density distribution to CSV with proper normalization metadata."""
    log10_N_HI = cddf_dict.get('log10_N_HI', cddf_dict.get('log_bin_edges'))
    f_N = cddf_dict.get('f_N', cddf_dict.get('f_N_HI'))
    
    data = {
        'log10_N_HI': log10_N_HI,
        'f_N_HI': f_N,
        'counts': cddf_dict['counts'],
    }
    
    if 'delta_log_N' in cddf_dict:
        data['delta_log_N'] = cddf_dict['delta_log_N']
    
    if 'bin_centers' in cddf_dict:
        data['bin_center'] = cddf_dict['bin_centers']
    
    df = pd.DataFrame(data)
    
    # Write with metadata comment at the top
    with open(output_path, 'w') as f:
        # Write metadata as comment
        if 'n_sightlines' in cddf_dict:
            f.write(f"# n_sightlines = {cddf_dict['n_sightlines']}\n")

        # dx_mode = 1 makes dX the dimensionless absorption distance X(z), not a
        # comoving length. Both are echoed when available.
        dx_mode = cddf_dict.get('dx_mode')
        if 'dX' in cddf_dict:
            if dx_mode == 1:
                f.write(f"# X = {cddf_dict['dX']:.6e} (absorption distance, dimensionless)\n")
            else:
                f.write(f"# dX = {cddf_dict['dX']:.6f} Mpc (comoving)\n")
        if 'dX_comoving_mpc' in cddf_dict:
            f.write(f"# dX_comoving_mpc = {cddf_dict['dX_comoving_mpc']:.6f}\n")
        if 'X_absorption' in cddf_dict:
            f.write(f"# X_absorption = {cddf_dict['X_absorption']:.6e}\n")

        if 'redshift' in cddf_dict:
            f.write(f"# redshift = {cddf_dict['redshift']:.6f}\n")
        if 'n_absorbers' in cddf_dict:
            f.write(f"# n_absorbers = {cddf_dict['n_absorbers']}\n")

        # Configuration echo, so a CDDF file is self-describing and files written
        # under different settings cannot be silently compared. See
        # config.CDDF_OPTIONS.
        for key in ('absorber_mode', 'cell_dv', 'colden_mode', 'norm_mode',
                    'log_N_min', 'log_N_max', 'n_bins',
                    'fit_log_N_min', 'fit_log_N_max', 'min_N_gate'):
            if key in cddf_dict:
                f.write(f"# {key} = {cddf_dict[key]}\n")
        if dx_mode is not None:
            f.write(f"# dx_mode = {dx_mode}\n")

        if 'beta_fit' in cddf_dict and not np.isnan(cddf_dict['beta_fit']):
            f.write(f"# beta_fit = {cddf_dict['beta_fit']:.6f}\n")
        if cddf_dict.get('saturated'):
            # beta_fit is deliberately absent here; consumers read it with
            # .get(..., nan), so it degrades to "no fit".
            f.write("# saturated = True  (forest saturated; beta_fit suppressed)\n")
            raw = cddf_dict.get('beta_fit_raw')
            if raw is not None and not np.isnan(raw):
                f.write(f"# beta_fit_raw = {raw:.6f}  (DO NOT USE -- unphysical)\n")

        # norm_mode = 1 gives true f(N) = dn/dN dX in cm^2; mode 0 gives the
        # per-dex quantity N f(N) ln10, which fits to beta_true - 1.
        if cddf_dict.get('norm_mode') == 1:
            f.write("# f_N_HI units: [cm^2]  (f(N) = dn/dN dX)\n")
        else:
            f.write("# f_N_HI units: [Mpc^-1] (comoving)\n")
        f.write("#\n")
        
        # Write the data
        df.to_csv(f, index=False, float_format='%.6e')


def save_flux_stats_csv(stats_dict, output_path):
    """Save flux statistics to CSV."""
    # Convert dict to two-column format: statistic, value
    data = {
        'statistic': list(stats_dict.keys()),
        'value': list(stats_dict.values())
    }
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)


def save_line_widths_csv(lwd_dict, output_path):
    """Save line width distribution to CSV."""
    if lwd_dict['n_absorbers'] == 0:
        # Create empty file with headers
        df = pd.DataFrame(columns=['N_HI', 'b_param_km_s'])
        df.to_csv(output_path, index=False)
        return
    
    data = {
        'N_HI': lwd_dict['N_HI'],
        'b_param_km_s': lwd_dict['b_params'],
    }
    
    # Add optical depth if available
    if 'tau_peak' in lwd_dict:
        data['tau_peak'] = lwd_dict['tau_peak']
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, float_format='%.6e')


def save_temp_density_csv(tdens_dict, output_path):
    """Save temperature-density relation data to CSV."""
    # Save raw scatter data (subsampled for size)
    log_T = tdens_dict.get('log_T', np.array([]))
    log_rho = tdens_dict.get('log_rho', np.array([]))
    
    if len(log_T) == 0 or len(log_rho) == 0:
        # No data to save
        with open(output_path, 'w') as f:
            f.write("# No temperature-density data available\n")
            f.write(f"# T0 = {tdens_dict.get('T0', np.nan)} K\n")
            f.write(f"# gamma = {tdens_dict.get('gamma', np.nan)}\n")
            f.write(f"# gamma_err = {tdens_dict.get('gamma_err', np.nan)}\n")
            f.write("log_density,log_temperature\n")
        return
    
    # Subsample if too large (save every 10th point for scatter plots)
    if len(log_T) > 10000:
        stride = len(log_T) // 10000
        log_T = log_T[::stride]
        log_rho = log_rho[::stride]
    
    data = {
        'log_density': log_rho,
        'log_temperature': log_T,
    }
    
    df = pd.DataFrame(data)
    
    # Add fit parameters as comments in CSV header
    with open(output_path, 'w') as f:
        f.write(f"# T0 = {tdens_dict.get('T0', np.nan)} K\n")
        f.write(f"# gamma = {tdens_dict.get('gamma', np.nan)}\n")
        f.write(f"# gamma_err = {tdens_dict.get('gamma_err', np.nan)}\n")
        f.write(f"# n_pixels = {tdens_dict.get('n_pixels', len(log_T))}\n")
        df.to_csv(f, index=False, float_format='%.6e')


def save_metal_lines_csv(metal_lines_list, output_path):
    """Save multi-line statistics to CSV."""
    # Extract key statistics from each line
    rows = []
    for stats in metal_lines_list:
        row = {
            'ion_name': stats['ion_name'],
            'n_absorbers': stats['n_absorbers'],
            'dN_dz': stats.get('dN_dz', np.nan),
            'covering_fraction': stats.get('covering_fraction', np.nan),
            'mean_tau': stats.get('mean_tau', np.nan),
            'median_tau': stats.get('median_tau', np.nan),
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, float_format='%.6e')


def convert_for_json(obj):
    """Convert numpy arrays and other non-JSON types to JSON-compatible types."""
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_for_json(item) for item in obj]
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return obj


def get_analysis_output_dir(spectra_file, suite=None, sim_set=None, sim_name=None, snap_num=None):
    """Get output directory for analysis results based on file hierarchy."""
    import scripts.config as config
    
    # Try to extract info from filepath if not provided
    if suite is None or sim_set is None or sim_name is None or snap_num is None:
        info = config.extract_simulation_info(spectra_file)
        suite = suite or info['suite']
        sim_set = sim_set or info['sim_set']
        sim_name = sim_name or info['sim_name']
        snap_num = snap_num or info['snap_num']
    
    # Create path: output/analysis/{suite}/{sim_set}/{sim_name}/snap-{N}/
    output_dir = config.OUTPUT_DIR / 'analysis' / suite / sim_set / sim_name / f'snap-{snap_num}'
    
    return output_dir
