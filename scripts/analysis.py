import numpy as np
import os
from pathlib import Path
from astropy.cosmology import FlatLambdaCDM

from scripts import analysis_cpp


#######################
# COSMOLOGY UTILITIES #
#######################

def get_cosmology(omega_m=0.3089, omega_lambda=0.6911, h=0.6774):
    return FlatLambdaCDM(H0=100*h, Om0=omega_m)


def compute_absorption_path_length(redshift, box_size_ckpc_h, hubble=0.6774, 
                                   omega_m=0.3089, use_cosmology=True):
    """
    Notes:

    For a periodic box, the absorption path length is simply:
        dX = box_size / h  [comoving Mpc]
    
    The proper distance would be dX / (1+z), but for CDDF we use comoving.
    """
    if use_cosmology:
        # Convert ckpc/h to Mpc (comoving)
        dX = box_size_ckpc_h / hubble / 1000.0  # Mpc
    else:
        # Simple conversion
        dX = box_size_ckpc_h / hubble / 1000.0  # Mpc
    
    return dX


#############################
# DATA PROCESSING FUNCTIONS #
#############################

def compute_flux_statistics(tau):
    return analysis_cpp.compute_flux_statistics(tau)


def compute_power_spectrum(flux, velocity_spacing, chunk_size=1000):
    return analysis_cpp.compute_power_spectrum(flux, velocity_spacing, chunk_size)


def ensure_directory(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_snapshot_number(filepath):
    """Extract snapshot number from filepath."""
    basename = os.path.basename(filepath)
    if 'snap_' in basename:
        num_str = basename.split('snap_')[1].split('.')[0]
        return int(num_str)
    return None


def compute_column_density_distribution(tau, velocity_spacing, threshold=0.5, colden=None,
                                       redshift=None, box_size_ckpc_h=None, hubble=0.6774,
                                       omega_m=0.3089, **options):
    """Compute f(N_HI). See analysis_cpp.compute_column_density_distribution for the
    optional mode arguments (absorber_mode, colden_mode, dx_mode, norm_mode, ...);
    production values are in config.CDDF_OPTIONS."""
    return analysis_cpp.compute_column_density_distribution(
        tau, velocity_spacing, threshold, colden,
        redshift, box_size_ckpc_h, hubble, omega_m, **options
    )


def compute_effective_optical_depth(tau):
    """tau_eff = -ln<F>, with sightline-to-sightline errors.

    Errors are internal to one box: no cosmic variance. *_std is the per-sightline
    spread, *_err is that over sqrt(N).
    """
    flux = np.exp(-tau)

    # Global tau_eff
    mean_flux = np.mean(flux, dtype=np.float64)
    tau_eff = -np.log(mean_flux)

    # Per-sightline tau_eff
    mean_flux_per_los = np.mean(flux, axis=1, dtype=np.float64)
    tau_eff_per_los = -np.log(mean_flux_per_los)
    n_los = len(tau_eff_per_los)

    return {
        'tau_eff': float(tau_eff),
        'mean_flux': float(mean_flux),
        'tau_eff_per_sightline': tau_eff_per_los,
        'tau_eff_std': float(np.std(tau_eff_per_los)),
        'tau_eff_err': float(np.std(tau_eff_per_los) / np.sqrt(n_los)),
        'mean_flux_std': float(np.std(mean_flux_per_los)),
        'mean_flux_err': float(np.std(mean_flux_per_los) / np.sqrt(n_los)),
        'n_sightlines': int(n_los),
    }


##############################
# FLUX / OPTICAL DEPTH PDFs  #
##############################

# Fixed grids: PDFs are only comparable bin-for-bin across redshift and cosmology if
# the bins never move. tau is binned in log10 because the forest spans four decades.
FLUX_PDF_BINS = 50
FLUX_PDF_RANGE = (0.0, 1.0)
LOG_TAU_PDF_BINS = 60
LOG_TAU_PDF_RANGE = (-3.0, 2.0)


def _pdf_from_counts(counts, edges, n_total):
    """Normalised density and its Poisson error, given raw bin counts."""
    counts = counts.astype(np.float64)
    widths = np.diff(edges)
    density = counts / (n_total * widths)
    density_err = np.sqrt(counts) / (n_total * widths)
    return density, density_err


def compute_flux_tau_pdf(tau, flux=None, chunk_size=500):
    """Flux PDF and log10(tau) PDF over every pixel, with Poisson errors.

    Chunked over sightlines to avoid another full-size copy (tau is ~400 MB at
    1e4 x 1e4 float32). Counts are exact, not subsampled.
    """
    n_sightlines = tau.shape[0]

    flux_counts = np.zeros(FLUX_PDF_BINS, dtype=np.int64)
    tau_counts = np.zeros(LOG_TAU_PDF_BINS, dtype=np.int64)
    flux_edges = np.linspace(*FLUX_PDF_RANGE, FLUX_PDF_BINS + 1)
    tau_edges = np.linspace(*LOG_TAU_PDF_RANGE, LOG_TAU_PDF_BINS + 1)

    n_total = 0
    # Out-of-grid pixels are counted, not folded into the edge bins.
    n_tau_underflow = 0
    n_tau_overflow = 0
    n_tau_zero = 0

    for start in range(0, n_sightlines, chunk_size):
        stop = min(start + chunk_size, n_sightlines)
        tau_chunk = tau[start:stop].ravel()

        if flux is None:
            flux_chunk = np.exp(-tau_chunk.astype(np.float64))
        else:
            flux_chunk = flux[start:stop].ravel()

        flux_counts += np.histogram(flux_chunk, bins=flux_edges)[0]

        positive = tau_chunk > 0
        n_tau_zero += int(tau_chunk.size - np.count_nonzero(positive))
        log_tau = np.log10(tau_chunk[positive].astype(np.float64))
        tau_counts += np.histogram(log_tau, bins=tau_edges)[0]
        n_tau_underflow += int(np.count_nonzero(log_tau < LOG_TAU_PDF_RANGE[0]))
        n_tau_overflow += int(np.count_nonzero(log_tau > LOG_TAU_PDF_RANGE[1]))

        n_total += tau_chunk.size

    n_tau_in_grid = int(tau_counts.sum())

    flux_density, flux_density_err = _pdf_from_counts(flux_counts, flux_edges, n_total)
    # Normalised over in-grid pixels only; the fractions below say what was left out.
    tau_density, tau_density_err = _pdf_from_counts(
        tau_counts, tau_edges, max(n_tau_in_grid, 1))

    return {
        'flux_bin_edges': flux_edges,
        'flux_bin_centers': 0.5 * (flux_edges[1:] + flux_edges[:-1]),
        'flux_counts': flux_counts,
        'flux_density': flux_density,
        'flux_density_err': flux_density_err,
        'log_tau_bin_edges': tau_edges,
        'log_tau_bin_centers': 0.5 * (tau_edges[1:] + tau_edges[:-1]),
        'log_tau_counts': tau_counts,
        'log_tau_density': tau_density,
        'log_tau_density_err': tau_density_err,
        'n_pixels': n_total,
        'n_tau_in_grid': n_tau_in_grid,
        'frac_tau_zero': n_tau_zero / n_total if n_total else np.nan,
        'frac_tau_underflow': n_tau_underflow / n_total if n_total else np.nan,
        'frac_tau_overflow': n_tau_overflow / n_total if n_total else np.nan,
    }


#########################
# MEAN FLUX RESCALING   #
#########################

# Kim, Bolton, Viel, Haehnelt & Carswell (2007), MNRAS 382, 1657 (arXiv:0711.1862):
#   tau_eff = (0.0023 +/- 0.0007) (1+z)^(3.65 +/- 0.21), fitted over 1.7 < z < 4.
KIM07_TAU_EFF_A = 0.0023
KIM07_TAU_EFF_GAMMA = 3.65
KIM07_Z_MIN = 1.7
KIM07_Z_MAX = 4.0


def observed_tau_eff(redshift):
    """Observed tau_eff(z), NaN outside 1.7 < z < 4 rather than extrapolated.

    No low-z branch: checked 2026-08-22, neither Danforth+2016 (arXiv:1402.2655,
    a dN/dz catalogue) nor Khaire+2019 (arXiv:1808.05605, P_F(k)) publishes a
    tau_eff(z) fit. Both stop at z = 0.47 anyway, so snaps 050 (z=1.60) and 060
    (z=1.05) would stay in a gap regardless -- don't bridge it by extrapolating.
    """
    if redshift is None or not np.isfinite(redshift):
        return float('nan')
    if redshift < KIM07_Z_MIN or redshift > KIM07_Z_MAX:
        return float('nan')
    return float(KIM07_TAU_EFF_A * (1.0 + redshift) ** KIM07_TAU_EFF_GAMMA)


def rescale_to_mean_flux(tau, mean_flux_desired, tol=1e-6, max_iter=50,
                         chunk_size=500):
    """Solve <exp(-A tau)> = mean_flux_desired for the scalar A, by Newton-Raphson.

    Absorbs the UVB-amplitude uncertainty: only the shape of tau is trusted, not its
    normalisation. Same result as fake_spectra.fluxstatistics.mean_flux(), but chunked
    -- that one casts the whole array to float64 (+800 MB at 1e4 x 1e4).

    NaN target (z outside the Kim+07 range) returns NaN.
    """
    if mean_flux_desired is None or not np.isfinite(mean_flux_desired):
        return float('nan')
    if not (0.0 < mean_flux_desired < 1.0):
        return float('nan')

    n_sightlines = tau.shape[0]

    def _mean_flux_and_derivative(scale):
        # d<F>/dA = -<tau exp(-A tau)>
        total_f = 0.0
        total_df = 0.0
        n = 0
        for start in range(0, n_sightlines, chunk_size):
            stop = min(start + chunk_size, n_sightlines)
            t = tau[start:stop].ravel().astype(np.float64)
            f = np.exp(-scale * t)
            total_f += f.sum()
            total_df -= (t * f).sum()
            n += t.size
        return total_f / n, total_df / n

    scale = 1.0
    for _ in range(max_iter):
        mf, dmf = _mean_flux_and_derivative(scale)
        residual = mf - mean_flux_desired
        if abs(residual) < tol:
            return float(scale)
        if dmf == 0.0:
            break
        new_scale = scale - residual / dmf
        # The scaling is positive by construction; a step through zero means the
        # target is far, so halve instead of diverging.
        scale = new_scale if new_scale > 0 else scale / 2.0

    return float(scale)


def compute_line_width_distribution(tau, velocity_spacing, threshold=0.5, colden=None):
    """Compute line width (Doppler b-parameter) distribution."""
    return analysis_cpp.compute_line_width_distribution(tau, velocity_spacing, threshold, colden)


def compute_temperature_density_relation(temperature, density, tau, min_tau=0.1):
    return analysis_cpp.compute_temperature_density_relation(temperature, density, tau, min_tau)


# Mirrors src/cpp/analysis/constants.h:9. Keep the two in sync.
TAU_TO_COLDEN_CONSTANT = 8.51e11  # cm^-2 / (km/s)


def compute_metal_line_statistics(tau, velocity_spacing, ion_name='Metal', threshold=0.05, colden=None):
    """Compute statistics for metal line absorption systems."""
    n_sightlines, n_pixels = tau.shape

    # Covering fraction: fraction of pixels with detectable absorption
    covering_fraction = np.sum(tau > threshold) / tau.size

    # Mean/median tau in absorbing regions
    tau_absorbing = tau[tau > threshold]
    if len(tau_absorbing) > 0:
        mean_tau_abs = np.mean(tau_absorbing)
        median_tau_abs = np.median(tau_absorbing)
    else:
        mean_tau_abs = 0.0
        median_tau_abs = 0.0

    # Count absorption systems
    column_densities = []
    n_systems = 0

    for i in range(n_sightlines):
        tau_line = tau[i, :]
        colden_line = colden[i, :] if colden is not None else None

        # Find absorption features
        absorbing = tau_line > threshold

        # Count connected regions
        in_feature = False
        feature_start = 0

        for j in range(len(tau_line)):
            if absorbing[j] and not in_feature:
                # Start of new feature
                in_feature = True
                feature_start = j
                n_systems += 1
            elif not absorbing[j] and in_feature:
                # End of feature
                in_feature = False
                
                # colden is per-pixel, so N is the sum over the absorber --
                # column_density.cpp's COLDEN_SUM, i.e. colden_mode = 1.
                if colden_line is not None:
                    N_ion = np.sum(colden_line[feature_start:j])
                else:
                    feature_tau = tau_line[feature_start:j]
                    N_ion = TAU_TO_COLDEN_CONSTANT * np.sum(feature_tau) * velocity_spacing
                column_densities.append(N_ion)

        # Handle case where feature extends to edge
        if in_feature:
            if colden_line is not None:
                N_ion = np.sum(colden_line[feature_start:])
            else:
                feature_tau = tau_line[feature_start:]
                N_ion = TAU_TO_COLDEN_CONSTANT * np.sum(feature_tau) * velocity_spacing
            column_densities.append(N_ion)

    column_densities = np.array(column_densities)

    # Compute dN/dz (line density per unit redshift)
    # Assuming each sightline samples ~dz = 0.1 (rough estimate)
    # For more accurate dN/dz, need actual path length
    dN_dz = n_systems / n_sightlines / 0.1 if n_sightlines > 0 else 0.0

    # Equivalent width (in velocity space, convert to Angstroms later)
    # EW ~ integral (1 - exp(-tau)) dv
    flux = np.exp(-tau)
    equivalent_width_vel = np.sum(1.0 - flux) * velocity_spacing  # km/s

    return {
        'ion_name': ion_name,
        'n_absorbers': n_systems,
        'n_sightlines': n_sightlines,
        'dN_dz': float(dN_dz),
        'covering_fraction': float(covering_fraction),
        'mean_tau': float(mean_tau_abs),
        'median_tau': float(median_tau_abs),
        # Per sightline
        'equivalent_width_vel': float(equivalent_width_vel / n_sightlines),
        'column_densities': column_densities,
        'log_N_mean': float(np.log10(np.mean(column_densities))) if len(column_densities) > 0 else np.nan,
        'log_N_median': float(np.log10(np.median(column_densities))) if len(column_densities) > 0 else np.nan
    }


def format_stats_table(stats):
    """Format flux statistics as a text table."""
    lines = [
        "Statistics:",
        "=" * 40,
        f"Mean flux:              {stats['mean_flux']:.4f}",
        f"Median flux:            {stats['median_flux']:.4f}",
        f"Std dev:                {stats['std_flux']:.4f}",
        "",
        f"Effective tau_eff:        {stats['effective_tau']:.4f}",
        f"Mean tau:                 {stats['mean_tau']:.4f}",
        f"Median tau:               {stats['median_tau']:.4f}",
        "",
        f"Deep absorption:        {stats['deep_absorption_frac']*100:.2f}%",
        f"Moderate absorption:    {
            stats['moderate_absorption_frac']*100:.2f}%",
        f"Weak absorption:        {stats['weak_absorption_frac']*100:.2f}%",
    ]
    return "\n".join(lines)


####################################
# ADVANCED ANALYSIS USING VoigtFit #
####################################

# Compute column density distribution using VoigtFit profile fitting.
def compute_column_density_distribution_vpfit(flux, wavelength, redshift, threshold=0.05,
                                              continuum_window=50, min_snr=5.0,
                                              output_dir=None):
    try:
        import VoigtFit
        from astropy import units as u
        import astropy.constants as const
    except ImportError:
        print("ERROR: VoigtFit not installed. Install with: pip install VoigtFit")
        return {'error': 'VoigtFit not available'}

    # Convert wavelength to velocity space (km/s) relative to Lyman-alpha
    lambda_lya = 1215.67  # Angstroms
    lambda_rest = lambda_lya * (1 + redshift)

    # Convert wavelength to velocity (km/s)
    c = 299792.458  # km/s
    velocity = c * (wavelength - lambda_rest) / lambda_rest

    # Convert flux to optical depth
    tau = -np.log(flux)

    column_densities = []
    b_parameters = []
    absorber_redshifts = []
    errors_N = []
    errors_b = []
    chi_squared_values = []

    # Process each sightline
    for i in range(flux.shape[0]):
        flux_line = flux[i, :]
        tau_line = tau[i, :]

        # Find absorption features
        absorbers = _find_absorption_features(tau_line, velocity, threshold)

        for absorber in absorbers:
            try:
                # Fit Voigt profile to this absorber
                result = _fit_single_absorber(flux_line, wavelength, absorber,
                                              redshift, continuum_window, min_snr)

                if result is not None:
                    column_densities.append(result['N_HI'])
                    b_parameters.append(result['b'])
                    absorber_redshifts.append(result['z_abs'])
                    errors_N.append(result['N_err'])
                    errors_b.append(result['b_err'])
                    chi_squared_values.append(result['chi2'])

            except Exception as e:
                print(f"Warning: Failed to fit absorber in sightline {i}: {e}")
                continue

    # Convert to numpy arrays
    column_densities = np.array(column_densities)
    b_parameters = np.array(b_parameters)
    absorber_redshifts = np.array(absorber_redshifts)
    errors_N = np.array(errors_N)
    errors_b = np.array(errors_b)
    chi_squared_values = np.array(chi_squared_values)

    return {
        'N_HI': column_densities,
        'b_params': b_parameters,
        'redshifts': absorber_redshifts,
        'errors_N': errors_N,
        'errors_b': errors_b,
        'chi_squared': chi_squared_values,
        'n_absorbers': len(column_densities),
        'method': 'VoigtFit'
    }


# Find absorption features in optical depth spectrum.
def _find_absorption_features(tau, velocity, threshold):
    absorbers = []

    # Find contiguous regions above threshold
    in_feature = False
    start_idx = 0

    for i in range(len(tau)):
        if tau[i] > threshold and not in_feature:
            # Start of feature
            in_feature = True
            start_idx = i
        elif tau[i] <= threshold and in_feature:
            # End of feature
            end_idx = i - 1

            # Check if feature is significant
            if end_idx - start_idx >= 3:  # At least 3 pixels
                absorbers.append({
                    'vel_start': velocity[start_idx],
                    'vel_end': velocity[end_idx],
                    'tau_max': np.max(tau[start_idx:end_idx+1]),
                    'pixel_start': start_idx,
                    'pixel_end': end_idx
                })

            in_feature = False

    # Handle feature at end
    if in_feature and len(tau) - start_idx >= 3:
        absorbers.append({
            'vel_start': velocity[start_idx],
            'vel_end': velocity[-1],
            'tau_max': np.max(tau[start_idx:]),
            'pixel_start': start_idx,
            'pixel_end': len(tau) - 1
        })

    return absorbers


# Fit a single absorber with VoigtFit.
def _fit_single_absorber(flux, wavelength, absorber, redshift, continuum_window, min_snr):
    try:
        from VoigtFit import VoigtFitter
        import astropy.units as u

        # Extract region around absorber
        pixel_start = max(0, absorber['pixel_start'] - continuum_window)
        pixel_end = min(len(flux), absorber['pixel_end'] + continuum_window)

        flux_region = flux[pixel_start:pixel_end]
        wave_region = wavelength[pixel_start:pixel_end]

        # Create VoigtFit spectrum object
        spec = VoigtFitter(wave_region, flux_region)

        # Estimate continuum (simple linear fit to edges)
        edge_size = min(10, len(flux_region)//4)
        left_flux = np.mean(flux_region[:edge_size])
        right_flux = np.mean(flux_region[-edge_size:])
        continuum = np.linspace(left_flux, right_flux, len(flux_region))
        spec.set_continuum(continuum)

        # Estimate initial parameters
        center_vel = (absorber['vel_start'] + absorber['vel_end']) / 2
        center_wave = 1215.67 * (1 + redshift) * (1 + center_vel/299792.458)

        # Rough column density estimate from optical depth
        tau_max = absorber['tau_max']
        dv = np.abs(absorber['vel_end'] - absorber['vel_start'])
        N_est = 1.13e14 * tau_max * dv  # Simplified estimate

        # Rough b estimate from width
        b_est = dv / 2.355  # FWHM to sigma conversion

        # Add HI Lyman-alpha component
        spec.add_line('HI1215', center_wave, N_est, b_est, z=redshift)

        # Fit the spectrum
        spec.fit()

        # Extract results
        if spec.lines:
            line = spec.lines[0]  # First (and only) line

            # Check fit quality
            chi2 = spec.chi2
            if chi2 > 100:  # Poor fit
                return None

            return {
                'N_HI': line.N.value,
                'b': line.b.value,
                'z_abs': line.z,
                'N_err': line.N.std if hasattr(line.N, 'std') else 0.1,
                'b_err': line.b.std if hasattr(line.b, 'std') else 5.0,
                'chi2': chi2
            }

    except Exception as e:
        print(f"VoigtFit error: {e}")
        return None


def compute_tdens_binned(temperature, density, n_bins=30):
    """Compute binned temperature-density statistics (memory-efficient).
    
    Args:
        temperature: 1D array of temperature values (filtered)
        density: 1D array of density values (filtered, same size)
        n_bins: Number of density bins (default 30)
    
    Returns:
        dict with T0, gamma, rho_mean, n_pixels, T_median, rho_centers, counts_per_bin
    """
    return analysis_cpp.compute_tdens_binned(temperature, density, n_bins)


def compute_temperature_density_chunked(
    spectra_file,
    tau_path,
    min_tau=0.1,
    chunk_size=1000,
    n_bins=30,
    verbose=True
):
    """Compute temperature-density relation in chunks to limit memory.
    
    Streams through HDF5 datasets in chunks, filters valid pixels (tau > min_tau,
    valid temperature and density), and computes binned statistics.
    
    Args:
        spectra_file: Path to HDF5 spectra file
        tau_path: Path to tau dataset (e.g., 'tau/H/1/1215')
        min_tau: Minimum optical depth for filtering (default 0.1)
        chunk_size: Number of sightlines per chunk (default 1000)
        n_bins: Number of density bins for T-ρ relation (default 30)
        verbose: Print progress messages (default True)
    
    Returns:
        dict with T0, gamma, rho_mean, n_pixels, T_median, rho_centers, counts_per_bin
    """
    import h5py
    
    temp_elem = 'H'
    temp_ion = '1'
    if '/' in tau_path:
        parts = tau_path.split('/')
        if len(parts) >= 3:
            temp_elem = parts[1]
            temp_ion = parts[2]
    
    with h5py.File(spectra_file, 'r') as f:
        tau_shape = f[tau_path].shape
        n_sightlines = tau_shape[0]
        n_pixels = tau_shape[1]
    
    if verbose:
        print(f"Processing T-ρ data: {n_sightlines:,} sightlines × {n_pixels:,} pixels")
        print(f"Using chunk size: {chunk_size}")
    
    temp_filtered = []
    dens_filtered = []
    
    n_chunks = (n_sightlines + chunk_size - 1) // chunk_size
    print_every = max(1, n_chunks // 10)
    
    with h5py.File(spectra_file, 'r') as f:
        temp_path = f'temperature/{temp_elem}/{temp_ion}'
        dens_path = f'density_weight_density/{temp_elem}/{temp_ion}'
        
        if temp_path not in f or dens_path not in f:
            return {'error': 'Temperature or density data not found'}
        
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, n_sightlines)
            
            if verbose and (chunk_idx % print_every == 0 or chunk_idx == n_chunks - 1):
                print(f"  Processing chunk {chunk_idx + 1}/{n_chunks} ({start_idx:,}-{end_idx:,})")
            
            tau_chunk = f[tau_path][start_idx:end_idx]
            temp_chunk = f[temp_path][start_idx:end_idx]
            dens_chunk = f[dens_path][start_idx:end_idx]
            
            tau_flat = tau_chunk.ravel()
            temp_flat = temp_chunk.ravel()
            dens_flat = dens_chunk.ravel()
            
            mask = (
                (tau_flat > min_tau) &
                (temp_flat > 0) & np.isfinite(temp_flat) &
                (dens_flat > 0) & np.isfinite(dens_flat)
            )
            
            if np.any(mask):
                temp_filtered.append(temp_flat[mask].astype(np.float64))
                dens_filtered.append(dens_flat[mask].astype(np.float64))
    
    if len(temp_filtered) == 0:
        return {
            'T0': float('nan'),
            'gamma': float('nan'),
            'rho_mean': float('nan'),
            'n_pixels': 0,
            'T_median': np.array([]),
            'rho_centers': np.array([]),
            'counts_per_bin': np.array([]),
            'n_bins': n_bins,
            'log_T': np.array([]),
            'log_rho': np.array([]),
        }
    
    temperature = np.concatenate(temp_filtered)
    density = np.concatenate(dens_filtered)
    
    if verbose:
        print(f"  Total valid pixels: {len(temperature):,}")
    
    del temp_filtered, dens_filtered
    
    result = compute_tdens_binned(temperature, density, n_bins)
    rho_mean = result['rho_mean']

    return {
        'T0': result['T0'],
        'gamma': result['gamma'],
        # .get(): absent until the extension is rebuilt.
        'gamma_err': result.get('gamma_err', float('nan')),
        'T0_err': result.get('T0_err', float('nan')),
        'n_bins_fit': result.get('n_bins_fit', -1),
        'rho_mean': rho_mean,
        'n_pixels': result['n_pixels'],
        'T_median': np.array(result['T_median']),
        'rho_centers': np.array(result['rho_centers']),
        'counts_per_bin': np.array(result['counts_per_bin']),
        'n_bins': result['n_bins'],
        # Per-pixel arrays for the 2D phase-diagram plot (Option A).
        # temperature/density are already concatenated/resident here, so
        # this only adds two log10 copies (~free vs the streaming I/O).
        'log_T': np.log10(temperature),
        'log_rho': np.log10(density / rho_mean),
    }
