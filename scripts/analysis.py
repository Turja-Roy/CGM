import numpy as np
import os
from pathlib import Path


# ================#
# DATA PROCESSING #
# ================#

# Compute flux statistics from optical depth.
def compute_flux_statistics(tau):
    flux = np.exp(-tau)

    stats = {
        'mean_flux': float(np.mean(flux)),
        'median_flux': float(np.median(flux)),
        'std_flux': float(np.std(flux)),
        'min_flux': float(np.min(flux)),
        'max_flux': float(np.max(flux)),
        'mean_tau': float(np.mean(tau)),
        'median_tau': float(np.median(tau)),
        'effective_tau': float(-np.log(np.mean(flux))),
    }

    # Absorption statistics
    total_pixels = flux.size
    stats['deep_absorption_frac'] = float((flux < 0.1).sum() / total_pixels)
    stats['moderate_absorption_frac'] = float(
        ((flux >= 0.1) & (flux < 0.5)).sum() / total_pixels)
    stats['weak_absorption_frac'] = float((flux >= 0.5).sum() / total_pixels)

    return stats


# Compute 1D flux power spectrum P_F(k).
def compute_power_spectrum(flux, velocity_spacing):
    n_sightlines, n_pixels = flux.shape

    # Normalize flux to get flux contrast
    mean_flux = np.mean(flux)
    delta_F = flux / mean_flux - 1.0

    # Wavenumber array (s/km units)
    k = np.fft.rfftfreq(n_pixels, d=velocity_spacing)

    # Compute power for each sightline
    power_spectra = []
    for i in range(n_sightlines):
        # FFT of flux contrast
        flux_fft = np.fft.rfft(delta_F[i])
        # Power spectrum (dimensionless, normalized by pixel count)
        power = np.abs(flux_fft)**2 / n_pixels
        power_spectra.append(power)

    power_spectra = np.array(power_spectra)

    # Average over sightlines with proper normalization
    P_k_mean = np.mean(power_spectra, axis=0) * velocity_spacing
    P_k_std = np.std(power_spectra, axis=0) * velocity_spacing
    P_k_err = P_k_std / np.sqrt(n_sightlines)

    # Number of independent modes (useful for error estimation)
    n_modes = np.ones_like(k) * n_sightlines

    return {
        'k': k,
        'P_k_mean': P_k_mean,
        'P_k_std': P_k_std,
        'P_k_err': P_k_err,
        'mean_flux': mean_flux,
        'delta_F': delta_F,
        'n_modes': n_modes,
        'n_sightlines': n_sightlines,
        'velocity_spacing': velocity_spacing
    }


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


# Compute column density distribution f(N_HI) from optical depth.
def compute_column_density_distribution(tau, velocity_spacing, threshold=0.5):
    c = 2.998e5  # km/s
    lambda_lya = 1215.67  # Angstroms
    f_osc = 0.4162  # Oscillator strength for Lyman-alpha

    # Convert to frequency space for column density
    # N_HI = (m_e * c * tau) / (pi * e^2 * f * lambda)
    # Simplified: N_HI ~ 1.13e14 * tau * (dv in km/s)  for Lyman-alpha

    column_densities = []

    for i in range(tau.shape[0]):
        tau_line = tau[i, :]

        # Find absorption features (contiguous pixels above threshold)
        absorbing = tau_line > threshold

        # Label connected regions
        in_feature = False
        feature_start = 0

        for j in range(len(tau_line)):
            if absorbing[j] and not in_feature:
                # Start of new feature
                in_feature = True
                feature_start = j
            elif not absorbing[j] and in_feature:
                # End of feature
                in_feature = False
                feature_tau = tau_line[feature_start:j]

                # Estimate column density using pixel optical depth method
                # N_HI = 1.13e14 * integral(tau * dv) cm^-2
                N_HI = 1.13e14 * np.sum(feature_tau) * velocity_spacing

                if N_HI > 1e12:  # Only count above sensitivity threshold
                    column_densities.append(N_HI)

        # Handle case where feature extends to edge
        if in_feature:
            feature_tau = tau_line[feature_start:]
            N_HI = 1.13e14 * np.sum(feature_tau) * velocity_spacing
            if N_HI > 1e12:
                column_densities.append(N_HI)

    column_densities = np.array(column_densities)

    # Create histogram in log space
    if len(column_densities) > 0:
        log_N_min = 12.0  # log10(N_HI)
        log_N_max = 22.0
        n_bins = 50

        bins = np.logspace(log_N_min, log_N_max, n_bins)
        counts, bin_edges = np.histogram(column_densities, bins=bins)

        # Fit power law in range 13 < log(N) < 17 (typical Lyman-alpha forest)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        fit_mask = (np.log10(bin_centers) > 13.0) & (
            np.log10(bin_centers) < 17.0)

        if np.sum(fit_mask) > 5 and np.sum(counts[fit_mask]) > 0:
            # Fit f(N) = A * N^-beta
            log_N_fit = np.log10(bin_centers[fit_mask])
            log_f_fit = np.log10(counts[fit_mask] + 1e-10)  # Avoid log(0)

            # Linear fit in log-log space
            valid = np.isfinite(log_f_fit) & (counts[fit_mask] > 0)
            if np.sum(valid) > 2:
                coeffs = np.polyfit(log_N_fit[valid], log_f_fit[valid], 1)
                beta_fit = -coeffs[0]  # Negative slope
            else:
                beta_fit = np.nan
        else:
            beta_fit = np.nan
    else:
        bins = np.logspace(12, 22, 50)
        counts = np.zeros(len(bins) - 1)
        bin_edges = bins
        beta_fit = np.nan

    return {
        'N_HI': column_densities,
        'counts': counts,
        'bins': bin_edges,
        'bin_centers': (bin_edges[:-1] + bin_edges[1:]) / 2,
        'beta_fit': beta_fit,
        'n_absorbers': len(column_densities)
    }


# Compute effective optical depth tau_eff from flux.
def compute_effective_optical_depth(tau):
    flux = np.exp(-tau)

    # Global tau_eff
    mean_flux = np.mean(flux)
    tau_eff = -np.log(mean_flux)

    # Per-sightline tau_eff
    mean_flux_per_los = np.mean(flux, axis=1)
    tau_eff_per_los = -np.log(mean_flux_per_los)

    return {
        'tau_eff': float(tau_eff),
        'mean_flux': float(mean_flux),
        'tau_eff_per_sightline': tau_eff_per_los,
        'tau_eff_std': float(np.std(tau_eff_per_los)),
        'tau_eff_err': float(np.std(tau_eff_per_los) / np.sqrt(len(tau_eff_per_los)))
    }


# Compute line width (Doppler b-parameter) distribution from absorption features.
def compute_line_width_distribution(tau, velocity_spacing, threshold=0.5):
    from scipy.optimize import curve_fit
    from scipy.signal import find_peaks

    # Physical constants
    m_H = 1.673e-24  # Hydrogen mass in grams
    k_B = 1.381e-16  # Boltzmann constant in CGS

    column_densities = []
    b_parameters = []

    # Approximate Voigt profile for optical depth.
    def voigt_approx(v, tau_0, b, v_center):
        a = 4.7e-4  # Damping parameter for Lyman-alpha
        u = (v - v_center) / b

        # Gaussian core (approximation for thermal broadening)
        tau = tau_0 * np.exp(-u**2)
        return tau

    for i in range(tau.shape[0]):
        tau_line = tau[i, :]

        # Find peaks in optical depth (absorption features)
        peaks, properties = find_peaks(tau_line, height=threshold, distance=5)

        for peak_idx in peaks:
            # Define feature extent (where tau drops to threshold)
            left = peak_idx
            while left > 0 and tau_line[left] > threshold * 0.3:
                left -= 1

            right = peak_idx
            while right < len(tau_line) - 1 and tau_line[right] > threshold * 0.3:
                right += 1

            if right - left < 3:  # Too narrow to fit
                continue

            # Extract feature
            feature_tau = tau_line[left:right+1]
            feature_v = np.arange(len(feature_tau)) * velocity_spacing

            # Initial guess for Voigt fit
            tau_0_guess = tau_line[peak_idx]
            v_center_guess = (peak_idx - left) * velocity_spacing
            b_guess = 20.0  # km/s, typical IGM value

            try:
                # Fit Voigt profile
                popt, _ = curve_fit(
                    voigt_approx,
                    feature_v,
                    feature_tau,
                    p0=[tau_0_guess, b_guess, v_center_guess],
                    bounds=([0, 1.0, 0], [np.inf, 100.0, feature_v[-1]]),
                    maxfev=1000
                )

                tau_0_fit, b_fit, v_center_fit = popt

                # Estimate column density from integrated optical depth
                N_HI = 1.13e14 * np.sum(feature_tau) * velocity_spacing

                # Only keep physically reasonable absorbers
                if N_HI > 1e12 and 2.0 < b_fit < 80.0:
                    column_densities.append(N_HI)
                    b_parameters.append(b_fit)

            except (RuntimeError, ValueError):
                # Fit failed, skip this feature
                continue

    column_densities = np.array(column_densities)
    b_parameters = np.array(b_parameters)

    # Convert b-parameters to temperatures (assuming thermal broadening)
    # b = sqrt(2kT/m) => T = (b^2 * m) / (2k)
    # For HI: T(K) = 1.28e4 * b(km/s)^2
    temperatures = 1.28e4 * b_parameters**2

    return {
        'N_HI': column_densities,
        'b_params': b_parameters,
        'temperatures': temperatures,
        'b_median': float(np.median(b_parameters)) if len(b_parameters) > 0 else np.nan,
        'b_mean': float(np.mean(b_parameters)) if len(b_parameters) > 0 else np.nan,
        'b_std': float(np.std(b_parameters)) if len(b_parameters) > 0 else np.nan,
        'n_absorbers': len(b_parameters)
    }


# Compute temperature-density (T-ρ) relation from IGM gas along sightlines.
def compute_temperature_density_relation(temperature, density, tau, min_tau=0.1):
    # Flatten arrays and filter by optical depth
    temp_flat = temperature.flatten()
    dens_flat = density.flatten()
    tau_flat = tau.flatten()

    # Filter: only include absorbing gas (tau > min_tau) and valid values
    mask = (tau_flat > min_tau) & (temp_flat > 0) & (dens_flat > 0)
    mask &= np.isfinite(temp_flat) & np.isfinite(dens_flat)

    temp_filtered = temp_flat[mask]
    dens_filtered = dens_flat[mask]

    if len(temp_filtered) < 100:
        print(f"  Warning: Only {len(temp_filtered)} valid pixels for T-ρ fit")
        return {
            'temperature': temp_filtered,
            'density': dens_filtered,
            'log_T': np.array([]),
            'log_rho': np.array([]),
            'T0': np.nan,
            'gamma': np.nan,
            'gamma_err': np.nan,
            'n_pixels': len(temp_filtered)
        }

    # Convert density to overdensity (ρ/ρ_mean)
    rho_mean = np.median(dens_filtered)
    overdensity = dens_filtered / rho_mean

    # Take logarithms for power-law fit
    log_T = np.log10(temp_filtered)
    log_rho = np.log10(overdensity)

    # Fit T-ρ relation: log(T) = log(T0) + (gamma-1) * log(ρ/ρ_mean)
    # Robust fit using median binning
    rho_bins = np.linspace(log_rho.min(), log_rho.max(), 30)
    T_median = []
    rho_centers = []

    for i in range(len(rho_bins) - 1):
        mask_bin = (log_rho >= rho_bins[i]) & (log_rho < rho_bins[i+1])
        if np.sum(mask_bin) > 10:
            T_median.append(np.median(log_T[mask_bin]))
            rho_centers.append((rho_bins[i] + rho_bins[i+1]) / 2)

    if len(rho_centers) > 5:
        # Linear fit in log-log space
        coeffs = np.polyfit(rho_centers, T_median, 1)
        gamma_minus_1 = coeffs[0]
        log_T0 = coeffs[1]

        T0 = 10**log_T0
        gamma = gamma_minus_1 + 1.0

        # Estimate uncertainty
        T_pred = np.polyval(coeffs, rho_centers)
        residuals = np.array(T_median) - T_pred
        gamma_err = np.std(
            residuals) / np.std(rho_centers) if len(rho_centers) > 1 else np.nan
    else:
        T0 = np.nan
        gamma = np.nan
        gamma_err = np.nan

    return {
        'temperature': temp_filtered,
        'density': overdensity,
        'log_T': log_T,
        'log_rho': log_rho,
        'T0': float(T0) if np.isfinite(T0) else np.nan,
        'gamma': float(gamma) if np.isfinite(gamma) else np.nan,
        'gamma_err': float(gamma_err) if np.isfinite(gamma_err) else np.nan,
        'n_pixels': len(temp_filtered),
        'rho_mean': float(rho_mean)
    }


# Compute statistics for metal line absorption systems.
def compute_metal_line_statistics(tau, velocity_spacing, ion_name='Metal', threshold=0.05):
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
                feature_tau = tau_line[feature_start:j]

                # Estimate column density (simplified, ion-specific factors needed)
                # For now use generic: N ~ 10^13 * integral(tau * dv)
                N_ion = 1e13 * np.sum(feature_tau) * velocity_spacing
                column_densities.append(N_ion)

        # Handle case where feature extends to edge
        if in_feature:
            feature_tau = tau_line[feature_start:]
            N_ion = 1e13 * np.sum(feature_tau) * velocity_spacing
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


# Format statistics dictionary as a nice text table.
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
