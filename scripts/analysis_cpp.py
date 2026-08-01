"""
CGM Analysis - C++ Implementation Wrapper

This module provides Python bindings to the C++ implementation.
"""

import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import numpy as np
from typing import Optional, Dict, Any

from _analysis_cpp import (
    compute_power_spectrum as _cpp_compute_power_spectrum,
    compute_column_density_distribution as _cpp_compute_colden,
    compute_line_width_distribution as _cpp_compute_line_width,
    compute_flux_statistics as _cpp_compute_flux_stats,
    compute_temperature_density_relation as _cpp_compute_temp_dens,
    compute_tdens_binned,
)


def compute_flux_statistics(tau: np.ndarray) -> Dict[str, float]:
    """Compute basic flux statistics from optical depth."""
    if tau.dtype != np.float32:
        tau = tau.astype(np.float32)
    result = _cpp_compute_flux_stats(tau)
    return {
        'mean_flux': result['mean_flux'],
        'median_flux': result['median_flux'],
        'std_flux': result['std_flux'],
        'min_flux': result['min_flux'],
        'max_flux': result['max_flux'],
        'mean_tau': result['mean_tau'],
        'median_tau': result['median_tau'],
        'effective_tau': result['effective_tau'],
        'deep_absorption_frac': result['deep_absorption_frac'],
        'moderate_absorption_frac': result['moderate_absorption_frac'],
        'weak_absorption_frac': result['weak_absorption_frac'],
    }


def _scipy_power_spectrum(
    flux: np.ndarray,
    velocity_spacing: float,
) -> Dict[str, Any]:
    """Vectorized scipy power spectrum. Always-available, trusted reference path."""
    from scipy import fft
    
    n_sightlines, n_pixels = flux.shape
    
    mean_flux = float(flux.mean())
    if mean_flux <= 0:
        raise ValueError("Mean flux must be positive")
    
    n_k = n_pixels // 2 + 1
    k = 2.0 * np.pi * np.arange(n_k) / (n_pixels * velocity_spacing)
    
    # Vectorized over all sightlines: one batched real FFT, threaded across cores
    # (scipy.fft releases the GIL via workers=-1). Physics is identical to the old
    # per-sightline loop; float64 accumulation matches its precision.
    delta_F = flux / mean_flux - 1.0                    # (n_sightlines, n_pixels)
    fft_result = fft.rfft(delta_F, axis=1, workers=-1)  # (n_sightlines, n_k)
    power = np.abs(fft_result) ** 2 / n_pixels          # (n_sightlines, n_k)
    power_sum = power.sum(axis=0, dtype=np.float64)
    power_sum_sq = (power.astype(np.float64) ** 2).sum(axis=0)
    
    P_k_mean = (power_sum / n_sightlines) * velocity_spacing
    mean_power = power_sum / n_sightlines
    mean_power_sq = power_sum_sq / n_sightlines
    
    variance = np.maximum(mean_power_sq - mean_power ** 2, 0)
    P_k_std = np.sqrt(variance) * velocity_spacing
    P_k_err = P_k_std / np.sqrt(n_sightlines)
    n_modes = np.ones(n_k) * n_sightlines
    
    return {
        'k': k,
        'P_k_mean': P_k_mean,
        'P_k_std': P_k_std,
        'P_k_err': P_k_err,
        'mean_flux': mean_flux,
        'n_modes': n_modes,
        'n_sightlines': n_sightlines,
        'velocity_spacing': velocity_spacing,
    }


def compute_power_spectrum(
    flux: np.ndarray,
    velocity_spacing: float,
    chunk_size: int = 1000,
    use_cpp: bool = False,
) -> Dict[str, Any]:
    """Flux power spectrum P_F(k): scipy by default, optional C++ FFTW via use_cpp.

    TODO: when sightline counts grow large, revisit the C++ FFTW path (fix CMake
    FFTW detection + the OpenMP thread race) and benchmark before enabling use_cpp.
    """
    if use_cpp:
        try:
            f32 = flux if flux.dtype == np.float32 else flux.astype(np.float32)
            result = _cpp_compute_power_spectrum(f32, velocity_spacing, chunk_size)
            return {
                'k': np.array(result['k']),
                'P_k_mean': np.array(result['P_k_mean']),
                'P_k_std': np.array(result['P_k_std']),
                'P_k_err': np.array(result['P_k_err']),
                'mean_flux': result['mean_flux'],
                'n_modes': np.array(result['n_modes']),
                'n_sightlines': result['n_sightlines'],
                'velocity_spacing': result['velocity_spacing'],
            }
        except Exception as e:
            print(f"  [power spectrum] C++ path unavailable ({e}); using scipy")
    return _scipy_power_spectrum(flux, velocity_spacing)


def compute_column_density_distribution(
    tau: np.ndarray,
    velocity_spacing: float,
    threshold: float = 0.5,
    colden: Optional[np.ndarray] = None,
    redshift: Optional[float] = None,
    box_size_ckpc_h: Optional[float] = None,
    hubble: float = 0.6774,
    omega_m: float = 0.3089,
    absorber_mode: int = 0,
    cell_dv: float = 50.0,
    colden_mode: int = 0,
    dx_mode: int = 0,
    norm_mode: int = 0,
    log_N_min: float = 12.0,
    log_N_max: float = 22.0,
    n_bins: int = 50,
    fit_log_N_min: float = 12.0,
    fit_log_N_max: float = 14.4771212547196624,
    min_N_gate: float = 1e12,
) -> Dict[str, Any]:
    """Compute column density distribution function.

    Uses colden (peak) method when available.

    The mode arguments select between absorber definitions and normalisations so
    the CDDF systematics can be swept without recompiling. Production values live
    in config.CDDF_OPTIONS; the defaults here are the unmodified legacy behaviour.

    absorber_mode  0 = contiguous run of pixels with tau > threshold (ours)
                   1 = whole sightline (fake_spectra column_density_function line=True)
                   2 = fixed cell_dv km/s cells (fake_spectra line=False, close=cell_dv)
    colden_mode    0 = max over the feature, 1 = sum (fake_spectra's definition)
    dx_mode        0 = comoving Mpc box length, 1 = absorption distance X(z) with (1+z)^2
    norm_mode      0 = divide by dex, 1 = divide by linear dN (true f(N) = dn/dN dX)
    min_N_gate     absorbers at or below this N_HI are dropped; set 0 to disable
                   (fake_spectra has no such cut)
    """
    if tau.dtype != np.float32:
        tau = tau.astype(np.float32)
    
    # Make contiguous - let C++ handle the layout correctly
    if not tau.flags.c_contiguous:
        tau = np.ascontiguousarray(tau)
    
    # Handle colden - make contiguous for C++
    if colden is not None and colden.size > 0:
        if colden.dtype != np.float32:
            colden = colden.astype(np.float32)
        if not colden.flags.c_contiguous:
            colden = np.ascontiguousarray(colden)
        colden_arg = colden
    else:
        colden_arg = np.array([], dtype=np.float32).reshape(0, 0)
    
    redshift_val = redshift if redshift is not None else float('nan')
    box_size_val = box_size_ckpc_h if box_size_ckpc_h is not None else float('nan')
    
    result = _cpp_compute_colden(
        tau, velocity_spacing, threshold,
        colden_arg, redshift_val, box_size_val, hubble, omega_m,
        absorber_mode, cell_dv, colden_mode, dx_mode, norm_mode,
        log_N_min, log_N_max, n_bins, fit_log_N_min, fit_log_N_max, min_N_gate
    )

    bins_arr = np.array(result['bins'])
    bin_centers_arr = np.array(result['bin_centers'])

    out = {
        'N_HI': np.array(result['N_HI']),
        'counts': np.array(result['counts']),
        'bins': bins_arr,
        'bin_centers': bin_centers_arr,
        'log_bin_edges': np.log10(bins_arr),
        'log10_N_HI': np.log10(bin_centers_arr),
        'delta_log_N': np.diff(np.log10(bins_arr)),
        'f_N': np.array(result['f_N']),
        'f_N_HI': np.array(result['f_N']),
        'beta_fit': result['beta_fit'],
        'n_absorbers': result['n_absorbers'],
        'n_sightlines': result['n_sightlines'],
        'dX': result['dX'],
        'redshift': result['redshift'],
    }

    # Per-absorber diagnostics and the config echo. Guarded so an older,
    # not-yet-rebuilt extension module still works with the current Python.
    for key in ('N_HI_alt', 'peak_tau', 'feature_pixels'):
        if key in result:
            out[key] = np.array(result[key])
    for key in ('dX_comoving_mpc', 'X_absorption', 'n_features_total', 'used_colden',
                'absorber_mode', 'cell_dv', 'colden_mode', 'dx_mode', 'norm_mode',
                'log_N_min', 'log_N_max', 'n_bins', 'fit_log_N_min', 'fit_log_N_max',
                'min_N_gate',
                # Fit uncertainties; absent until the extension is rebuilt.
                'beta_fit_err', 'beta_fit_weighted', 'beta_fit_weighted_err',
                'beta_fit_n_bins'):
        if key in result:
            out[key] = result[key]
    return out


def compute_line_width_distribution(
    tau: np.ndarray,
    velocity_spacing: float,
    threshold: float = 0.5,
    colden: Optional[np.ndarray] = None,
    elem: str = "H",
    ion: int = 1,
    line: int = 1215
) -> Dict[str, Any]:
    """Compute line width (b-parameter) distribution using C++ implementation."""
    if tau.dtype != np.float32:
        tau = tau.astype(np.float32)
    
    colden_arg = np.array([], dtype=np.float32).reshape(0, 0)
    
    result = _cpp_compute_line_width(tau, velocity_spacing, threshold, colden_arg)
    
    return {
        'N_HI': np.array(result['N_HI']),
        'b_params': np.array(result['b_params']),
        'temperatures': np.array(result['temperatures']),
        'b_median': result['b_median'],
        'b_mean': result['b_mean'],
        'b_std': result['b_std'],
        'n_absorbers': result['n_absorbers'],
    }


def compute_temperature_density_relation(
    temperature: np.ndarray,
    density: np.ndarray,
    tau: np.ndarray,
    min_tau: float = 0.1
) -> Dict[str, Any]:
    """Compute temperature-density relation."""
    if temperature.dtype != np.float32:
        temperature = temperature.astype(np.float32)
    if density.dtype != np.float32:
        density = density.astype(np.float32)
    if tau.dtype != np.float32:
        tau = tau.astype(np.float32)
    
    result = _cpp_compute_temp_dens(temperature, density, tau, min_tau)
    
    return {
        'temperature': np.array(result['temperature']),
        'density': np.array(result['density']),
        'log_T': np.array(result['log_T']),
        'log_rho': np.array(result['log_rho']),
        'T0': result['T0'],
        'gamma': result['gamma'],
        'gamma_err': result['gamma_err'],
        'rho_mean': result.get('rho_mean', np.nan),
        'n_pixels': result['n_pixels'],
    }
