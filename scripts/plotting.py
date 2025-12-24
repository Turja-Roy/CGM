import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import cm
from matplotlib.colors import LogNorm


# ===================#
# PLOTTING UTILITIES #
# ===================#

def setup_plot_style():
    """Setup consistent matplotlib plotting style."""
    plt.style.use('default')
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9


def save_plot(fig, filepath, dpi=150):
    """Save plot to file, creating directories if needed."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"  Saved: {filepath}")


def create_sample_spectra_plot(velocity, flux, redshift, n_samples=5, output_path=None):
    """
    Create plot showing sample spectra sightlines.
    
    Args:
        velocity: Velocity array (km/s)
        flux: Flux array (n_sightlines, n_pixels)
        redshift: Redshift of spectra
        n_samples: Number of sample sightlines to plot
        output_path: Output file path (optional)
        
    Returns:
        fig, axes: Matplotlib figure and axes objects
    """
    n_samples = min(n_samples, flux.shape[0])
    indices = np.random.choice(flux.shape[0], n_samples, replace=False)
    
    fig, axes = plt.subplots(n_samples, 1, figsize=(14, 2*n_samples))
    if n_samples == 1:
        axes = [axes]
    
    for i, (ax, idx) in enumerate(zip(axes, indices)):
        ax.plot(velocity, flux[idx], 'k-', lw=0.8, alpha=0.8)
        ax.set_ylabel('Flux', fontsize=10)
        ax.set_ylim(-0.05, 1.15)
        ax.axhline(1.0, color='r', ls='--', alpha=0.3, lw=1)
        ax.axhline(0.0, color='gray', ls='--', alpha=0.3, lw=1)
        ax.grid(alpha=0.2)
        ax.set_title(f'Sightline {idx} (z={redshift:.2f})', fontsize=11)
        
        if i == n_samples - 1:
            ax.set_xlabel('Velocity [km/s]', fontsize=11)
    
    plt.suptitle(f'CAMEL Lyman-α Spectra (z={redshift:.2f})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        save_plot(fig, output_path)
    
    return fig, axes


# Create comparison plot for multiple spectral lines.
def plot_multi_line_comparison(line_stats_list, redshift, output_path, title=None):
    """
    Create multi-panel comparison plot for different spectral lines.
    
    Args:
        line_stats_list: List of metal line statistics dicts
        redshift: Redshift of spectra
        output_path: Output file path
        title: Plot title (optional)
    """
    if len(line_stats_list) == 0:
        print("  Warning: No line statistics provided")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ion_names = [stats['ion_name'] for stats in line_stats_list]
    n_ions = len(ion_names)
    cmap = cm.get_cmap('tab10')
    colors = [cmap(i) for i in np.linspace(0, 1, n_ions)]
    
    # Panel 1: Number of absorbers (dN/dz)
    ax = axes[0, 0]
    dN_dz_values = [stats['dN_dz'] for stats in line_stats_list]
    bars = ax.bar(range(n_ions), dN_dz_values, color=colors, edgecolor='black', alpha=0.7)
    ax.set_xticks(range(n_ions))
    ax.set_xticklabels(ion_names, rotation=45, ha='right')
    ax.set_ylabel('dN/dz (absorbers per unit redshift)', fontsize=12)
    ax.set_title('Absorber Line Density', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 2: Covering fraction
    ax = axes[0, 1]
    covering_fractions = [stats['covering_fraction'] * 100 for stats in line_stats_list]
    bars = ax.bar(range(n_ions), covering_fractions, color=colors, edgecolor='black', alpha=0.7)
    ax.set_xticks(range(n_ions))
    ax.set_xticklabels(ion_names, rotation=45, ha='right')
    ax.set_ylabel('Covering Fraction (%)', fontsize=12)
    ax.set_title('Sky Coverage', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: Mean optical depth (in absorbing regions)
    ax = axes[1, 0]
    mean_taus = [stats['mean_tau'] for stats in line_stats_list]
    # Use log scale if range is large
    if max(mean_taus) / min([t for t in mean_taus if t > 0] + [1]) > 100:
        ax.set_yscale('log')
    bars = ax.bar(range(n_ions), mean_taus, color=colors, edgecolor='black', alpha=0.7)
    ax.set_xticks(range(n_ions))
    ax.set_xticklabels(ion_names, rotation=45, ha='right')
    ax.set_ylabel('Mean τ (absorbing regions)', fontsize=12)
    ax.set_title('Absorption Strength', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: Column density distributions (overlaid histograms)
    ax = axes[1, 1]
    for i, stats in enumerate(line_stats_list):
        if len(stats['column_densities']) > 0:
            log_N = np.log10(stats['column_densities'])
            ax.hist(log_N, bins=20, alpha=0.5, label=stats['ion_name'],
                   color=colors[i], edgecolor='black', linewidth=0.5)
    ax.set_xlabel('log₁₀(N / cm⁻²)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Column Density Distributions', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Overall title
    if title is None:
        title = f'Multi-Line Absorption Comparison - Redshift z = {redshift:.2f}'
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    save_plot(fig, output_path)
    plt.close()


# Create flux power spectrum plot.
def plot_flux_power_spectrum(power_dict, redshift, output_path, title=None):
    """
    Create flux power spectrum plot.
    
    Args:
        power_dict: Power spectrum data dict from compute_power_spectrum()
        redshift: Redshift of spectra
        output_path: Output file path
        title: Plot title (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    k = power_dict['k']
    P_k = power_dict['P_k_mean']
    P_k_err = power_dict['P_k_err']
    
    # Only plot positive k (skip DC component)
    mask = k > 0
    k = k[mask]
    P_k = P_k[mask]
    P_k_err = P_k_err[mask]
    
    # Plot with error bars
    ax.loglog(k, P_k, 'o-', color='steelblue', linewidth=2, 
              markersize=4, label=f'z = {redshift:.2f}')
    ax.fill_between(k, P_k - P_k_err, P_k + P_k_err, 
                     alpha=0.3, color='steelblue')
    
    # Formatting
    ax.set_xlabel(r'Wavenumber $k$ [s/km]', fontsize=14)
    ax.set_ylabel(r'Power Spectrum $P_F(k)$ [km/s]', fontsize=14)
    ax.set_xlim(k[1], k[-1])
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=12)
    
    if title is None:
        title = f'Lyman-α Flux Power Spectrum (z={redshift:.2f})'
    ax.set_title(title, fontsize=15, fontweight='bold')
    
    # Add info box
    info_text = f"N_sightlines = {power_dict['n_sightlines']}\n"
    info_text += f"Mean flux = {power_dict['mean_flux']:.3f}"
    ax.text(0.05, 0.05, info_text, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10, verticalalignment='bottom')
    
    plt.tight_layout()
    save_plot(fig, output_path)
    plt.close()


# Create column density distribution plot.
def plot_column_density_distribution(cddf_dict, redshift, output_path, title=None):
    """
    Create column density distribution function plot.
    
    Args:
        cddf_dict: CDDF data dict from compute_column_density_distribution()
        redshift: Redshift of spectra
        output_path: Output file path
        title: Plot title (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    bins = cddf_dict['bins']
    counts = cddf_dict['counts']
    bin_centers = cddf_dict['bin_centers']
    beta = cddf_dict['beta_fit']
    
    # Normalize to get f(N_HI)
    bin_widths = np.diff(bins)
    f_N = counts / bin_widths
    
    # Plot
    mask = f_N > 0
    ax.loglog(bin_centers[mask], f_N[mask], 'o-', color='coral',
              linewidth=2, markersize=5, label='Measured')
    
    # Add power law fit if available
    if not np.isnan(beta):
        fit_range = (np.log10(bin_centers) > 13) & (np.log10(bin_centers) < 17)
        N_fit = bin_centers[fit_range]
        # Normalize to data at N ~ 1e14
        norm_idx = np.argmin(np.abs(bin_centers - 1e14))
        A_norm = f_N[norm_idx] / (bin_centers[norm_idx]**(-beta))
        f_fit = A_norm * N_fit**(-beta)
        
        ax.loglog(N_fit, f_fit, '--', color='red', linewidth=2,
                  label=f'Power law: β = {beta:.2f}')
    
    # Formatting
    ax.set_xlabel(r'Column Density $N_{\rm HI}$ [cm$^{-2}$]', fontsize=14)
    ax.set_ylabel(r'$f(N_{\rm HI})$ [cm$^2$]', fontsize=14)
    ax.set_xlim(1e12, 1e22)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=12)
    
    if title is None:
        title = f'Column Density Distribution (z={redshift:.2f})'
    ax.set_title(title, fontsize=15, fontweight='bold')
    
    # Add info box
    info_text = f"N_absorbers = {cddf_dict['n_absorbers']}\n"
    if not np.isnan(beta):
        info_text += f"β = {beta:.2f}"
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    save_plot(fig, output_path)
    plt.close()


# Create line width (b-parameter) distribution plot.
def plot_line_width_distribution(lwd_dict, redshift, output_path, title=None):
    """
    Create line width distribution plot with b-N_HI correlation.
    
    Args:
        lwd_dict: Line width data dict from compute_line_width_distribution()
        redshift: Redshift of spectra
        output_path: Output file path
        title: Plot title (optional)
    """
    if lwd_dict['n_absorbers'] == 0:
        print(f"  Warning: No absorbers found for line width analysis")
        return
    
    fig = plt.figure(figsize=(14, 6))
    
    # Create two subplots: b histogram and b(N_HI) correlation
    gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    
    b_params = lwd_dict['b_params']
    N_HI = lwd_dict['N_HI']
    
    # ===== Left panel: b-parameter histogram =====
    ax1.hist(b_params, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(lwd_dict['b_median'], color='red', linestyle='--', linewidth=2,
                label=f"Median = {lwd_dict['b_median']:.1f} km/s")
    ax1.axvline(lwd_dict['b_mean'], color='orange', linestyle='--', linewidth=2,
                label=f"Mean = {lwd_dict['b_mean']:.1f} km/s")
    
    ax1.set_xlabel('Doppler b-parameter (km/s)', fontsize=13)
    ax1.set_ylabel('Count', fontsize=13)
    ax1.set_title(f'Line Width Distribution (z = {redshift:.2f})', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Add temperature scale on top
    ax1_top = ax1.twiny()
    b_ticks = np.array([5, 10, 20, 30, 40])
    T_ticks = 1.28e4 * b_ticks**2
    ax1_top.set_xlim(ax1.get_xlim())
    ax1_top.set_xticks(b_ticks)
    ax1_top.set_xticklabels([f'{T/1e3:.0f}' for T in T_ticks])
    ax1_top.set_xlabel('Temperature (10³ K)', fontsize=12)
    
    # ===== Right panel: b(N_HI) correlation =====
    # Bin by column density for cleaner visualization
    log_N = np.log10(N_HI)
    N_bins = np.linspace(12, 17, 20)
    b_median_binned = []
    b_16th = []
    b_84th = []
    N_centers = []
    
    for i in range(len(N_bins) - 1):
        mask = (log_N >= N_bins[i]) & (log_N < N_bins[i+1])
        if np.sum(mask) > 5:
            b_in_bin = b_params[mask]
            b_median_binned.append(np.median(b_in_bin))
            b_16th.append(np.percentile(b_in_bin, 16))
            b_84th.append(np.percentile(b_in_bin, 84))
            N_centers.append((N_bins[i] + N_bins[i+1]) / 2)
    
    # Plot individual points (transparency for density)
    ax2.scatter(log_N, b_params, alpha=0.1, s=10, color='gray', label='Individual')
    
    # Plot binned median with scatter
    if len(N_centers) > 0:
        N_centers = np.array(N_centers)
        b_median_binned = np.array(b_median_binned)
        b_16th = np.array(b_16th)
        b_84th = np.array(b_84th)
        
        ax2.plot(N_centers, b_median_binned, 'o-', color='red', linewidth=2,
                 markersize=8, label='Median (binned)')
        ax2.fill_between(N_centers, b_16th, b_84th, alpha=0.3, color='red',
                         label='16-84th percentile')
    
    ax2.set_xlabel('log₁₀(N_HI / cm⁻²)', fontsize=13)
    ax2.set_ylabel('Doppler b-parameter (km/s)', fontsize=13)
    ax2.set_title(f'b-N_HI Correlation (z = {redshift:.2f})', fontsize=14, fontweight='bold')
    ax2.set_xlim(12, 17)
    ax2.set_ylim(0, 60)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Overall title
    if title is None:
        title = f'Line Width Analysis - Redshift z = {redshift:.2f}'
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # Add info box
    info_text = f"N_absorbers = {lwd_dict['n_absorbers']}\n"
    info_text += f"⟨b⟩ = {lwd_dict['b_mean']:.1f} ± {lwd_dict['b_std']:.1f} km/s\n"
    if lwd_dict['n_absorbers'] > 0:
        T_mean = 1.28e4 * lwd_dict['b_mean']**2
        info_text += f"⟨T⟩ = {T_mean/1e3:.0f} × 10³ K"
    
    ax1.text(0.95, 0.95, info_text, transform=ax1.transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=10, verticalalignment='top', horizontalalignment='right')
    
    plt.tight_layout()
    save_plot(fig, output_path)
    plt.close()


# Create temperature-density relation plot.
def plot_temperature_density_relation(tdens_dict, redshift, output_path, title=None):
    """
    Create temperature-density relation plot.
    
    Args:
        tdens_dict: T-rho data dict from compute_temperature_density_relation()
        redshift: Redshift of spectra
        output_path: Output file path
        title: Plot title (optional)
    """
    if tdens_dict['n_pixels'] < 100:
        print(f"  Warning: Insufficient data for T-ρ plot ({tdens_dict['n_pixels']} pixels)")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    log_T = tdens_dict['log_T']
    log_rho = tdens_dict['log_rho']
    T0 = tdens_dict['T0']
    gamma = tdens_dict['gamma']
    
    # Create 2D histogram for density visualization
    h, xedges, yedges = np.histogram2d(log_rho, log_T, bins=50)
    h = h.T  # Transpose for correct orientation
    
    # Plot 2D histogram
    extent = (xedges[0], xedges[-1], yedges[0], yedges[-1])
    im = ax.imshow(h, origin='lower', extent=extent, aspect='auto',
                   cmap='YlOrRd', norm=LogNorm(vmin=1, vmax=h.max()),
                   interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Number of pixels')
    
    # Plot best-fit power law
    if np.isfinite(T0) and np.isfinite(gamma):
        rho_range = np.linspace(log_rho.min(), log_rho.max(), 100)
        T_fit = np.log10(T0) + (gamma - 1) * rho_range
        ax.plot(rho_range, T_fit, 'b--', linewidth=3, label=f'T = T₀(ρ/ρ̄)^(γ-1)')
        
        # Add fit parameters as text
        fit_text = f'T₀ = {T0:.0f} K\nγ = {gamma:.3f}'
        ax.text(0.05, 0.95, fit_text, transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=12, verticalalignment='top', fontweight='bold')
    
    ax.set_xlabel('log₁₀(ρ/ρ̄)', fontsize=14)
    ax.set_ylabel('log₁₀(T / K)', fontsize=14)
    
    if title is None:
        title = f'Temperature-Density Relation - Redshift z = {redshift:.2f}'
    ax.set_title(title, fontsize=15, fontweight='bold')
    
    if np.isfinite(gamma):
        ax.legend(fontsize=11, loc='lower right')
    
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add info box
    info_text = f"N_pixels = {tdens_dict['n_pixels']:,}\n"
    info_text += f"z = {redshift:.3f}"
    ax.text(0.95, 0.05, info_text, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10, verticalalignment='bottom', horizontalalignment='right')
    
    plt.tight_layout()
    save_plot(fig, output_path)
    plt.close()
