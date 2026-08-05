import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LogNorm


######################
# PLOTTING UTILITIES #
######################

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


def plot_multi_line_comparison(line_stats_list, redshift, output_path, title=None):
    if len(line_stats_list) == 0:
        print("  Warning: No line statistics provided")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ion_names = [stats['ion_name'] for stats in line_stats_list]
    n_ions = len(ion_names)
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in np.linspace(0, 1, n_ions)]

    # Panel 1: Number of absorbers (dN/dz)
    ax = axes[0, 0]
    dN_dz_values = [stats['dN_dz'] for stats in line_stats_list]
    bars = ax.bar(range(n_ions), dN_dz_values, color=colors,
                  edgecolor='black', alpha=0.7)
    ax.set_xticks(range(n_ions))
    ax.set_xticklabels(ion_names, rotation=45, ha='right')
    ax.set_ylabel('dN/dz (absorbers per unit redshift)', fontsize=12)
    ax.set_title('Absorber Line Density', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 2: Covering fraction
    ax = axes[0, 1]
    covering_fractions = [stats['covering_fraction']
                          * 100 for stats in line_stats_list]
    bars = ax.bar(range(n_ions), covering_fractions,
                  color=colors, edgecolor='black', alpha=0.7)
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
    bars = ax.bar(range(n_ions), mean_taus, color=colors,
                  edgecolor='black', alpha=0.7)
    ax.set_xticks(range(n_ions))
    ax.set_xticklabels(ion_names, rotation=45, ha='right')
    ax.set_ylabel('Mean tau (absorbing regions)', fontsize=12)
    ax.set_title('Absorption Strength', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 4: Column density distributions (overlaid histograms)
    ax = axes[1, 1]
    for i, stats in enumerate(line_stats_list):
        if len(stats['column_densities']) > 0:
            log_N = np.log10(stats['column_densities'])
            ax.hist(log_N, bins=20, alpha=0.5, label=stats['ion_name'],
                    color=colors[i], edgecolor='black', linewidth=0.5)
    ax.set_xlabel('log_10(N / cm^-2)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Column Density Distributions',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Overall title
    if title is None:
        title = f'Multi-Line Absorption Comparison - Redshift z = {
            redshift:.2f}'
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    save_plot(fig, output_path)
    plt.close()


def plot_flux_power_spectrum(power_dict, redshift, output_path, title=None):
    fig, ax = plt.subplots(figsize=(10, 7))

    k = power_dict['k']
    P_k = power_dict['P_k_mean']
    P_k_err = power_dict['P_k_err']

    # Only plot positive k (skip DC component)
    mask = k > 0
    k = k[mask]
    P_k = P_k[mask]
    P_k_err = P_k_err[mask]

    # Compute k*P(k)/pi following Khaire et al. (2019) convention
    kPk_pi = k * P_k / np.pi
    kPk_pi_err = k * P_k_err / np.pi

    # Plot with error bars
    ax.loglog(k, kPk_pi, 'o-', color='steelblue', linewidth=2,
              markersize=4, label=f'z = {redshift:.2f}')
    ax.fill_between(k, kPk_pi - kPk_pi_err, kPk_pi + kPk_pi_err,
                    alpha=0.3, color='steelblue')

    # Formatting
    ax.set_xlabel(r'Wavenumber $k$ [s/km]', fontsize=14)
    ax.set_ylabel(r'$k \cdot P_F(k) / \pi$ [dimensionless]', fontsize=14)
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


def plot_column_density_distribution(cddf_dict, redshift, output_path, title=None):
    """
    Plot the column density distribution function (CDDF).
    
    Under config.CDDF_OPTIONS this is f(N) = dn/dN dX in cm^2, normalised by the
    number of sightlines and the dimensionless absorption distance X(z). Axis
    labels and limits follow the norm_mode / dx_mode echoed in cddf_dict, so
    files carrying the per-dex Mpc^-1 quantity still render correctly.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    bins = cddf_dict['bins']
    counts = cddf_dict['counts']
    bin_centers = cddf_dict['bin_centers']
    beta = cddf_dict['beta_fit']

    # Use the properly normalized f(N) from the cddf_dict
    # f(N) is now in units of [Mpc^-1]
    f_N = cddf_dict['f_N']

    # Poisson errors per bin: the high-N end is often two or three absorbers.
    mask = f_N > 0
    counts_arr = np.asarray(counts, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        f_N_err = np.where(counts_arr > 0,
                           f_N / np.sqrt(np.maximum(counts_arr, 1.0)), np.nan)
    ax.errorbar(bin_centers[mask], f_N[mask], yerr=f_N_err[mask],
                fmt='o-', color='coral', linewidth=2, markersize=5,
                capsize=3, ecolor='coral', label='Measured (Poisson errors)')
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Draw the fit only over the range it was fitted on, so the line cannot
    # imply it constrains decades it never saw.
    if not np.isnan(beta):
        log_centers = np.log10(bin_centers)
        # `or` rather than a .get default: these keys are present but None for
        # CSVs that carry no config echo.
        lo = cddf_dict.get('fit_log_N_min') or 12.0
        hi = cddf_dict.get('fit_log_N_max') or 16.0
        fit_range = (log_centers >= lo) & (log_centers <= hi)
        N_fit = bin_centers[fit_range]
        # Normalise to the data at the centre of the fit range
        norm_idx = np.argmin(np.abs(log_centers - 0.5 * (lo + hi)))
        if f_N[norm_idx] > 0:
            A_norm = f_N[norm_idx] / (bin_centers[norm_idx]**(-beta))
            f_fit = A_norm * N_fit**(-beta)

            ax.loglog(N_fit, f_fit, '--', color='red', linewidth=2,
                      label=f'Power law: β = {beta:.2f}')

    # Formatting
    ax.set_xlabel(r'Column Density $N_{\rm HI}$ [cm$^{-2}$]', fontsize=14)

    # Units follow norm_mode: 1 gives f(N) = dn/dN dX in cm^2, which runs ~1e-12
    # at log N ~ 13 down to ~1e-23 in the DLA tail and so needs its own limits;
    # otherwise the per-dex Mpc^-1 quantity, which sits near unity.
    per_dN = cddf_dict.get('norm_mode') == 1
    ax.set_ylabel(r'$f(N_{\rm HI})$ [cm$^{2}$]' if per_dN
                  else r'$f(N_{\rm HI})$ [Mpc$^{-1}$]', fontsize=14)

    if np.any(f_N > 0):
        if per_dN:
            xlo, xhi = 10.0 ** (cddf_dict.get('log_N_min') or 13.0), 1e19
        else:
            xlo, xhi = 1e12, 1e16
        ax.set_xlim(xlo, xhi)
        in_plot = f_N[(bin_centers >= xlo) & (bin_centers <= xhi)]
        in_plot = in_plot[in_plot > 0]
        if in_plot.size:
            ax.set_ylim(in_plot.min() / 10, in_plot.max() * 10)
        else:
            ax.set_ylim(1e-3, 1e3)
    else:
        ax.set_xlim(1e12, 1e16)
        ax.set_ylim(1e-3, 1e3)

    # if np.any(f_N > 0):
    #     positive = f_N[f_N > 0]
    #     if per_dN:
    #         ax.set_xlim(10.0 ** (cddf_dict.get('log_N_min') or 13.0), 1e19)
    #         ax.set_ylim(positive.min() / 10, positive.max() * 10)
    #     else:
    #         ax.set_xlim(1e12, 1e16)
    #         ax.set_ylim(max(1e-5, positive.min() / 10),
    #                     min(1e4, positive.max() * 10))
    # else:
    #     ax.set_xlim(1e12, 1e16)
    #     ax.set_ylim(1e-3, 1e3)

    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=12)

    if title is None:
        title = f'Column Density Distribution (z={redshift:.2f})'
    ax.set_title(title, fontsize=15, fontweight='bold')

    # Add info box with metadata
    info_text = f"N_absorbers = {cddf_dict['n_absorbers']}\n"
    info_text += f"N_sightlines = {cddf_dict.get('n_sightlines', 'N/A')}\n"
    if 'dX' in cddf_dict and cddf_dict['dX'] > 0:
        info_text += (f"X = {cddf_dict['dX']:.4f}\n" if cddf_dict.get('dx_mode') == 1
                      else f"dX = {cddf_dict['dX']:.1f} Mpc\n")
    if not np.isnan(beta):
        beta_err = cddf_dict.get('beta_fit_err', float('nan'))
        if beta_err is not None and np.isfinite(beta_err):
            info_text += f"β = {beta:.2f} ± {beta_err:.2f}"
            bw = cddf_dict.get('beta_fit_weighted', float('nan'))
            if bw is not None and np.isfinite(bw):
                info_text += f"\nβ (Poisson-weighted) = {bw:.2f}"
        else:
            info_text += f"β = {beta:.2f}"
    elif cddf_dict.get('saturated'):
        info_text += "β: suppressed (saturated)"

    ax.grid(True, alpha=0.3, which='both')
    leg = ax.legend(fontsize=12, loc='upper right')

    plt.tight_layout()
    fig.canvas.draw()

    renderer = fig.canvas.get_renderer()
    leg_bb = leg.get_window_extent(renderer)
    ax_bb = ax.get_window_extent(renderer)
    ax.text((leg_bb.x0 - ax_bb.x0) / ax_bb.width + 0.01,
            (leg_bb.y0 - ax_bb.y0) / ax_bb.height - 0.02,
            info_text, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)

    # ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
    #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
    #         fontsize=10, verticalalignment='top')
    #
    # plt.tight_layout()
    save_plot(fig, output_path)
    plt.close()


def plot_line_width_distribution(lwd_dict, redshift, output_path, title=None):
    if lwd_dict['n_absorbers'] == 0:
        print("Warning: No absorbers found for line width analysis")
        return

    fig = plt.figure(figsize=(14, 6))

    # Create two subplots: b histogram and b(N_HI) correlation
    gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    b_params = lwd_dict['b_params']
    N_HI = lwd_dict['N_HI']

    # ===== Left panel: b-parameter histogram =====
    ax1.hist(b_params, bins=30, color='steelblue',
             alpha=0.7, edgecolor='black')
    ax1.axvline(lwd_dict['b_median'], color='red', linestyle='--', linewidth=2,
                label=f"Median = {lwd_dict['b_median']:.1f} km/s")
    ax1.axvline(lwd_dict['b_mean'], color='orange', linestyle='--', linewidth=2,
                label=f"Mean = {lwd_dict['b_mean']:.1f} km/s")

    ax1.set_xlabel('Doppler b-parameter (km/s)', fontsize=13)
    ax1.set_ylabel('Count', fontsize=13)
    ax1.set_title(f'Line Width Distribution (z = {
                  redshift:.2f})', fontsize=14, fontweight='bold')
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
    ax2.scatter(log_N, b_params, alpha=0.1, s=10,
                color='gray', label='Individual')

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

    ax2.set_xlabel('log_10(N_HI / cm^-2)', fontsize=13)
    ax2.set_ylabel('Doppler b-parameter (km/s)', fontsize=13)
    ax2.set_title(
        f'b-N_HI Correlation (z = {redshift:.2f})', fontsize=14, fontweight='bold')
    ax2.set_xlim(12, 17)
    ax2.set_ylim(0, 60)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

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


def plot_temperature_density_relation(tdens_dict, redshift, output_path, title=None):
    if tdens_dict['n_pixels'] < 100:
        print(
            f"Warning: Insufficient data for T-ρ plot ({tdens_dict['n_pixels']} pixels)")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    log_T = tdens_dict.get('log_T')
    log_rho = tdens_dict.get('log_rho')
    T0 = tdens_dict.get('T0')
    gamma = tdens_dict.get('gamma')

    if log_T is not None and log_rho is not None and len(log_T) > 0:
        # 2D histogram (phase diagram) of the per-pixel T-rho distribution
        h, xedges, yedges = np.histogram2d(log_rho, log_T, bins=50)
        h = h.T  # transpose for correct orientation
        extent = (xedges[0], xedges[-1], yedges[0], yedges[-1])
        im = ax.imshow(h, origin='lower', extent=extent, aspect='auto',
                       cmap='YlOrRd', norm=LogNorm(vmin=1, vmax=h.max()),
                       interpolation='nearest')
        cbar = plt.colorbar(im, ax=ax, label='Number of pixels')

    if np.isfinite(T0) and np.isfinite(gamma):
        if log_rho is not None and len(log_rho) > 0:
            rho_range = np.linspace(log_rho.min(), log_rho.max(), 100)
        else:
            rho_range = np.linspace(-2, 2, 100)
        T_fit = np.log10(T0) + (gamma - 1) * rho_range
        ax.plot(rho_range, T_fit, 'b--', linewidth=3,
                label=f'T = T_0(rho/rho_bar)^(gamma-1)')

        fit_text = f'T_0 = {T0:.0f} K\ngamma = {gamma:.3f}'
        ax.text(0.05, 0.95, fit_text, transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=12, verticalalignment='top', fontweight='bold')

    ax.set_xlabel('log_10(rho/rho_bar)', fontsize=14)
    ax.set_ylabel('log_10(T / K)', fontsize=14)

    if title is None:
        title = f'Temperature-Density Relation - Redshift z = {redshift:.2f}'
    ax.set_title(title, fontsize=15, fontweight='bold')

    if np.isfinite(gamma):
        ax.legend(fontsize=11, loc='lower right')

    ax.grid(True, alpha=0.3, linestyle='--')

    info_text = f"N_pixels = {tdens_dict['n_pixels']:,}\n"
    info_text += f"z = {redshift:.3f}"
    ax.text(0.95, 0.05, info_text, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10, verticalalignment='bottom', horizontalalignment='right')

    plt.tight_layout()
    save_plot(fig, output_path)
    plt.close()


def plot_flux_statistics(pdf_dict, stats, output_path, flux=None, tau=None,
                         mean_flux_std=None, mean_flux_err=None, title=None):
    """The 2x2 flux-statistics figure.

    Panels 1 and 2 (flux PDF, log tau PDF) come from pdf_dict, which is exactly
    what compute_flux_tau_pdf returns and what flux_pdf.csv / tau_pdf.csv are
    written from -- so they replot from the CSVs alone.

    Panels 3 and 4 are per-sightline and need the raw arrays. Pass flux and tau
    to draw them; omit both (the CSV-only path, once the spectra HDF5 is gone)
    and they render as labelled placeholders rather than being silently dropped,
    keeping the layout comparable across the two paths.

    mean_flux_std / mean_flux_err default to the matching keys in stats, which
    analyze only stashes there at export time -- hence the explicit arguments.
    """
    import scripts.config as config

    if mean_flux_std is None:
        mean_flux_std = stats.get('mean_flux_std', np.nan)
    if mean_flux_err is None:
        mean_flux_err = stats.get('mean_flux_err', np.nan)

    fig, axes = plt.subplots(2, 2, figsize=config.FIGSIZE_QUAD)

    # Exact PDFs, not the 1e5-of-1e8-pixel subsample this replaced.
    ax = axes[0, 0]
    if pdf_dict is not None and 'flux_bin_centers' in pdf_dict:
        ax.errorbar(pdf_dict['flux_bin_centers'], pdf_dict['flux_density'],
                    yerr=pdf_dict['flux_density_err'], fmt='o-', ms=3, lw=1.2,
                    color='steelblue', ecolor='steelblue', capsize=2,
                    label='PDF (Poisson errors)')
        ax.axvline(stats['mean_flux'], color='red', linestyle='--',
                   label=f"Mean = {stats['mean_flux']:.3f}")
        ax.axvline(stats['median_flux'], color='orange', linestyle='--',
                   label=f"Median = {stats['median_flux']:.3f}")
        ax.set_yscale('log')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No flux PDF available', ha='center', va='center',
                transform=ax.transAxes, fontsize=11, color='gray')
    ax.set_xlabel('Flux $F = e^{-\\tau}$')
    ax.set_ylabel('Probability Density')
    ax.set_title('Flux Distribution')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if pdf_dict is not None and 'log_tau_bin_centers' in pdf_dict:
        ax.errorbar(pdf_dict['log_tau_bin_centers'], pdf_dict['log_tau_density'],
                    yerr=pdf_dict['log_tau_density_err'], fmt='o-', ms=3, lw=1.2,
                    color='coral', ecolor='coral', capsize=2)
        ax.axvline(np.log10(max(stats['median_tau'], 1e-30)), color='red',
                   linestyle='--', label=f"Median = {stats['median_tau']:.3g}")
        ax.set_yscale('log')
        ax.legend(fontsize=8)
        ax.set_title(r'Optical Depth Distribution'
                     f"\n({pdf_dict['frac_tau_overflow']*100:.2f}% above grid, "
                     f"{pdf_dict['frac_tau_zero']*100:.2f}% at $\\tau=0$)")
    else:
        ax.text(0.5, 0.5, 'No optical-depth PDF available', ha='center',
                va='center', transform=ax.transAxes, fontsize=11, color='gray')
        ax.set_title('Optical Depth Distribution')
    ax.set_xlabel(r'$\log_{10} \tau$')
    ax.set_ylabel('Probability Density')
    ax.grid(True, alpha=0.3)

    # Panel 3: Mean flux per sightline
    ax = axes[1, 0]
    if flux is not None:
        mean_flux_per_los = flux.mean(axis=1)
        ax.plot(mean_flux_per_los, marker='o',
                linestyle='-', markersize=3, alpha=0.6)
        ax.axhline(stats['mean_flux'], color='red',
                   linestyle='--', label='Overall mean')
        # Band: per-sightline spread. The ensemble mean is known ~100x better.
        ax.axhspan(stats['mean_flux'] - mean_flux_std,
                   stats['mean_flux'] + mean_flux_std,
                   color='red', alpha=0.12,
                   label=r'$\pm\sigma$ (per sightline)')
        ax.set_xlabel('Sightline Index')
        ax.set_ylabel('Mean Flux')
        ax.set_title(f"Mean Flux per Sightline "
                     f"($\\sigma/\\sqrt{{N}}$ = {mean_flux_err:.4f})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        _placeholder_panel(ax, 'Mean Flux per Sightline',
                           'per-sightline mean flux')

    # Panel 4: Transmission statistics
    ax = axes[1, 1]
    if flux is not None and tau is not None:
        n_pixels = flux.shape[1]
        transmitted_frac = (flux > 0.1).sum(axis=1) / n_pixels
        saturated_frac = (tau > 5.0).sum(axis=1) / n_pixels

        ax.scatter(transmitted_frac, saturated_frac, alpha=0.5, s=20)
        ax.set_xlabel('Fraction with F > 0.1')
        ax.set_ylabel('Fraction with tau > 5 (saturated)')
        ax.set_title('Transmission vs Saturation')
        ax.grid(True, alpha=0.3)
    else:
        _placeholder_panel(ax, 'Transmission vs Saturation',
                           'per-pixel flux and tau')

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_snapshot_diagnostic(snapshot_path, output_path, stride=100, title=None):
    """Gas projection and neutral-fraction histogram from a raw snapshot.

    stride subsamples the particles. Note that the CAMELS snapshots store
    PartType0 gzip-chunked, so a strided read still decompresses every chunk:
    the cost is the full dataset regardless of stride, ~0.5 GB per snapshot.
    """
    import h5py

    with h5py.File(snapshot_path, 'r') as f:
        coords = f['PartType0/Coordinates'][::stride]
        nH = f['PartType0/NeutralHydrogenAbundance'][::stride]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    h1 = axes[0].hist2d(coords[:, 0], coords[:, 1], bins=200,
                        cmap='viridis', cmin=0)
    axes[0].set_xlabel('X [ckpc/h]')
    axes[0].set_ylabel('Y [ckpc/h]')
    axes[0].set_title('Gas Density Projection (xy plane)')
    plt.colorbar(h1[3], ax=axes[0], label='N particles per bin')

    axes[1].hist(np.log10(nH + 1e-10), bins=100, color='steelblue',
                 edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('log10(Neutral Fraction)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Neutral Hydrogen Distribution')
    axes[1].grid(alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    save_plot(fig, output_path)
    plt.close(fig)

    return coords.shape[0]


def _placeholder_panel(ax, title, needed):
    """Empty axes stating what data the panel would have needed."""
    ax.text(0.5, 0.5, f'Not reproducible from CSVs\n(needs {needed})',
            ha='center', va='center', transform=ax.transAxes,
            fontsize=10, color='gray')
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


#########################################
# COMPARISON/OVERLAY PLOTTING FUNCTIONS #
#########################################

def plot_power_spectrum_overlay(power_dicts, labels, output_path, redshift=None, 
                                  fiducial_idx=None, title=None):
    """Plot power spectra from multiple simulations overlaid with optional ratio panel."""
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    import numpy as np
    
    setup_plot_style()
    
    # Create figure with two panels: power spectrum and ratio
    if fiducial_idx is not None:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), 
                                 gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})
        ax_main, ax_ratio = axes
    else:
        fig, ax_main = plt.subplots(1, 1, figsize=(10, 6))
        ax_ratio = None
    
    # Get colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(power_dicts)))
    
    # Plot each power spectrum
    for i, (power_dict, label) in enumerate(zip(power_dicts, labels)):
        k = power_dict['k']
        kPk_pi = power_dict.get('kPk_pi', k * power_dict['P_k_mean'] / np.pi)
        
        # Plot main data
        linestyle = '--' if i == fiducial_idx else '-'
        linewidth = 2.5 if i == fiducial_idx else 1.5
        ax_main.plot(k, kPk_pi, label=label, color=colors[i], 
                     linestyle=linestyle, linewidth=linewidth, alpha=0.8)
        
        # Add error bars if available
        if 'kPk_pi_err' in power_dict:
            ax_main.fill_between(k, 
                                  kPk_pi - power_dict['kPk_pi_err'],
                                  kPk_pi + power_dict['kPk_pi_err'],
                                  color=colors[i], alpha=0.2)
        
        # Plot ratio if fiducial is specified
        if ax_ratio is not None and fiducial_idx is not None and i != fiducial_idx:
            k_fid = power_dicts[fiducial_idx]['k']
            kPk_pi_fid = power_dicts[fiducial_idx].get('kPk_pi', 
                                                         k_fid * power_dicts[fiducial_idx]['P_k_mean'] / np.pi)
            
            # Interpolate to match k values if needed
            if len(k) != len(k_fid) or not np.allclose(k, k_fid):
                kPk_pi_fid_interp = np.interp(k, k_fid, kPk_pi_fid)
            else:
                kPk_pi_fid_interp = kPk_pi_fid
            
            ratio = kPk_pi / kPk_pi_fid_interp
            ax_ratio.plot(k, ratio, color=colors[i], linewidth=1.5, alpha=0.8)
    
    # Format main axis
    ax_main.set_xscale('log')
    ax_main.set_yscale('log')
    if ax_ratio is None:
        ax_main.set_xlabel('k [s/km]', fontsize=12)
    else:
        ax_main.set_xticklabels([])
    ax_main.set_ylabel('k P(k) / π', fontsize=12)
    ax_main.grid(True, alpha=0.3, which='both')
    ax_main.legend(fontsize=9, loc='best', framealpha=0.9)
    
    if title:
        ax_main.set_title(title, fontsize=14)
    elif redshift is not None:
        ax_main.set_title(f'Flux Power Spectrum (z = {redshift:.3f})', fontsize=14)
    
    # Format ratio axis
    if ax_ratio is not None:
        ax_ratio.axhline(1, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
        ax_ratio.set_xscale('log')
        ax_ratio.set_xlabel('k [s/km]', fontsize=12)
        ax_ratio.set_ylabel(f'Ratio to\n{labels[fiducial_idx]}', fontsize=10)
        ax_ratio.grid(True, alpha=0.3, which='both')
        ax_ratio.set_ylim([0.8, 1.2])
    
    plt.tight_layout()
    save_plot(fig, output_path)
    plt.close()


def plot_cddf_overlay(cddf_dicts, labels, output_path, redshift=None, title=None):
    """
    Plot column density distribution functions overlaid for comparison.
    
    Uses properly normalized f(N) in units of [Mpc^-1].
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    setup_plot_style()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Get colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(cddf_dicts)))
    
    # Plot each CDDF
    for i, (cddf_dict, label) in enumerate(zip(cddf_dicts, labels)):
        log_N = np.asarray(cddf_dict['log10_N_HI'])
        f_N = np.asarray(cddf_dict['f_N'])

        # Only plot non-zero values
        mask = f_N > 0

        ax.plot(log_N[mask], f_N[mask], label=label, color=colors[i],
                linewidth=2, alpha=0.8, marker='o', markersize=4)

        err = cddf_dict.get('f_N_HI_err')
        if err is None and cddf_dict.get('counts') is not None:
            c = np.asarray(cddf_dict['counts'], dtype=float)
            with np.errstate(divide='ignore', invalid='ignore'):
                err = np.where(c > 0, f_N / np.sqrt(np.maximum(c, 1.0)), np.nan)
        if err is not None:
            err = np.asarray(err, dtype=float)
            ax.fill_between(log_N[mask],
                            np.clip(f_N[mask] - err[mask], 1e-300, None),
                            f_N[mask] + err[mask],
                            color=colors[i], alpha=0.18, lw=0)
    
    # Format axis
    ax.set_xlabel(r'log$_{10}$(N$_{\rm HI}$ [cm$^{-2}$])', fontsize=12)
    ax.set_ylabel(r'$f(N_{\rm HI})$ [Mpc$^{-1}$]', fontsize=12)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=10, loc='best', framealpha=0.9)
    
    if title:
        ax.set_title(title, fontsize=14)
    elif redshift is not None:
        ax.set_title(f'Column Density Distribution (z = {redshift:.3f})', fontsize=14)
    
    plt.tight_layout()
    save_plot(fig, output_path)
    plt.close()


def plot_flux_stats_comparison(stats_list, labels, output_path, redshift=None, title=None):
    """Plot comparison of flux statistics as bar charts."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    setup_plot_style()
    
    # Select key statistics to plot
    # effective_tau, not mean_tau: <tau> is dominated by saturated pixels and is a
    # saturation artefact, not an observable.
    key_stats = ['mean_flux', 'median_flux', 'effective_tau', 'weak_absorption_frac']
    stat_labels = ['Mean Flux', 'Median Flux', r'$\tau_{\rm eff}$', 'Weak Abs. Frac.']
    stat_errs = ['mean_flux_err', None, 'tau_eff_err', None]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(stats_list)))
    x_pos = np.arange(len(labels))
    width = 0.8
    
    for idx, (stat_key, stat_label) in enumerate(zip(key_stats, stat_labels)):
        ax = axes[idx]

        # Extract values
        values = [stats.get(stat_key, np.nan) for stats in stats_list]

        err_key = stat_errs[idx]
        errs = None
        if err_key is not None:
            errs = [stats.get(err_key, np.nan) for stats in stats_list]
            if not np.any(np.isfinite(np.asarray(errs, dtype=float))):
                errs = None

        # Create bar chart
        bars = ax.bar(x_pos, values, width, yerr=errs, capsize=4,
                      color=colors, alpha=0.7, edgecolor='black')
        
        # Format
        ax.set_ylabel(stat_label, fontsize=11)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    if title:
        fig.suptitle(title, fontsize=14, y=0.98)
    elif redshift is not None:
        fig.suptitle(f'Flux Statistics Comparison (z = {redshift:.3f})', fontsize=14, y=0.98)
    
    plt.tight_layout()
    save_plot(fig, output_path)
    plt.close()


def plot_tau_eff_comparison(tau_eff_list, labels, output_path, redshift=None, title=None):
    """Plot comparison of effective optical depths with error bars."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    setup_plot_style()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(tau_eff_list)))
    x_pos = np.arange(len(labels))
    
    # Extract values and errors
    tau_eff_vals = [tau_dict['tau_eff'] for tau_dict in tau_eff_list]
    tau_eff_errs = [tau_dict.get('tau_eff_err', 0) for tau_dict in tau_eff_list]
    tau_eff_errs = [err if err is not None else 0 for err in tau_eff_errs]  # Handle None
    
    # Create bar chart with error bars
    bars = ax.bar(x_pos, tau_eff_vals, color=colors, alpha=0.7, 
                  edgecolor='black', linewidth=1.5)
    
    # Only show error bars if at least one is non-zero
    if any(err > 0 for err in tau_eff_errs):
        ax.errorbar(x_pos, tau_eff_vals, yerr=tau_eff_errs, fmt='none', 
                    color='black', capsize=5, linewidth=2)
    
    # Format
    ax.set_ylabel('τ_eff', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, val, err) in enumerate(zip(bars, tau_eff_vals, tau_eff_errs)):
        height = bar.get_height()
        if err > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + err,
                   f'{val:.3f}±{err:.3f}', ha='center', va='bottom', fontsize=9)
        else:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    if title:
        ax.set_title(title, fontsize=14)
    elif redshift is not None:
        ax.set_title(f'Effective Optical Depth (z = {redshift:.3f})', fontsize=14)
    
    plt.tight_layout()
    save_plot(fig, output_path)
    plt.close()


def plot_sample_spectra_comparison(flux_arrays, labels, velocity, output_path, 
                                     n_samples=5, redshift=None, title=None):
    """Plot sample spectra from multiple simulations for same sightlines."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    setup_plot_style()
    
    n_sims = len(flux_arrays)
    n_samples = min(n_samples, flux_arrays[0].shape[0])
    
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 2.5 * n_samples), sharex=True)
    
    if n_samples == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_sims))
    
    # Select random sightlines (same for all simulations)
    np.random.seed(42)
    sample_indices = np.random.choice(flux_arrays[0].shape[0], n_samples, replace=False)
    
    for i, idx in enumerate(sample_indices):
        ax = axes[i]
        
        for j, (flux, label) in enumerate(zip(flux_arrays, labels)):
            alpha = 0.8 if j == 0 else 0.6
            linewidth = 2 if j == 0 else 1.5
            ax.plot(velocity, flux[idx], label=label, color=colors[j], 
                   alpha=alpha, linewidth=linewidth)
        
        ax.set_ylabel('Flux', fontsize=10)
        ax.set_ylim([-0.05, 1.1])
        ax.grid(True, alpha=0.3)
        ax.text(0.98, 0.95, f'Sightline {idx}', transform=ax.transAxes,
               ha='right', va='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        if i == 0:
            ax.legend(fontsize=8, loc='upper left', ncol=min(3, n_sims))
    
    axes[-1].set_xlabel('Velocity [km/s]', fontsize=12)
    
    if title:
        fig.suptitle(title, fontsize=14, y=0.995)
    elif redshift is not None:
        fig.suptitle(f'Sample Spectra Comparison (z = {redshift:.3f})', fontsize=14, y=0.995)
    
    plt.tight_layout()
    save_plot(fig, output_path)
    plt.close()
