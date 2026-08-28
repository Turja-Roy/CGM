"""
Tier 1 hypothesis test: does 'less Ω0 -> less feedback/reionization -> more HI'
explain the p1 CDDF / tau_eff inversion in the 1P scan?

Consumes the per-variant CSVs that `analyze_spectra.py analyze` already writes
under output/analysis/<suite>/1P/1P_p{idx}_{n2,n1,0,1,2}/snap-{XXX}/:
    cddf.csv, flux_stats.csv, power_spectrum.csv, temp_density.csv,
    line_widths.csv  (snap-080 only)

Reads the CosmoAstroSeed CSV to map variant label -> parameter value.

Does NOT touch raw .hdf5 snapshots. Designed to run on HPC in the directory tree
where these CSVs live. Run:

    python scripts/hypothesis_test_p1.py \\
        --analysis-root output/analysis/IllustrisTNG/1P \\
        --cosmo-csv data/IllustrisTNG/1P/CosmoAstroSeed_IllustrisTNG_L25n256_1P.csv \\
        --snaps snap-080,snap-014 \\
        --out-dir plots/hypothesis_p1_test

The script is tolerant: if a file is missing for a given variant/snap, that
variant is skipped for that test and a warning is printed. A single variant
missing will never kill the whole run.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------------------------------------------- #
# Scan definitions: which 1P parameter axis each group varies, and the #
# CosmoAstroSeed column that holds the varied value. Only p1..p5 here  #
# because those are the ones relevant to the hypothesis.               #
# -------------------------------------------------------------------- #
SCANS = {
    'p1': {'column': 'Omega0',       'label': r'$\Omega_0$',       'direction_pred': 'down'},
    'p2': {'column': 'sigma8',       'label': r'$\sigma_8$',       'direction_pred': 'down'},
    'p7': {'column': 'OmegaBaryon',  'label': r'$\Omega_b$',       'direction_pred': 'up'},
    'p8': {'column': 'HubbleParam',  'label': r'$h$',              'direction_pred': 'mild'},
    'p9': {'column': 'n_s',          'label': r'$n_s$',            'direction_pred': 'down'},
}
# direction_pred: predicted direction of tau_eff ratio when the parameter is
# raised above fiducial, under the "structure-growth drives feedback" hypothesis.
# 'down' = tau_eff falls (less HI). 'up' = tau_eff rises (more HI). 'mild' = small.
VARIANT_SUFFIXES = ['n2', 'n1', '0', '1', '2']   # fixed order; '0' is fiducial
FIDUCIAL = '0'


# =====================================================================
# CSV loaders
# =====================================================================

def _read_csv_or_none(path, **kwargs):
    """pd.read_csv that returns None for a missing or empty/header-only file
    instead of raising EmptyDataError."""
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.EmptyDataError:
        return None


def _parse_headered_csv(path):
    """Read a CSV that begins with '# key = value' lines. Return (header_dict, dataframe)."""
    header = {}
    data_start = 0
    with open(path, 'r') as fh:
        for i, line in enumerate(fh):
            if not line.startswith('#'):
                data_start = i
                break
            s = line[1:].strip()
            if not s or '=' not in s:
                continue
            k, v = s.split('=', 1)
            k, v = k.strip(), v.strip().split(' ')[0]
            try:
                header[k] = float(v)
            except ValueError:
                header[k] = v
    df = _read_csv_or_none(path, skiprows=data_start)
    return header, df


def load_temp_density(path):
    """Return T0, gamma, their errors and n_pixels from the comment header."""
    out = {'T0': np.nan, 'gamma': np.nan, 'gamma_err': np.nan, 'T0_err': np.nan,
           'n_bins_fit': np.nan, 'n_pixels': np.nan}
    if not path.exists():
        return out
    with open(path, 'r') as fh:
        for line in fh:
            if not line.startswith('#'):
                break
            s = line[1:].strip()
            for k in out:
                if s.startswith(k + ' '):
                    _, _, v = s.partition('=')
                    v = v.strip().split(' ')[0]
                    try:
                        out[k] = float(v)
                    except ValueError:
                        pass
    return out


def load_flux_stats(path):
    """Return a dict statistic -> value."""
    df = _read_csv_or_none(path)
    if df is None:
        return {}
    return dict(zip(df['statistic'], df['value']))


def load_cddf(path):
    """Return (header_dict, dataframe) for the CDDF file."""
    if not path.exists():
        return {}, None
    return _parse_headered_csv(path)


def load_power_spectrum(path):
    return _read_csv_or_none(path)


def load_line_widths(path):
    return _read_csv_or_none(path)


# =====================================================================
# Cosmology helpers
# =====================================================================

def hubble_ratio(z, Omega_m, Omega_L=None):
    """E(z) = H(z)/H_0 for flat wCDM with w=-1."""
    if Omega_L is None:
        Omega_L = 1.0 - Omega_m
    return np.sqrt(Omega_m * (1.0 + z) ** 3 + Omega_L)


def dXdz(z, Omega_m):
    """Cosmological absorption distance path dX/dz = (1+z)^2 / E(z)."""
    return (1.0 + z) ** 2 / hubble_ratio(z, Omega_m)


# =====================================================================
# Variant discovery
# =====================================================================

def variant_dir(analysis_root, scan, suffix):
    return analysis_root / f'1P_{scan}_{suffix}'


def _param_label(rows):
    """Axis label for whichever parameter these rows scan."""
    return rows[0]['param_label']


def snap_dir(analysis_root, scan, suffix, snap):
    return variant_dir(analysis_root, scan, suffix) / snap


def load_cosmo_table(cosmo_csv):
    df = pd.read_csv(cosmo_csv)
    df = df.set_index('Name')
    return df


# =====================================================================
# Assemble per-variant data for one scan at one snap
# =====================================================================

def load_snap_row(d):
    """Every CSV under one snap-XXX directory, as the row dict the observable
    extractors expect. Knows nothing about which set or scan it came from, so
    both the 1P scans and the EX set load through the same code path.

    Missing files leave fields as np.nan rather than raising, so a hole shows up
    in the figures instead of killing the run.
    """
    row = {'snap_dir': d}

    td = load_temp_density(d / 'temp_density.csv')
    row.update(td)

    fs = load_flux_stats(d / 'flux_stats.csv')
    row['mean_flux']  = fs.get('mean_flux',     np.nan)
    row['tau_eff']    = fs.get('effective_tau', np.nan)
    row['median_flux'] = fs.get('median_flux',  np.nan)
    row['deep_frac']  = fs.get('deep_absorption_frac', np.nan)
    row['weak_frac']  = fs.get('weak_absorption_frac', np.nan)
    # _err is sigma/sqrt(N) on the ensemble value; _std is the per-sightline
    # spread. Neither contains cosmic variance: one box, one realisation.
    row['tau_eff_err']   = fs.get('tau_eff_err',   np.nan)
    row['tau_eff_std']   = fs.get('tau_eff_std',   np.nan)
    row['mean_flux_err'] = fs.get('mean_flux_err', np.nan)
    row['mean_flux_std'] = fs.get('mean_flux_std', np.nan)
    # NaN outside 1.7 < z < 4 (rescaling) or without the lya_h line.
    row['tau_scale_factor'] = fs.get('tau_scale_factor', np.nan)
    row['tau_eff_H_total']  = fs.get('tau_eff_H_total',  np.nan)
    row['hi_fraction_eff']  = fs.get('hi_fraction_eff',  np.nan)

    cddf_hdr, cddf_df = load_cddf(d / 'cddf.csv')
    row['redshift']    = cddf_hdr.get('redshift',    np.nan)
    row['dX_file']     = cddf_hdr.get('dX',          np.nan)
    row['n_absorbers'] = cddf_hdr.get('n_absorbers', np.nan)
    row['beta_fit']    = cddf_hdr.get('beta_fit',     np.nan)
    row['beta_fit_err'] = cddf_hdr.get('beta_fit_err', np.nan)
    row['cddf']        = cddf_df

    row['power_spectrum'] = load_power_spectrum(d / 'power_spectrum.csv')
    row['line_widths']    = load_line_widths(d / 'line_widths.csv')

    return row


def build_scan_frame(analysis_root, cosmo_table, scan, snap):
    """Return list of dicts, one per available variant, in the order n2..2.

    Each dict carries:
      label, suffix, param_value, redshift, T0, gamma, n_pixels,
      mean_flux, tau_eff, n_absorbers, dX_file, paths ...
    Missing files leave fields as np.nan but the row is still included so you
    can see holes.
    """
    col = SCANS[scan]['column']
    rows = []
    for suffix in VARIANT_SUFFIXES:
        run_label = f'1P_{scan}_{suffix}'
        row = load_snap_row(snap_dir(analysis_root, scan, suffix, snap))
        row.update({
            'suffix': suffix,
            'label': run_label,
            'scan': scan,
            'param_label': SCANS[scan]['label'],
            'param_value': (cosmo_table.loc[run_label, col]
                            if run_label in cosmo_table.index else np.nan),
            # NOT param_value: E(z) and dX/dz need the real matter density, which
            # every scan but p1 holds at the fiducial.
            'omega0': (cosmo_table.loc[run_label, 'Omega0']
                       if run_label in cosmo_table.index else np.nan),
        })
        rows.append(row)
    return rows


# =====================================================================
# Plots
# =====================================================================

def _setup_style():
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 9


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


def plot_t1_1_thermal_trend(rows, out_path, snap_label):
    """T1.1: T0, gamma, n_pixels vs the scanned parameter."""
    plab = _param_label(rows)
    x   = np.array([r['param_value'] for r in rows], dtype=float)
    T0  = np.array([r['T0']          for r in rows], dtype=float)
    T0e = np.array([r.get('T0_err', np.nan) for r in rows], dtype=float)
    g   = np.array([r['gamma']       for r in rows], dtype=float)
    gerr= np.array([r['gamma_err']   for r in rows], dtype=float)
    npx = np.array([r['n_pixels']    for r in rows], dtype=float)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))

    ax0.errorbar(x, T0 / 1e3, yerr=T0e / 1e3, fmt='o-', color='C0', lw=2, ms=7,
                 capsize=3)
    for xi, Ti, r in zip(x, T0, rows):
        if np.isfinite(Ti):
            ax0.annotate(r['suffix'], (xi, Ti / 1e3),
                         xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax0.set_xlabel(plab)
    ax0.set_ylabel(r'$T_0$  [$10^3$ K]')
    ax0.set_title(f'IGM temperature at mean density ({snap_label})')
    ax0.grid(alpha=0.3)

    ax0b = ax0.twinx()
    ax0b.plot(x, npx / 1e6, 's--', color='C3', lw=1.2, ms=5, alpha=0.7,
              label='n_pixels in TDR fit')
    ax0b.set_ylabel(r'n_pixels$_{\rm diffuse}$ [$10^6$]', color='C3')
    ax0b.tick_params(axis='y', labelcolor='C3')

    ax1.errorbar(x, g, yerr=gerr, fmt='o-', color='C2', lw=2, ms=7, capsize=3)
    ax1.set_xlabel(plab)
    ax1.set_ylabel(r'$\gamma$ (TDR slope)')
    ax1.set_title(f'TDR slope ({snap_label})')
    ax1.grid(alpha=0.3)

    fig.suptitle('T1.1 — Thermal state of the diffuse IGM')
    fig.tight_layout()
    _save(fig, out_path)


def plot_t1_2_pathlength(rows, out_path, snap_label):
    """T1.2: apply the analytic dX(Omega_0)/dX(fid) correction to each CDDF
    and show the ordering is preserved. Off the p1 scan Omega_0 is fixed, so the
    correction is 1 and the right panel reproduces the left -- no artefact.
    """
    plab = _param_label(rows)
    fid = next(r for r in rows if r['suffix'] == FIDUCIAL)
    z_fid = fid['redshift']
    if not np.isfinite(z_fid):
        print('  [T1.2] fiducial redshift missing, skipping'); return

    dX_fid = dXdz(z_fid, fid['omega0'])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(rows)))

    for r, c in zip(rows, colors):
        cddf = r['cddf']
        if cddf is None or np.isnan(r['param_value']):
            continue
        mask = cddf['f_N_HI'] > 0
        lbl = f"{r['suffix']} ({plab}={r['param_value']:.2f})"
        err = _cddf_poisson_err(cddf)

        dX_var = dXdz(r['redshift'], r['omega0'])
        corr   = dX_var / dX_fid

        for ax, denom in ((axL, 1.0), (axR, corr)):
            y = cddf['f_N_HI'][mask] / denom
            ax.plot(cddf['log10_N_HI'][mask], y, 'o-', color=c, lw=2, ms=4,
                    label=lbl, alpha=0.85)
            if err is not None:
                e = err[mask] / denom
                ax.fill_between(cddf['log10_N_HI'][mask],
                                np.clip(y - e, 1e-300, None), y + e,
                                color=c, alpha=0.18, lw=0)

    for ax, title in [(axL, 'As-published CDDF'),
                      (axR, r'after $\times$ dX(fid)/dX($\Omega_0$)')]:
        ax.set_yscale('log')
        ax.set_xlabel(r'$\log_{10}\, N_{\rm HI}\,[{\rm cm}^{-2}]$')
        ax.set_ylabel(r'$f(N_{\rm HI})$  [Mpc$^{-1}$]')
        ax.set_xlim(12, 16)
        ax.grid(alpha=0.3, which='both')
        ax.set_title(title)
        ax.legend(fontsize=8, loc='best')

    fig.suptitle(f'T1.2 — CDDF path-length control ({snap_label})')
    fig.tight_layout()
    _save(fig, out_path)


def plot_t1_3_fgpa(rows, out_path, snap_label):
    """T1.3: compare measured tau_eff ratio to the FGPA thermal-only prediction.

    FGPA: tau ~ Delta^(2-0.7(gamma-1)) * T0^-0.7 / H(z) * Gamma_HI^-1 * (Omega_b h^2)^2
    With Omega_b, h, Gamma_HI fixed (external UVB), the variant-to-variant ratio
    at fixed density Delta=1 reduces to:
        R_pred = (T0_fid/T0)^0.7 * (H_fid/H)
    The measured ratio is R_meas = tau_eff / tau_eff_fid.
    R_meas - R_pred is the part NOT accounted for by thermal+Hubble — i.e. the
    contribution of the absorber population itself (structure/feedback).
    """
    fid = next(r for r in rows if r['suffix'] == FIDUCIAL)
    if not np.isfinite(fid['T0']) or not np.isfinite(fid['tau_eff']):
        print('  [T1.3] fiducial T0/tau_eff missing, skipping'); return

    plab = _param_label(rows)
    T0_fid = fid['T0']
    H_fid  = hubble_ratio(fid['redshift'], fid['omega0'])
    tau_fid= fid['tau_eff']

    tau_fid_err = fid.get('tau_eff_err', np.nan)

    x = np.array([r['param_value'] for r in rows], dtype=float)
    R_meas, R_pred, R_meas_err = [], [], []
    for r in rows:
        if not np.isfinite(r['T0']) or not np.isfinite(r['tau_eff']):
            R_meas.append(np.nan); R_pred.append(np.nan)
            R_meas_err.append(np.nan); continue
        H_v = hubble_ratio(r['redshift'], r['omega0'])
        R_pred.append((T0_fid / r['T0']) ** 0.7 * (H_fid / H_v))
        ratio = r['tau_eff'] / tau_fid
        R_meas.append(ratio)
        # In quadrature with the fiducial's own error, so the fiducial point does
        # not come out with a zero bar.
        rel_v = r.get('tau_eff_err', np.nan) / r['tau_eff']
        rel_f = tau_fid_err / tau_fid
        R_meas_err.append(ratio * np.sqrt(rel_v ** 2 + rel_f ** 2))
    R_meas = np.array(R_meas); R_pred = np.array(R_pred)
    R_meas_err = np.array(R_meas_err, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(x, R_meas, yerr=R_meas_err, fmt='o-', lw=2, ms=8, capsize=3,
                label=r'measured $\tau_{\rm eff}/\tau_{\rm eff, fid}$')
    ax.plot(x, R_pred, 's--', lw=2, ms=8, label=r'FGPA thermal-only prediction')
    ax.axhline(1.0, color='gray', lw=0.8, ls=':')
    ax.set_xlabel(plab)
    ax.set_ylabel('ratio to fiducial')
    ax.set_yscale('log')
    ax.set_title(f'T1.3 — FGPA thermal-only vs. measured ({snap_label})')
    ax.grid(alpha=0.3, which='both')
    ax.legend()
    fig.tight_layout()
    _save(fig, out_path)

    return {'param_value': x.tolist(),
            'R_measured': R_meas.tolist(),
            'R_fgpa_pred': R_pred.tolist(),
            'residual': (R_meas / R_pred).tolist()}


def plot_t1_4_power_spectrum_scale_split(rows, out_path, snap_label,
                                          k_large_max=0.01, k_small_min=0.05):
    """T1.4: integrate k*P(k) in a large-scale band and a small-scale band,
    plot both vs parameter value."""
    plab = _param_label(rows)
    x = np.array([r['param_value'] for r in rows], dtype=float)
    large, small = [], []
    for r in rows:
        ps = r['power_spectrum']
        if ps is None:
            large.append(np.nan); small.append(np.nan); continue
        k = ps['k_s_per_km'].values
        P = ps['P_k_mean_km_per_s'].values
        kP = k * P
        m_L = (k > 0) & (k <= k_large_max)
        m_S = k >= k_small_min
        _trapz = getattr(np, 'trapezoid', None) or np.trapz
        large.append(_trapz(kP[m_L], k[m_L]) if m_L.sum() > 1 else np.nan)
        small.append(_trapz(kP[m_S], k[m_S]) if m_S.sum() > 1 else np.nan)
    large = np.array(large); small = np.array(small)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, large / np.nanmax(large), 'o-', lw=2, ms=7,
            label=rf'large-scale ($k \leq {k_large_max}$ s/km)')
    ax.plot(x, small / np.nanmax(small), 's--', lw=2, ms=7,
            label=rf'small-scale ($k \geq {k_small_min}$ s/km)')
    ax.set_xlabel(plab)
    ax.set_ylabel(r'$\int k P(k) dk$, normalized to max')
    ax.set_title(f'T1.4 — Flux power in two k-bands ({snap_label})')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save(fig, out_path)

    return {'param_value': x.tolist(),
            'large_scale_integral': large.tolist(),
            'small_scale_integral': small.tolist()}


def plot_t1_6_bparam(rows, out_path, snap_label):
    """T1.6: b-parameter median + distribution vs the scanned parameter."""
    plab = _param_label(rows)
    x   = []
    med = []
    p25 = []
    p75 = []
    bs_by_var = []
    for r in rows:
        lw = r['line_widths']
        if lw is None:
            x.append(r['param_value']); med.append(np.nan)
            p25.append(np.nan); p75.append(np.nan); bs_by_var.append(None); continue
        b = lw['b_param_km_s'].values
        b = b[np.isfinite(b) & (b > 0)]
        x.append(r['param_value'])
        med.append(np.median(b))
        p25.append(np.percentile(b, 25))
        p75.append(np.percentile(b, 75))
        bs_by_var.append(b)

    x = np.array(x, dtype=float)
    med = np.array(med); p25 = np.array(p25); p75 = np.array(p75)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))

    axL.plot(x, med, 'o-', lw=2, ms=8, label='median')
    axL.fill_between(x, p25, p75, alpha=0.25, label='25–75 %')
    axL.set_xlabel(plab)
    axL.set_ylabel('b-parameter [km/s]')
    axL.set_title(f'b-param vs {plab}')
    axL.grid(alpha=0.3); axL.legend()

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(rows)))
    for r, c, b in zip(rows, colors, bs_by_var):
        if b is None or len(b) == 0:
            continue
        axR.hist(b, bins=np.linspace(0, 150, 80),
                 histtype='step', lw=1.8, color=c,
                 label=f"{r['suffix']} ({plab}={r['param_value']:.2f})",
                 density=True)
    axR.set_xlabel('b-parameter [km/s]')
    axR.set_ylabel('density')
    axR.set_title('b-param distributions')
    axR.grid(alpha=0.3); axR.legend(fontsize=8)

    fig.suptitle(f'T1.6 — Doppler b-parameter ({snap_label})')
    fig.tight_layout()
    _save(fig, out_path)

    return {'param_value': x.tolist(),
            'median_b': med.tolist(),
            'p25_b': p25.tolist(),
            'p75_b': p75.tolist()}


# =====================================================================
# Cross-parameter (T1.5)
# =====================================================================

def plot_t1_5_cross_parameter(analysis_root, cosmo_table, snap, out_path):
    """Overlay tau_eff(parameter) and T0(parameter) for all scans p1..p5,
    each normalized to its own fiducial value. Same direction across scans
    corroborates the feedback/structure-growth story."""
    fig, (axT, axTau) = plt.subplots(1, 2, figsize=(13, 5))

    summary = {}
    for scan, meta in SCANS.items():
        rows = build_scan_frame(analysis_root, cosmo_table, scan, snap)
        fid  = next((r for r in rows if r['suffix'] == FIDUCIAL), None)
        if fid is None or not np.isfinite(fid['tau_eff']):
            print(f'  [T1.5] skip {scan}: fiducial tau_eff missing'); continue

        x = np.array([r['param_value'] for r in rows], dtype=float)
        T = np.array([r['T0']          for r in rows], dtype=float)
        tau = np.array([r['tau_eff']    for r in rows], dtype=float)

        T_rel  = T   / fid['T0']   if np.isfinite(fid['T0']) else np.full_like(T, np.nan)
        tau_rel= tau / fid['tau_eff']

        axT  .plot(x / fid['param_value'], T_rel,  'o-', lw=1.8, ms=6, label=f'{scan} ({meta["label"]})')
        axTau.plot(x / fid['param_value'], tau_rel,'o-', lw=1.8, ms=6, label=f'{scan} ({meta["label"]})')

        summary[scan] = {
            'param_value':  x.tolist(),
            'T0_over_fid':  T_rel.tolist(),
            'tau_over_fid': tau_rel.tolist(),
        }

    for ax, ylabel in [(axT, r'$T_0 / T_{0,{\rm fid}}$'),
                       (axTau, r'$\tau_{\rm eff} / \tau_{\rm eff,fid}$')]:
        ax.axhline(1.0, color='gray', lw=0.8, ls=':')
        ax.axvline(1.0, color='gray', lw=0.8, ls=':')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('parameter value / fiducial')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3, which='both')
        ax.legend(fontsize=9)

    fig.suptitle(f'T1.5 — direction of effect across parameter scans ({snap})')
    fig.tight_layout()
    _save(fig, out_path)
    return summary


# =====================================================================
# Across-snapshot grids: one figure per observable, one panel per snap,
# all five Omega_0 variants overlaid in each panel. Lets you read the
# redshift evolution of the p1 ordering at a glance.
# =====================================================================

def _variant_colors():
    return plt.cm.viridis(np.linspace(0, 0.9, len(VARIANT_SUFFIXES)))


def _grid_axes(n, ncols=3, panel=(4.4, 3.5)):
    """Return (fig, flat_axes_list) with unused trailing axes hidden."""
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel[0] * ncols, panel[1] * nrows),
                             squeeze=False)
    flat = list(axes.ravel())
    for ax in flat[n:]:
        ax.axis('off')
    return fig, flat


def _panel_z(rows):
    fid = next((r for r in rows if r['suffix'] == FIDUCIAL), None)
    z = fid['redshift'] if fid is not None else np.nan
    return z


def _panel_title(rows, snap):
    z = _panel_z(rows)
    return f'{snap}  (z = {z:.2f})' if np.isfinite(z) else snap


def _frames_scan(frames):
    """(scan_name, axis_label) for a {snap: rows} mapping."""
    rows = next(iter(frames.values()))
    return rows[0]['scan'], rows[0]['param_label']


def _cddf_poisson_err(cddf):
    """Per-bin Poisson error on f(N). Falls back to f/sqrt(counts) for CSVs written
    before the f_N_HI_err column existed."""
    if cddf is None:
        return None
    if 'f_N_HI_err' in cddf.columns:
        return cddf['f_N_HI_err']
    if 'counts' in cddf.columns:
        counts = cddf['counts'].astype(float)
        return cddf['f_N_HI'] / np.sqrt(counts.where(counts > 0))
    return None


def _grid_cddf(frames, snaps, out_path):
    scan, plab = _frames_scan(frames)
    colors = _variant_colors()
    fig, flat = _grid_axes(len(snaps))
    for ax, snap in zip(flat, snaps):
        rows = frames[snap]
        for r, c in zip(rows, colors):
            cddf = r['cddf']
            if cddf is None or not np.isfinite(r['param_value']):
                continue
            m = cddf['f_N_HI'] > 0
            ax.plot(cddf['log10_N_HI'][m], cddf['f_N_HI'][m], '-',
                    color=c, lw=1.5, label=f"{r['param_value']:.1f}")
            # The high-N bins hold single-digit counts in a 25 Mpc/h box.
            err = _cddf_poisson_err(cddf)
            if err is not None:
                ax.fill_between(cddf['log10_N_HI'][m],
                                np.clip(cddf['f_N_HI'][m] - err[m], 1e-300, None),
                                cddf['f_N_HI'][m] + err[m],
                                color=c, alpha=0.18, lw=0)
        ax.set_yscale('log')
        ax.set_title(_panel_title(rows, snap))
        ax.set_xlabel(r'$\log_{10}\, N_{\rm HI}$')
        ax.set_ylabel(r'$f(N_{\rm HI})$ [Mpc$^{-1}$]')
        ax.set_xlim(12, 16)
        ax.grid(alpha=0.3, which='both')
    flat[0].legend(title=plab, fontsize=7, loc='best')
    fig.suptitle(f'CDDF vs redshift ({scan} {plab} scan)')
    fig.tight_layout()
    _save(fig, out_path)


def _grid_power(frames, snaps, out_path):
    scan, plab = _frames_scan(frames)
    colors = _variant_colors()
    fig, flat = _grid_axes(len(snaps))
    for ax, snap in zip(flat, snaps):
        rows = frames[snap]
        for r, c in zip(rows, colors):
            ps = r['power_spectrum']
            if ps is None or not np.isfinite(r['param_value']):
                continue
            k = ps['k_s_per_km'].values
            P = ps['P_k_mean_km_per_s'].values
            m = (k > 0) & (P > 0)
            ax.loglog(k[m], P[m], '-', color=c, lw=1.5,
                      label=f"{r['param_value']:.1f}")
            if 'P_k_err' in ps.columns:
                e = ps['P_k_err'].values
                ax.fill_between(k[m], np.clip(P[m] - e[m], 1e-300, None), P[m] + e[m],
                                color=c, alpha=0.18, lw=0)
        ax.set_title(_panel_title(rows, snap))
        ax.set_xlabel(r'$k$ [s/km]')
        ax.set_ylabel(r'$P_F(k)$ [km/s]')
        ax.grid(alpha=0.3, which='both')
    flat[0].legend(title=plab, fontsize=7, loc='best')
    fig.suptitle(f'Flux power spectrum vs redshift ({scan} {plab} scan)')
    fig.tight_layout()
    _save(fig, out_path)


def _grid_scalar(frames, snaps, out_path, key, ylabel, title,
                 scale=1.0, logy=False, err_key=None, band_key=None):
    """One panel per snap: scalar quantity `key` vs the scanned parameter.

    err_key -> error bar (sigma/sqrt(N)); band_key -> shaded band (per-sightline
    sigma). Both internal to one box.
    """
    _, plab = _frames_scan(frames)
    fig, flat = _grid_axes(len(snaps))
    for ax, snap in zip(flat, snaps):
        rows = frames[snap]
        x = np.array([r['param_value'] for r in rows], dtype=float)
        y = np.array([r.get(key, np.nan) for r in rows], dtype=float) * scale
        yerr = None
        if err_key is not None:
            yerr = np.array([r.get(err_key, np.nan) for r in rows], dtype=float) * scale
            if not np.any(np.isfinite(yerr)):
                yerr = None
        if band_key is not None:
            band = np.array([r.get(band_key, np.nan) for r in rows], dtype=float) * scale
            if np.any(np.isfinite(band)):
                ax.fill_between(x, y - band, y + band, color='C0', alpha=0.15,
                                lw=0, label=r'$\pm\sigma$ per sightline')
        ax.errorbar(x, y, yerr=yerr, fmt='o-', color='C0', lw=1.8, ms=6, capsize=3)
        if logy:
            ax.set_yscale('log')
        ax.set_title(_panel_title(rows, snap))
        ax.set_xlabel(plab)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3, which='both')
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, out_path)


def _grid_bparam(frames, snaps, out_path):
    """One panel per snap: median b + 25-75% band vs the scanned parameter.
    Panels with no line_widths data are left blank."""
    scan, plab = _frames_scan(frames)
    have = [s for s in snaps
            if any(frames[s][i]['line_widths'] is not None
                   for i in range(len(frames[s])))]
    if not have:
        print('  [grid] no line_widths anywhere — skipping b-param grid')
        return
    fig, flat = _grid_axes(len(have))
    for ax, snap in zip(flat, have):
        rows = frames[snap]
        x, med, p25, p75 = [], [], [], []
        for r in rows:
            lw = r['line_widths']
            x.append(r['param_value'])
            if lw is None:
                med.append(np.nan); p25.append(np.nan); p75.append(np.nan)
                continue
            b = lw['b_param_km_s'].values
            b = b[np.isfinite(b) & (b > 0)]
            if b.size == 0:
                med.append(np.nan); p25.append(np.nan); p75.append(np.nan)
                continue
            med.append(np.median(b))
            p25.append(np.percentile(b, 25))
            p75.append(np.percentile(b, 75))
        x = np.array(x, float)
        ax.plot(x, med, 'o-', color='C0', lw=1.8, ms=6, label='median')
        ax.fill_between(x, p25, p75, alpha=0.25)
        ax.set_title(_panel_title(rows, snap))
        ax.set_xlabel(plab)
        ax.set_ylabel('b [km/s]')
        ax.grid(alpha=0.3)
    fig.suptitle(f'Doppler b-parameter vs redshift ({scan} {plab} scan)')
    fig.tight_layout()
    _save(fig, out_path)


def _overlay_tau_eff(frames, snaps, out_path):
    """Single panel: tau_eff vs the scanned parameter, one line per snap."""
    _, plab = _frames_scan(frames)
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.plasma(np.linspace(0, 0.85, len(snaps)))
    for snap, c in zip(snaps, cmap):
        rows = frames[snap]
        x = np.array([r['param_value'] for r in rows], dtype=float)
        y = np.array([r['tau_eff']     for r in rows], dtype=float)
        e = np.array([r.get('tau_eff_err', np.nan) for r in rows], dtype=float)
        s = np.array([r.get('tau_eff_std', np.nan) for r in rows], dtype=float)
        z = _panel_z(rows)
        lbl = f'{snap} (z={z:.2f})' if np.isfinite(z) else snap
        if np.any(np.isfinite(s)):
            ax.fill_between(x, y - s, y + s, color=c, alpha=0.12, lw=0)
        ax.errorbar(x, y, yerr=e if np.any(np.isfinite(e)) else None,
                    fmt='o-', color=c, lw=2, ms=6, capsize=3, label=lbl)
    ax.set_yscale('log')
    ax.set_xlabel(plab)
    ax.set_ylabel(r'$\tau_{\rm eff}$')
    ax.set_title(rf'$\tau_{{\rm eff}}$({plab}) across redshift — ordering crossover'
                 '\n' r'bars: $\sigma/\sqrt{N}$   bands: per-sightline $\sigma$')
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out_path)


def _delta_tau_eff_vs_z(frames, snaps, out_path):
    """tau_eff(high parameter) - tau_eff(low parameter) as a function of z.

    Four panels: the raw difference, the same divided by the fiducial (so the
    z = 4 and z = 0.27 points are comparable), and both again after multiplying
    tau_eff by E(z). tau_eff per unit velocity carries a 1/H(z) factor, so a
    difference that vanishes in the bottom row is geometry, not gas.
    """
    _, label = _frames_scan(frames)
    entries = []
    for snap in snaps:
        rows = frames[snap]
        z = _panel_z(rows)
        usable = [r for r in rows
                  if np.isfinite(r['param_value']) and np.isfinite(r['tau_eff'])]
        if not np.isfinite(z) or len(usable) < 2:
            continue
        usable.sort(key=lambda r: r['param_value'])
        entries.append({'z': z, 'lo': usable[0], 'hi': usable[-1], 'rows': usable,
                        'fid': next((r for r in usable if r['suffix'] == FIDUCIAL),
                                    None)})

    if len(entries) < 2:
        print('  [delta-tau_eff] need >= 2 snapshots with two variants - skipping')
        return

    entries.sort(key=lambda e: e['z'])
    z = np.array([e['z'] for e in entries], float)

    def _err(r):
        return r.get('tau_eff_err', np.nan)

    def _Ez(r, zz):
        return hubble_ratio(zz, r['omega0'])

    d_raw, d_raw_err, d_geo, d_geo_err = [], [], [], []
    fid_raw, fid_geo = [], []
    for e, zz in zip(entries, z):
        hi, lo = e['hi'], e['lo']
        d_raw.append(hi['tau_eff'] - lo['tau_eff'])
        d_raw_err.append(np.sqrt(_err(hi) ** 2 + _err(lo) ** 2))

        hi_g = hi['tau_eff'] * _Ez(hi, zz)
        lo_g = lo['tau_eff'] * _Ez(lo, zz)
        d_geo.append(hi_g - lo_g)
        d_geo_err.append(np.sqrt((_err(hi) * _Ez(hi, zz)) ** 2 +
                                 (_err(lo) * _Ez(lo, zz)) ** 2))

        f = e['fid']
        fid_raw.append(f['tau_eff'] if f is not None else np.nan)
        fid_geo.append(f['tau_eff'] * _Ez(f, zz) if f is not None else np.nan)

    d_raw = np.array(d_raw); d_raw_err = np.array(d_raw_err, float)
    d_geo = np.array(d_geo); d_geo_err = np.array(d_geo_err, float)
    fid_raw = np.array(fid_raw); fid_geo = np.array(fid_geo)

    p_lo = entries[0]['lo']['param_value']
    p_hi = entries[0]['hi']['param_value']

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)

    def _bars(ax, y, yerr, color, lbl=None, ms=7):
        ax.errorbar(z, y, yerr=yerr if np.any(np.isfinite(yerr)) else None,
                    fmt='o-', color=color, lw=2, ms=ms, capsize=3, label=lbl)
        ax.axhline(0.0, color='gray', lw=0.8, ls=':')
        ax.grid(alpha=0.3)

    _bars(axes[0][0], d_raw, d_raw_err, 'C0',
          lbl=f'{label} = {p_hi:.2f} minus {p_lo:.2f}')
    axes[0][0].set_ylabel(r'$\Delta \tau_{\rm eff}$')
    axes[0][0].set_title(r'$\tau_{\rm eff}$ difference across the scan')
    axes[0][0].legend(fontsize=9)

    with np.errstate(divide='ignore', invalid='ignore'):
        _bars(axes[0][1], d_raw / fid_raw, d_raw_err / fid_raw, 'C0')
    axes[0][1].set_ylabel(r'$\Delta \tau_{\rm eff} / \tau_{\rm eff}^{\rm fid}$')
    axes[0][1].set_title('same, relative to the fiducial')

    _bars(axes[1][0], d_geo, d_geo_err, 'C1')
    axes[1][0].set_ylabel(r'$\Delta [\tau_{\rm eff} E(z)]$')
    axes[1][0].set_title(r'geometry removed ($\times E(z) = H(z)/H_0$)')

    with np.errstate(divide='ignore', invalid='ignore'):
        _bars(axes[1][1], d_geo / fid_geo, d_geo_err / fid_geo, 'C1')
    axes[1][1].set_ylabel(r'$\Delta [\tau_{\rm eff} E] / [\tau_{\rm eff} E]^{\rm fid}$')
    axes[1][1].set_title('same, relative to the fiducial')

    # Per-variant differences against the fiducial, so the ordering is visible
    # and not just the two extremes.
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(VARIANT_SUFFIXES)))
    for suf, c in zip(VARIANT_SUFFIXES, colors):
        if suf == FIDUCIAL:
            continue
        zv, dv, pv = [], [], np.nan
        for e, zz in zip(entries, z):
            r = next((x for x in e['rows'] if x['suffix'] == suf), None)
            if r is None or e['fid'] is None:
                continue
            zv.append(zz)
            dv.append(r['tau_eff'] - e['fid']['tau_eff'])
            pv = r['param_value']
        if len(zv) >= 2:
            axes[0][0].plot(zv, dv, '-', color=c, lw=1.0, alpha=0.7,
                            label=f'{pv:.2f} - fid')
    axes[0][0].legend(fontsize=8)

    for ax in axes[1]:
        ax.set_xlabel('redshift z')

    fig.suptitle(rf'$\Delta \tau_{{\rm eff}}(z)$ across the {label} scan '
                 r'— bars: $\sigma/\sqrt{N}$, added in quadrature', fontsize=13)
    fig.tight_layout()
    _save(fig, out_path)

    return {'z': z.tolist(),
            'param_low': float(p_lo), 'param_high': float(p_hi),
            'delta_tau_eff': d_raw.tolist(),
            'delta_tau_eff_err': d_raw_err.tolist(),
            'delta_tau_eff_Ez': d_geo.tolist(),
            'delta_tau_eff_Ez_err': d_geo_err.tolist()}


def make_snapshot_grids(analysis_root, cosmo_table, snaps, out_dir, scan='p1'):
    """Build the across-snapshot comparison figures (one panel per snap)."""
    plab = SCANS[scan]['label']
    print(f'\n=== across-snapshot grids, {scan} scan ===')
    frames = {snap: build_scan_frame(analysis_root, cosmo_table, scan, snap)
              for snap in snaps}
    # high-z first so panels read left->right as cosmic time advances backward
    snaps_sorted = sorted(
        snaps,
        key=lambda s: (_panel_z(frames[s]) if np.isfinite(_panel_z(frames[s]))
                       else -np.inf),
        reverse=True)

    grid_dir = out_dir / scan / 'across_snapshots'
    grid_dir.mkdir(parents=True, exist_ok=True)

    _grid_cddf  (frames, snaps_sorted, grid_dir / 'grid_CDDF.png')
    _grid_power (frames, snaps_sorted, grid_dir / 'grid_power_spectrum.png')
    _grid_scalar(frames, snaps_sorted, grid_dir / 'grid_tau_eff.png',
                 key='tau_eff', ylabel=r'$\tau_{\rm eff}$',
                 title=f'Effective optical depth vs redshift ({scan} {plab} scan)',
                 logy=True, err_key='tau_eff_err', band_key='tau_eff_std')
    _grid_scalar(frames, snaps_sorted, grid_dir / 'grid_mean_flux.png',
                 key='mean_flux', ylabel=r'$\langle F \rangle$',
                 title=f'Mean transmitted flux vs redshift ({scan} {plab} scan)',
                 err_key='mean_flux_err', band_key='mean_flux_std')
    _grid_scalar(frames, snaps_sorted, grid_dir / 'grid_T0.png',
                 key='T0', ylabel=r'$T_0$ [$10^3$ K]', scale=1e-3,
                 title=f'IGM $T_0$ vs redshift ({scan} {plab} scan)',
                 err_key='T0_err')
    # NaN above z = 4.5, where the forest is saturated.
    _grid_scalar(frames, snaps_sorted, grid_dir / 'grid_beta.png',
                 key='beta_fit', ylabel=r'$\beta$ (CDDF slope)',
                 title=f'CDDF power-law slope vs redshift ({scan} {plab} scan)',
                 err_key='beta_fit_err')
    _grid_bparam(frames, snaps_sorted, grid_dir / 'grid_bparam.png')
    _overlay_tau_eff(frames, snaps_sorted, grid_dir / f'tau_eff_vs_{scan}_overlay.png')
    delta = _delta_tau_eff_vs_z(frames, snaps_sorted,
                                grid_dir / 'delta_tau_eff_vs_z.png')
    if delta is not None:
        with open(grid_dir / 'delta_tau_eff_vs_z.json', 'w') as fh:
            json.dump(delta, fh, indent=2)
        print(f'  saved {grid_dir / "delta_tau_eff_vs_z.json"}')


# =====================================================================
# Entry
# =====================================================================

def run_one_snap(analysis_root, cosmo_table, snap, out_dir, scan='p1'):
    print(f'\n=== {scan} scan, {snap} ===')
    rows = build_scan_frame(analysis_root, cosmo_table, scan, snap)
    snap_out = out_dir / scan / snap
    snap_out.mkdir(parents=True, exist_ok=True)

    summary = {'scan': scan, 'snap': snap,
               'param': SCANS[scan]['column'], 'variants': []}
    for r in rows:
        summary['variants'].append({
            'suffix': r['suffix'], 'label': r['label'],
            'param_value': r['param_value'],
            'omega0': r['omega0'],
            'redshift': r['redshift'],
            'T0_K': r['T0'], 'gamma': r['gamma'], 'gamma_err': r['gamma_err'],
            'n_pixels_diffuse': r['n_pixels'],
            'mean_flux': r['mean_flux'],
            'tau_eff': r['tau_eff'],
            'deep_absorption_frac': r['deep_frac'],
            'weak_absorption_frac': r['weak_frac'],
            'n_absorbers_cddf': r['n_absorbers'],
            'dX_from_file_Mpc': r['dX_file'],
            'dX_analytic_dzdX': dXdz(r['redshift'], r['omega0'])
                               if np.isfinite(r['redshift']) and np.isfinite(r['omega0']) else np.nan,
        })

    plot_t1_1_thermal_trend(rows, snap_out / 'T1_1_T0_gamma_vs_param.png', snap)
    plot_t1_2_pathlength   (rows, snap_out / 'T1_2_CDDF_pathlength_control.png', snap)
    fgpa = plot_t1_3_fgpa  (rows, snap_out / 'T1_3_FGPA_vs_measured.png', snap)
    ps   = plot_t1_4_power_spectrum_scale_split(rows, snap_out / 'T1_4_power_spectrum_scale_split.png', snap)
    if snap == 'snap-080':
        bp = plot_t1_6_bparam(rows, snap_out / 'T1_6_bparam.png', snap)
        summary['T1_6_bparam'] = bp

    summary['T1_3_fgpa'] = fgpa
    summary['T1_4_power'] = ps
    return summary, snap_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--analysis-root', required=True, type=Path,
                    help='directory containing 1P_pK_{suffix}/snap-XXX/ trees '
                         '(e.g. output/analysis/IllustrisTNG/1P)')
    ap.add_argument('--cosmo-csv', required=True, type=Path,
                    help='CosmoAstroSeed CSV (keyed by Name column)')
    ap.add_argument('--snaps', default='snap-080,snap-014',
                    help='comma-separated snap dirs to process')
    ap.add_argument('--scans', default='p1,p2',
                    help='comma-separated scans to process (any of '
                         + ','.join(SCANS) + '). p2 is what shows that sigma8 '
                         'leaves no 1/H(z) geometric footprint.')
    ap.add_argument('--out-dir', type=Path,
                    default=Path('plots/hypothesis_p1_test'))
    ap.add_argument('--skip-cross-param', action='store_true',
                    help='skip the T1.5 p1..p5 cross-parameter figure')
    ap.add_argument('--skip-grids', action='store_true',
                    help='skip the across-snapshot comparison grids')
    args = ap.parse_args()

    _setup_style()
    cosmo = load_cosmo_table(args.cosmo_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    snaps = [s.strip() for s in args.snaps.split(',') if s.strip()]
    scans = [s.strip() for s in args.scans.split(',') if s.strip()]
    unknown = [s for s in scans if s not in SCANS]
    if unknown:
        ap.error(f'unknown scan(s) {unknown}; known: {list(SCANS)}')

    all_summary = {}
    for scan in scans:
        for snap in snaps:
            s, snap_out = run_one_snap(args.analysis_root, cosmo, snap,
                                       args.out_dir, scan=scan)
            all_summary[(scan, snap)] = s
            with open(snap_out / 'summary.json', 'w') as fh:
                json.dump(s, fh, indent=2, default=float)

    # T1.5 already overlays every scan in one figure, so it is per-snap, not
    # per-scan. Keep it at the top level rather than duplicating it under each.
    if not args.skip_cross_param:
        for snap in snaps:
            snap_out = args.out_dir / 'cross_parameter' / snap
            snap_out.mkdir(parents=True, exist_ok=True)
            xp = plot_t1_5_cross_parameter(
                args.analysis_root, cosmo, snap,
                snap_out / 'T1_5_cross_parameter_direction.png')
            with open(snap_out / 'summary.json', 'w') as fh:
                json.dump({'snap': snap, 'T1_5_cross_parameter': xp}, fh,
                          indent=2, default=float)

    if not args.skip_grids:
        for scan in scans:
            make_snapshot_grids(args.analysis_root, cosmo, snaps,
                                args.out_dir, scan=scan)

    print(f'\nAll outputs under {args.out_dir.resolve()}')


if __name__ == '__main__':
    sys.exit(main())
