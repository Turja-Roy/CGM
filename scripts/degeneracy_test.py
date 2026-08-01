"""
Omega_0 -- sigma_8 degeneracy test (Tier 1, CSV-only).

Goal: find observables that tell Omega_0 (p1) and sigma_8 (p2) apart, given that
both raise the matter-fluctuation amplitude and therefore move most one-point
statistics together along the combination

    S_8 = sigma_8 * sqrt(Omega_m / 0.3).

The degeneracy only breaks for observables sensitive to GEOMETRY or GROWTH RATE
(or the power-spectrum SHAPE), not just to the fluctuation amplitude. Each test
below targets one such handle and overlays the p1 (Omega_0) and p2 (sigma_8)
scans so the difference -- if any -- is read off directly.

Physics, per test:

  D1  S_8 collapse test (the master diagnostic).
      Plot each scalar observable vs S_8 for both scans, each normalized to its
      own fiducial. If the p1 and p2 curves lie on top of each other, that
      observable is purely a function of S_8 -> degenerate. If they separate,
      the observable breaks the degeneracy. This visualizes the whole question
      in one figure.

  D2  Geometric path length dX/dz.
      Omega_0 enters dX/dz = (1+z)^2 / E(z); sigma_8 does NOT (Omega_m is held
      at 0.3 in the p2 scan, so its dX/dz ratio is identically 1). The CDDF
      normalization therefore carries a pure-geometry Omega_0 signature with no
      sigma_8 counterpart -- a clean, fit-free discriminant.

  D3  Power-spectrum shape (scale split).
      k_eq ~ Omega_m h^2 sets the turnover, so Omega_0 tilts the SHAPE of P_F(k);
      sigma_8 raises every mode by the same factor (amplitude only). The
      small/large-scale band ratio should slope with Omega_0 but stay flat with
      sigma_8 once normalized to fiducial.

  D4  Redshift evolution / growth rate.
      f = dlnD/dlna ~ Omega_m^0.55, so the growth factor D(z) has a different
      z-slope for different Omega_0 even at matched sigma_8(z=0). The SLOPE of an
      observable's redshift evolution is an Omega_0 handle sigma_8 cannot mimic.
      Requires >= 2 snapshots.

  D5  Thermal state (T_0, b-parameter).
      Thermal history follows Omega_m (expansion + adiabatic cooling) more than
      sigma_8. T_0 and the Doppler-b width should respond differently to the two
      scans.

  D6  Observable-space map (joint figure).
      Plot a parametric curve of one observable against another (e.g. T_0 vs
      tau_eff, or power ratio vs tau_eff) for each scan. If the p1 and p2 curves
      are NOT collinear, the 2D observable pair breaks the 1D degeneracy even
      though either coordinate alone may not.

Consumes only the per-variant CSVs that `analyze` already writes (cddf.csv,
flux_stats.csv, power_spectrum.csv, temp_density.csv, line_widths.csv). Reuses
the loaders and cosmology helpers from hypothesis_test_p1.py.

Run:
    python scripts/degeneracy_test.py \\
        --analysis-root output/analysis/IllustrisTNG/1P \\
        --cosmo-csv data/IllustrisTNG/1P/CosmoAstroSeed_IllustrisTNG_L25n256_1P.csv \\
        --snaps snap-080,snap-014 \\
        --out-dir plots/degeneracy_test
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Reuse the Tier-1 infrastructure (same scripts/ dir, so a plain import works).
from hypothesis_test_p1 import (
    SCANS, FIDUCIAL, VARIANT_SUFFIXES,
    build_scan_frame, load_cosmo_table,
    dXdz, hubble_ratio,
    _setup_style, _save,
)

# Two scans under test. p1 varies Omega_0 (sigma_8 fixed 0.8);
# p2 varies sigma_8 (Omega_0 fixed 0.3). Fiducial of both: Omega_0=0.3, sigma_8=0.8.
DEGEN_SCANS = ['p1', 'p2']
SCAN_COLOR  = {'p1': 'C0', 'p2': 'C3'}      # Omega_0 blue, sigma_8 red
SCAN_MARKER = {'p1': 'o',  'p2': 's'}
S8_FID = 0.8 * np.sqrt(0.3 / 0.3)           # fiducial S_8 = 0.8


# =====================================================================
# Per-variant scalar observables
# =====================================================================

def _trapz(y, x):
    fn = getattr(np, 'trapezoid', None) or np.trapz
    return fn(y, x)


def scale_split_ratio(ps, k_large_max=0.01, k_small_min=0.05):
    """Return (large_integral, small_integral, small/large ratio) of k*P_F(k).

    The ratio is the shape discriminant: Omega_0 tilts it (via k_eq), sigma_8
    should leave it ~flat (uniform amplitude rescaling cancels in the ratio)."""
    if ps is None:
        return np.nan, np.nan, np.nan
    k = ps['k_s_per_km'].values
    P = ps['P_k_mean_km_per_s'].values
    kP = k * P
    mL = (k > 0) & (k <= k_large_max)
    mS = k >= k_small_min
    L = _trapz(kP[mL], k[mL]) if mL.sum() > 1 else np.nan
    S = _trapz(kP[mS], k[mS]) if mS.sum() > 1 else np.nan
    ratio = S / L if (np.isfinite(L) and L != 0) else np.nan
    return L, S, ratio


def cddf_value(cddf, logN_ref):
    """f(N_HI) interpolated (log-log) at a reference column density."""
    if cddf is None:
        return np.nan
    m = cddf['f_N_HI'] > 0
    x = cddf['log10_N_HI'][m].values
    y = np.log10(cddf['f_N_HI'][m].values)
    if x.size < 2 or not (x.min() <= logN_ref <= x.max()):
        return np.nan
    return 10.0 ** np.interp(logN_ref, x, y)


def cddf_slope(cddf, logN_lo=13.0, logN_hi=15.0):
    """Log-log slope of the CDDF between two columns. Low-N slope tracks the
    density-PDF shape; the high-N anchor tracks the halo-MF tail."""
    flo = cddf_value(cddf, logN_lo)
    fhi = cddf_value(cddf, logN_hi)
    if not (np.isfinite(flo) and np.isfinite(fhi) and flo > 0 and fhi > 0):
        return np.nan
    return (np.log10(fhi) - np.log10(flo)) / (logN_hi - logN_lo)


def cddf_value_err(cddf, logN_ref):
    """Poisson error on f(N_HI) at a reference column, taken from the nearest bin
    rather than interpolated."""
    if cddf is None:
        return np.nan
    m = cddf['f_N_HI'] > 0
    if not m.any():
        return np.nan
    x = cddf['log10_N_HI'][m].values
    if 'f_N_HI_err' in cddf.columns:
        e = cddf['f_N_HI_err'][m].values
    elif 'counts' in cddf.columns:
        c = cddf['counts'][m].values.astype(float)
        with np.errstate(divide='ignore', invalid='ignore'):
            e = np.where(c > 0, cddf['f_N_HI'][m].values / np.sqrt(c), np.nan)
    else:
        return np.nan
    if x.size == 0 or not (x.min() <= logN_ref <= x.max()):
        return np.nan
    return float(e[np.argmin(np.abs(x - logN_ref))])


def cddf_slope_err(cddf, logN_lo=13.0, logN_hi=15.0):
    """Error on the two-point log-log slope, propagated from the two endpoints."""
    flo, fhi = cddf_value(cddf, logN_lo), cddf_value(cddf, logN_hi)
    elo, ehi = cddf_value_err(cddf, logN_lo), cddf_value_err(cddf, logN_hi)
    if not all(np.isfinite(v) and v > 0 for v in (flo, fhi)):
        return np.nan
    if not (np.isfinite(elo) and np.isfinite(ehi)):
        return np.nan
    # d(log10 f) = (1/ln10) df/f
    s_lo = elo / flo / np.log(10.0)
    s_hi = ehi / fhi / np.log(10.0)
    return float(np.sqrt(s_lo ** 2 + s_hi ** 2) / (logN_hi - logN_lo))


def b_median_err(lw):
    """Standard error of the median b, 1.2533 sigma/sqrt(n). Normal approximation;
    the b distribution is skewed, so treat it as indicative."""
    if lw is None:
        return np.nan
    b = lw['b_param_km_s'].values
    b = b[np.isfinite(b) & (b > 0)]
    if b.size < 10:
        return np.nan
    return float(1.2533 * np.std(b) / np.sqrt(b.size))


def b_median(lw):
    if lw is None:
        return np.nan
    b = lw['b_param_km_s'].values
    b = b[np.isfinite(b) & (b > 0)]
    return np.median(b) if b.size else np.nan


# observable name -> (extractor(row), pretty label, prefer-log-y)
def _obs_extractors():
    return {
        'tau_eff':    (lambda r: r['tau_eff'],                         r'$\tau_{\rm eff}$',                False),
        'mean_flux':  (lambda r: r['mean_flux'],                       r'$\langle F\rangle$',              False),
        'T0':         (lambda r: r['T0'],                              r'$T_0$ [K]',                       False),
        'b_median':   (lambda r: b_median(r['line_widths']),          r'median $b$ [km/s]',               False),
        'power_ratio':(lambda r: scale_split_ratio(r['power_spectrum'])[2], r'$P_F$ small/large ratio',    False),
        'cddf_lowN':  (lambda r: cddf_value(r['cddf'], 13.0),          r'$f(N_{\rm HI}{=}10^{13.0})$',     True),
        'cddf_highN': (lambda r: cddf_value(r['cddf'], 15.0),          r'$f(N_{\rm HI}{=}10^{15.0})$',     True),
        'cddf_slope': (lambda r: cddf_slope(r['cddf']),                r'CDDF log-log slope (13$\to$15)',  False),
    }


# 1-sigma errors, all internal to one box (sightline scatter, Poisson counting).
# power_ratio has none: propagating P_k_err through two band integrals needs the
# k-mode covariance, which is not stored.
def _obs_error_extractors():
    return {
        'tau_eff':    lambda r: r.get('tau_eff_err', np.nan),
        'mean_flux':  lambda r: r.get('mean_flux_err', np.nan),
        'T0':         lambda r: r.get('T0_err', np.nan),
        'b_median':   lambda r: b_median_err(r['line_widths']),
        'power_ratio': lambda r: np.nan,
        'cddf_lowN':  lambda r: cddf_value_err(r['cddf'], 13.0),
        'cddf_highN': lambda r: cddf_value_err(r['cddf'], 15.0),
        'cddf_slope': lambda r: cddf_slope_err(r['cddf']),
    }


# =====================================================================
# Assemble a scan record (both cosmo params + observables) for one snap
# =====================================================================

def _S8(omega0, sigma8):
    return sigma8 * np.sqrt(omega0 / 0.3)


def scan_record(analysis_root, cosmo, scan, snap):
    """build_scan_frame + attach Omega_0, sigma_8, S_8 and the scalar obs arrays."""
    rows = build_scan_frame(analysis_root, cosmo, scan, snap)
    for r in rows:
        lab = r['label']
        om = cosmo.loc[lab, 'Omega0'] if lab in cosmo.index else np.nan
        s8 = cosmo.loc[lab, 'sigma8'] if lab in cosmo.index else np.nan
        r['Omega0'], r['sigma8'], r['S8'] = om, s8, _S8(om, s8)

    extr = _obs_extractors()
    rec = {
        'scan': scan, 'snap': snap, 'rows': rows,
        'Omega0': np.array([r['Omega0'] for r in rows], float),
        'sigma8': np.array([r['sigma8'] for r in rows], float),
        'S8':     np.array([r['S8']     for r in rows], float),
        'param':  np.array([r['param_value'] for r in rows], float),
        'obs':    {name: np.array([fn(r) for r in rows], float)
                   for name, (fn, _lbl, _lg) in extr.items()},
        'obs_err': {name: np.array([fn(r) for r in rows], float)
                    for name, fn in _obs_error_extractors().items()},
    }
    fid = next((r for r in rows if r['suffix'] == FIDUCIAL), None)
    rec['z'] = fid['redshift'] if fid is not None else np.nan
    rec['fid_idx'] = next((i for i, r in enumerate(rows)
                           if r['suffix'] == FIDUCIAL), None)
    return rec


def _norm_to_fid(arr, fid_idx):
    if fid_idx is None or not np.isfinite(arr[fid_idx]) or arr[fid_idx] == 0:
        return np.full_like(arr, np.nan)
    return arr / arr[fid_idx]


def _norm_err_to_fid(arr, err, fid_idx):
    """Error on arr/arr[fid], including the fiducial's own error so that point does
    not come out exact."""
    if fid_idx is None or err is None:
        return None
    with np.errstate(divide='ignore', invalid='ignore'):
        v_fid, e_fid = arr[fid_idx], err[fid_idx]
        if not np.isfinite(v_fid) or v_fid == 0:
            return None
        rel = err / arr
        rel_fid = e_fid / v_fid
        out = np.abs(arr / v_fid) * np.sqrt(rel ** 2 + rel_fid ** 2)
    return out if np.any(np.isfinite(out)) else None


# =====================================================================
# D1 -- S_8 collapse test (master diagnostic)
# =====================================================================

def d1_s8_collapse(records, out_path, snap_label):
    """Each observable vs S_8 for p1 and p2, normalized to fiducial. Overlap =
    degenerate (function of S_8 only); separation = degeneracy broken."""
    extr = _obs_extractors()
    names = list(extr.keys())
    ncols = 4
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.6 * nrows),
                             squeeze=False)
    flat = list(axes.ravel())

    split_score = {}
    for ax, name in zip(flat, names):
        _, lbl, logy = extr[name]
        curves = {}
        for scan in DEGEN_SCANS:
            rec = records[scan]
            y = _norm_to_fid(rec['obs'][name], rec['fid_idx'])
            yerr = _norm_err_to_fid(rec['obs'][name],
                                    rec.get('obs_err', {}).get(name),
                                    rec['fid_idx'])
            x = rec['S8']
            # The split score below is only meaningful if the p1/p2 separation
            # exceeds these bars.
            ax.errorbar(x, y, yerr=yerr, fmt=SCAN_MARKER[scan] + '-',
                        color=SCAN_COLOR[scan], lw=1.8, ms=6, capsize=3,
                        label=f'{scan} ({SCANS[scan]["label"]})')
            curves[scan] = (x, y)
        # Quantify separation: RMS gap between p2 and p1 interpolated onto a
        # shared S_8 grid (only where both are finite).
        try:
            x1, y1 = curves['p1']; x2, y2 = curves['p2']
            lo = max(np.nanmin(x1), np.nanmin(x2))
            hi = min(np.nanmax(x1), np.nanmax(x2))
            xs = np.linspace(lo, hi, 25)
            m1 = np.isfinite(x1) & np.isfinite(y1)
            m2 = np.isfinite(x2) & np.isfinite(y2)
            if m1.sum() >= 2 and m2.sum() >= 2 and hi > lo:
                g1 = np.interp(xs, x1[m1], y1[m1])
                g2 = np.interp(xs, x2[m2], y2[m2])
                split_score[name] = float(np.sqrt(np.mean((g2 - g1) ** 2)))
        except Exception:
            pass

        ax.axvline(S8_FID, color='gray', lw=0.8, ls=':')
        ax.axhline(1.0,    color='gray', lw=0.8, ls=':')
        if logy:
            ax.set_yscale('log')
        ax.set_xlabel(r'$S_8 = \sigma_8\sqrt{\Omega_m/0.3}$')
        ax.set_ylabel(lbl + ' / fid')
        sc = split_score.get(name, np.nan)
        tag = f'  (split={sc:.3f})' if np.isfinite(sc) else ''
        ax.set_title(name + tag, fontsize=10)
        ax.grid(alpha=0.3, which='both')
    for ax in flat[len(names):]:
        ax.axis('off')
    flat[0].legend(fontsize=9, loc='best')
    fig.suptitle(f'D1 -- $S_8$ collapse test: degenerate observables overlap '
                 f'({snap_label})', fontsize=13)
    fig.tight_layout()
    _save(fig, out_path)
    return split_score


# =====================================================================
# D2 -- geometric path length dX/dz
# =====================================================================

def d2_geometry(records, out_path, snap_label):
    """Left: dX/dz ratio vs parameter (p2 is flat by construction).
    Right: CDDF before/after the dX correction for p1 vs p2 -- only p1 moves."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))

    for scan in DEGEN_SCANS:
        rec = records[scan]
        rows = rec['rows']
        z = rec['z']
        fid_idx = rec['fid_idx']
        if fid_idx is None or not np.isfinite(z):
            continue
        om = rec['Omega0']
        dX = np.array([dXdz(z, o) if np.isfinite(o) else np.nan for o in om])
        dX_ratio = dX / dX[fid_idx]
        axL.plot(rec['param'] / rec['param'][fid_idx], dX_ratio,
                 SCAN_MARKER[scan] + '-', color=SCAN_COLOR[scan], lw=2, ms=7,
                 label=f'{scan} ({SCANS[scan]["label"]})')

    axL.axhline(1.0, color='gray', lw=0.8, ls=':')
    axL.axvline(1.0, color='gray', lw=0.8, ls=':')
    axL.set_xlabel('parameter / fiducial')
    axL.set_ylabel(r'$dX/dz \,/\, (dX/dz)_{\rm fid}$')
    axL.set_title(r'Geometric path length (only $\Omega_0$ moves it)')
    axL.grid(alpha=0.3); axL.legend()

    # Right: low-N CDDF amplitude raw vs path-length-corrected, both scans.
    width = 0.35
    xpos = np.arange(len(DEGEN_SCANS))
    for j, scan in enumerate(DEGEN_SCANS):
        rec = records[scan]
        fid_idx = rec['fid_idx']
        z = rec['z']
        raw = _norm_to_fid(rec['obs']['cddf_lowN'], fid_idx)
        # path-length correction: f_corr = f * dX(fid)/dX(variant)
        dX = np.array([dXdz(z, o) if np.isfinite(o) and np.isfinite(z)
                       else np.nan for o in rec['Omega0']])
        corr = (dX[fid_idx] / dX) if fid_idx is not None else np.ones_like(dX)
        cor = _norm_to_fid(rec['obs']['cddf_lowN'] * corr, fid_idx)
        # spread (max-min across variants) before vs after correction
        axR.bar(xpos[j] - width / 2, np.nanmax(raw) - np.nanmin(raw),
                width, color=SCAN_COLOR[scan], alpha=0.55,
                label='raw spread' if j == 0 else None)
        axR.bar(xpos[j] + width / 2, np.nanmax(cor) - np.nanmin(cor),
                width, color=SCAN_COLOR[scan], hatch='//', alpha=0.85,
                label='after dX correction' if j == 0 else None)
    axR.set_xticks(xpos)
    axR.set_xticklabels([f'{s}\n({SCANS[s]["label"]})' for s in DEGEN_SCANS])
    axR.set_ylabel(r'spread of $f(10^{13})$/fid across variants')
    axR.set_title('Path-length correction shrinks only the $\\Omega_0$ spread')
    axR.grid(alpha=0.3, axis='y'); axR.legend()

    fig.suptitle(f'D2 -- geometric discriminant ({snap_label})', fontsize=13)
    fig.tight_layout()
    _save(fig, out_path)


# =====================================================================
# D3 -- power-spectrum shape (scale split)
# =====================================================================

def d3_power_shape(records, out_path, snap_label):
    """small/large P_F(k) band ratio vs parameter, normalized to fiducial.
    Omega_0 tilts (k_eq shift); sigma_8 should stay ~flat."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    slopes = {}
    for scan in DEGEN_SCANS:
        rec = records[scan]
        fid_idx = rec['fid_idx']
        y = _norm_to_fid(rec['obs']['power_ratio'], fid_idx)
        x = rec['param'] / rec['param'][fid_idx] if fid_idx is not None else rec['param']
        ax.plot(x, y, SCAN_MARKER[scan] + '-', color=SCAN_COLOR[scan],
                lw=2, ms=7, label=f'{scan} ({SCANS[scan]["label"]})')
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 2:
            slopes[scan] = float(np.polyfit(x[m], y[m], 1)[0])
    ax.axhline(1.0, color='gray', lw=0.8, ls=':')
    ax.axvline(1.0, color='gray', lw=0.8, ls=':')
    ax.set_xlabel('parameter / fiducial')
    ax.set_ylabel(r'(small/large $P_F$ ratio) / fid')
    sub = ', '.join(f'{s} slope={slopes.get(s, np.nan):.2f}' for s in DEGEN_SCANS)
    ax.set_title(f'D3 -- $P_F(k)$ shape tilt ({snap_label})\n{sub}', fontsize=11)
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    _save(fig, out_path)
    return slopes


# =====================================================================
# D5 -- thermal discriminant (T0, b)
# =====================================================================

def d5_thermal(records, out_path, snap_label):
    fig, (axT, axB) = plt.subplots(1, 2, figsize=(13, 5))
    for scan in DEGEN_SCANS:
        rec = records[scan]
        fid_idx = rec['fid_idx']
        x = rec['param'] / rec['param'][fid_idx] if fid_idx is not None else rec['param']
        axT.plot(x, _norm_to_fid(rec['obs']['T0'], fid_idx),
                 SCAN_MARKER[scan] + '-', color=SCAN_COLOR[scan], lw=2, ms=7,
                 label=f'{scan} ({SCANS[scan]["label"]})')
        axB.plot(x, _norm_to_fid(rec['obs']['b_median'], fid_idx),
                 SCAN_MARKER[scan] + '-', color=SCAN_COLOR[scan], lw=2, ms=7,
                 label=f'{scan} ({SCANS[scan]["label"]})')
    for ax, ttl in [(axT, r'$T_0/T_{0,\rm fid}$'),
                    (axB, r'median $b$ / fid')]:
        ax.axhline(1.0, color='gray', lw=0.8, ls=':')
        ax.axvline(1.0, color='gray', lw=0.8, ls=':')
        ax.set_xlabel('parameter / fiducial')
        ax.set_ylabel(ttl)
        ax.grid(alpha=0.3); ax.legend()
    fig.suptitle(f'D5 -- thermal state response ({snap_label})', fontsize=13)
    fig.tight_layout()
    _save(fig, out_path)


# =====================================================================
# D6 -- observable-space map (joint figure)
# =====================================================================

def d6_observable_space(records, out_path, snap_label):
    """Parametric curves in two observable planes. Non-collinear p1 vs p2 =>
    the 2D observable breaks the 1D degeneracy."""
    pairs = [('tau_eff', 'T0'), ('tau_eff', 'power_ratio')]
    extr = _obs_extractors()
    fig, axes = plt.subplots(1, len(pairs), figsize=(6.5 * len(pairs), 5.5))
    for ax, (xn, yn) in zip(np.atleast_1d(axes), pairs):
        for scan in DEGEN_SCANS:
            rec = records[scan]
            fid_idx = rec['fid_idx']
            xv = _norm_to_fid(rec['obs'][xn], fid_idx)
            yv = _norm_to_fid(rec['obs'][yn], fid_idx)
            ax.plot(xv, yv, SCAN_MARKER[scan] + '-', color=SCAN_COLOR[scan],
                    lw=2, ms=7, label=f'{scan} ({SCANS[scan]["label"]})')
            # annotate variant suffixes along the curve
            for i, r in enumerate(rec['rows']):
                if np.isfinite(xv[i]) and np.isfinite(yv[i]):
                    ax.annotate(r['suffix'], (xv[i], yv[i]),
                                xytext=(4, 4), textcoords='offset points',
                                fontsize=7, color=SCAN_COLOR[scan])
        ax.axhline(1.0, color='gray', lw=0.8, ls=':')
        ax.axvline(1.0, color='gray', lw=0.8, ls=':')
        ax.set_xlabel(extr[xn][1] + ' / fid')
        ax.set_ylabel(extr[yn][1] + ' / fid')
        ax.grid(alpha=0.3); ax.legend()
    fig.suptitle(f'D6 -- observable-space map: separated tracks break the '
                 f'degeneracy ({snap_label})', fontsize=13)
    fig.tight_layout()
    _save(fig, out_path)


# =====================================================================
# D4 -- redshift evolution / growth rate (across snapshots)
# =====================================================================

def d4_redshift_evolution(records_by_snap, out_path, obs_names=('tau_eff', 'mean_flux')):
    """For each scan, plot observable vs z (one line per variant) in its own
    panel, and report the per-variant evolution slope d(obs)/dz. Growth-rate
    differences make the SLOPE depend on Omega_0 even at fixed sigma_8."""
    snaps = list(records_by_snap.keys())
    if len(snaps) < 2:
        print('  [D4] need >= 2 snaps for redshift evolution -- skipping')
        return {}

    nrow = len(obs_names)
    ncol = len(DEGEN_SCANS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.0 * ncol, 4.4 * nrow),
                             squeeze=False)
    slopes = {}
    for j, scan in enumerate(DEGEN_SCANS):
        snaps_z = sorted(
            snaps,
            key=lambda s: (records_by_snap[s][scan]['z']
                           if np.isfinite(records_by_snap[s][scan]['z']) else np.inf))
        zarr = np.array([records_by_snap[s][scan]['z'] for s in snaps_z], float)
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(VARIANT_SUFFIXES)))
        for i, name in enumerate(obs_names):
            ax = axes[i][j]
            for vi, suf in enumerate(VARIANT_SUFFIXES):
                yv = np.array([records_by_snap[s][scan]['obs'][name][vi]
                               for s in snaps_z], float)
                ev = np.array([records_by_snap[s][scan]
                               .get('obs_err', {}).get(name, [np.nan] * 5)[vi]
                               for s in snaps_z], float)
                pv = records_by_snap[snaps_z[0]][scan]['param'][vi]
                ax.errorbar(zarr, yv, yerr=ev if np.any(np.isfinite(ev)) else None,
                            fmt='o-', color=colors[vi], lw=1.6, ms=5, capsize=2,
                            label=f'{suf} ({pv:.2f})')
                m = np.isfinite(zarr) & np.isfinite(yv)
                if m.sum() >= 2:
                    slopes.setdefault(scan, {}).setdefault(name, {})[suf] = \
                        float(np.polyfit(zarr[m], yv[m], 1)[0])
            ax.invert_xaxis()
            ax.set_xlabel('redshift z')
            if name == 'tau_eff':
                ax.set_ylim(-0.5, 20)
                ax.set_ylabel('tau_eff')
                ax.set_title(f'{scan} ({SCANS[scan]["label"]}) -- $\\tau_{{\\rm eff}}(z)$')
            elif name == 'mean_flux':
                ax.set_ylabel('mean flux')
                ax.set_title(f'{scan} ({SCANS[scan]["label"]}) -- $\\langle F \\rangle(z)$')
            ax.grid(alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(title='variant', fontsize=7)
    fig.suptitle('D4 -- redshift evolution: slope differences are a growth-rate '
                 'handle', fontsize=13)
    fig.tight_layout()
    _save(fig, out_path)
    return slopes


# =====================================================================
# Entry
# =====================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--analysis-root', required=True, type=Path)
    ap.add_argument('--cosmo-csv', required=True, type=Path)
    ap.add_argument('--snaps', default='snap-080,snap-014',
                    help='comma-separated snap dirs; first is the primary '
                         'single-snap snapshot, all are used for D4')
    ap.add_argument('--out-dir', type=Path, default=Path('plots/degeneracy_test'))
    args = ap.parse_args()

    _setup_style()
    cosmo = load_cosmo_table(args.cosmo_csv)
    snaps = [s.strip() for s in args.snaps.split(',') if s.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Build records for every (scan, snap).
    records_by_snap = {}
    for snap in snaps:
        records_by_snap[snap] = {
            scan: scan_record(args.analysis_root, cosmo, scan, snap)
            for scan in DEGEN_SCANS}

    summary = {'snaps': snaps, 'per_snap': {}}

    # Single-snap diagnostics (D1, D2, D3, D5, D6) for each snap.
    for snap in snaps:
        print(f'\n=== degeneracy diagnostics, {snap} ===')
        recs = records_by_snap[snap]
        d = args.out_dir / snap
        d.mkdir(parents=True, exist_ok=True)
        split = d1_s8_collapse(recs, d / 'D1_S8_collapse.png', snap)
        d2_geometry          (recs, d / 'D2_geometry_pathlength.png', snap)
        pslopes = d3_power_shape(recs, d / 'D3_power_shape.png', snap)
        d5_thermal           (recs, d / 'D5_thermal.png', snap)
        d6_observable_space  (recs, d / 'D6_observable_space.png', snap)
        summary['per_snap'][snap] = {
            'z': {s: recs[s]['z'] for s in DEGEN_SCANS},
            'D1_split_score': split,        # bigger = more degeneracy-breaking
            'D3_power_ratio_slope': pslopes,
        }
        with open(d / 'summary.json', 'w') as fh:
            json.dump(summary['per_snap'][snap], fh, indent=2, default=float)

    # Across-snapshot growth-rate diagnostic (D4).
    print('\n=== redshift-evolution diagnostic (D4) ===')
    d4_slopes = d4_redshift_evolution(
        records_by_snap, args.out_dir / 'D4_redshift_evolution.png')
    summary['D4_evolution_slopes'] = d4_slopes
    with open(args.out_dir / 'summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2, default=float)

    print(f'\nAll outputs under {args.out_dir.resolve()}')


if __name__ == '__main__':
    sys.exit(main())
