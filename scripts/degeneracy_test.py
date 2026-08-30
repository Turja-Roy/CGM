"""
Omega_0 -- sigma_8 degeneracy test (CSV-only).

Goal: find observables that tell Omega_0 (the p1 scan) and sigma_8 (the p2 scan)
apart, given that both raise the matter-fluctuation amplitude and therefore move
most one-point statistics together along

    S_8 = sigma_8 * sqrt(Omega_m / 0.3).

READ THIS BEFORE TRUSTING ANY NUMBER OUT OF THIS SCRIPT
-------------------------------------------------------
1. The two scans do NOT span the same lever arm. p1 sweeps Omega_0 = 0.1..0.5 at
   sigma_8 = 0.8, so S_8 = 0.462..1.033 (range 0.571). p2 sweeps sigma_8 =
   0.6..1.0 at Omega_0 = 0.3, so S_8 = 0.600..1.000 (range 0.400). Comparing the
   raw spread of an observable across p1 against its spread across p2 credits
   Omega_0 with a 1.43x longer ruler and manufactures a discriminant out of
   nothing. Only the S_8-space split below is a fair comparison: it normalises
   each scan to its own fiducial and measures the gap between the two tracks at
   MATCHED S_8. Never quote a raw p1/p2 spread ratio as evidence.

2. Nothing here is compared to a noise floor. Changing only the IC seed at fixed
   cosmology (EX_0 vs 1P_p1_0, same parameters, seeds 13560 vs 67) moves tau_eff
   by 3.2% at z=4 rising to 13.5% at z=0, and moves the mean-flux evolution index
   by 1.2%. Most split scores this script reports are 0.02-0.26, i.e. the same
   order. Until the CAMELS CV set (27 seeds at fiducial parameters) is run
   through the same pipeline, treat every split as a candidate, not a detection.

3. k_eq ~ 0.015 h/Mpc is ~17x below the 25 Mpc/h box fundamental mode
   (2*pi/25 = 0.25 h/Mpc). No test here touches the LINEAR power-spectrum shape;
   the scale split measures nonlinear transfer only.

The tests
---------
  S_8 collapse test (the master diagnostic)
      Each observable, normalised to its own fiducial, plotted against S_8 for
      both scans. Overlapping tracks => the observable is a function of S_8 alone
      => degenerate. The reported split is the RMS gap between the tracks on a
      shared S_8 grid, and it is the only fair scalar in this script.

  Path-length geometry test
      Omega_0 enters dX/dz = (1+z)^2 / E(z); sigma_8 does not (Omega_m is held at
      0.3 across p2, so its dX/dz ratio is identically 1). Requires a scan that
      actually varies Omega_0 -- it is skipped otherwise, because on a
      fixed-Omega_0 set it can only draw a flat line at 1.0.

  Flux-power scale split
      k_eq ~ Omega_m h^2 tilts the shape of P_F(k) while sigma_8 lifts every mode
      equally, so the small/large band ratio should slope with Omega_0 and stay
      flat with sigma_8. Best-performing single-snapshot discriminant at z ~ 2
      (split 0.26), but feedback-dominated by z ~ 0.3 -- see
      feedback_robustness.py before using it at low z. The slope is only fitted
      for scans whose parameter is a real number; on a categorical set (EX) the
      x-axis is an arbitrary ordinal and the slope is meaningless.

  Evolution-index test
      f = dlnD/dlna ~ Omega_m^0.55, so the growth history differs between models
      matched at one epoch. DEMOTED: fitted as a power law d ln(obs) / d ln(1+z)
      -- the physically correct form for the forest -- every variant returns
      2.28..2.41 and the S_8-space split is 0.021 for the mean flux, SMALLER than
      the plain amplitude statistics this test was supposed to beat. The earlier
      "38% vs 7%" result came from a straight-line fit to a quantity spanning
      three decades, which is endpoint-dominated and changes with the snapshot
      list. Kept because it is an excellent null test: feedback moves it 0.9%.

  Observable-pair map
      One observable against another, parametrically. Non-collinear p1 and p2
      tracks mean the PAIR pins a measurement down even where either coordinate
      alone is degenerate. Add the EX track to see whether feedback moves the
      pair in a direction distinguishable from cosmology.

Dropped, and why:
  * Doppler-b response. line_width.cpp clamps b to [2, 80] km/s and 99.4% of the
    measured values at z=4 sit exactly at the 80 ceiling, so the median carried
    no information. Restore once the deblender is fixed.
  * Thermal-state panel figure. Its T_0 content duplicated the thermal-trend plot
    in hypothesis_test_p1.py, which has error bars and gamma; its only unique
    panel was the clamped b one.

Consumes only the per-variant CSVs that `analyze` already writes (cddf.csv,
flux_stats.csv, power_spectrum.csv, temp_density.csv). Reuses the loaders and
cosmology helpers from hypothesis_test_p1.py.

Run:
    python scripts/degeneracy_test.py \\
        --analysis-root output/analysis/IllustrisTNG/1P \\
        --ex-root       output/analysis/IllustrisTNG/EX \\
        --cosmo-csv data/IllustrisTNG/1P/CosmoAstroSeed_IllustrisTNG_L25n256_1P.csv \\
                    data/IllustrisTNG/EX/CosmoAstroSeed_IllustrisTNG_L25n256_EX.txt \\
        --scans p1,p2,ex \\
        --snaps snap-080,snap-044 \\
        --out-dir plots/degeneracy_test
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Shared loaders live in hypothesis_test_p1.py (same scripts/ dir, plain import).
from hypothesis_test_p1 import (
    SCANS, FIDUCIAL, members,
    build_scan_frame, load_cosmo_table,
    dXdz,
    _setup_style, _save,
)

# Two scans under test. p1 varies Omega_0 (sigma_8 fixed 0.8);
# p2 varies sigma_8 (Omega_0 fixed 0.3). Fiducial of both: Omega_0=0.3, sigma_8=0.8.
DEGEN_SCANS = ['p1', 'p2']
SCAN_COLOR  = {'p1': 'C0', 'p2': 'C3', 'ex': 'C2',   # Omega_0 blue, sigma_8 red, feedback green
               'p7': 'C4', 'p8': 'C5', 'p9': 'C6'}
SCAN_MARKER = {'p1': 'o',  'p2': 's',  'ex': '^',
               'p7': 'v',  'p8': 'D',  'p9': 'P'}
S8_FID = 0.8 * np.sqrt(0.3 / 0.3)           # fiducial S_8 = 0.8

# The S_8 collapse split is a gap BETWEEN these two scans; with only one of them
# selected there is nothing to compare and the score is simply not defined.
COSMO_PAIR = {'p1', 'p2'}


def varies_omega0(rec):
    """True if this scan actually moves Omega_0. The path-length test divides
    dX/dz by its fiducial, so on a fixed-Omega_0 set it can only ever draw 1.0."""
    om = np.asarray(rec['Omega0'], float)
    om = om[np.isfinite(om)]
    return om.size > 1 and not np.allclose(om, om[0])


def has_numeric_param(scan):
    """False for a categorical set such as EX, whose 'parameter' is a 1-based
    ordinal over sim names. Fitting a slope against that ordering is meaningless
    -- it measures the order the sims happen to be listed in."""
    return SCANS[scan].get('column') is not None


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


CDDF_LOWN = 13.5
CDDF_HIGHN = 15.0


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


def cddf_slope(cddf, logN_lo=CDDF_LOWN, logN_hi=CDDF_HIGHN):
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


def cddf_slope_err(cddf, logN_lo=CDDF_LOWN, logN_hi=CDDF_HIGHN):
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


# observable name -> (extractor(row), pretty label, prefer-log-y)
def _obs_extractors():
    return {
        'tau_eff':    (lambda r: r['tau_eff'],                         r'$\tau_{\rm eff}$',                False),
        'mean_flux':  (lambda r: r['mean_flux'],                       r'$\langle F\rangle$',              False),
        'T0':         (lambda r: r['T0'],                              r'$T_0$ [K]',                       False),
        'power_ratio':(lambda r: scale_split_ratio(r['power_spectrum'])[2], r'$P_F$ small/large ratio',    False),
        'cddf_lowN':  (lambda r: cddf_value(r['cddf'], CDDF_LOWN),     rf'$f(N_{{\rm HI}}{{=}}10^{{{CDDF_LOWN}}})$',  True),
        'cddf_highN': (lambda r: cddf_value(r['cddf'], CDDF_HIGHN),    rf'$f(N_{{\rm HI}}{{=}}10^{{{CDDF_HIGHN}}})$', True),
        'cddf_slope': (lambda r: cddf_slope(r['cddf']),                rf'CDDF log-log slope ({CDDF_LOWN}$\to${CDDF_HIGHN})', False),
    }


# 1-sigma errors, all internal to one box (sightline scatter, Poisson counting).
# power_ratio has none: propagating P_k_err through two band integrals needs the
# k-mode covariance, which is not stored.
def _obs_error_extractors():
    return {
        'tau_eff':    lambda r: r.get('tau_eff_err', np.nan),
        'mean_flux':  lambda r: r.get('mean_flux_err', np.nan),
        'T0':         lambda r: r.get('T0_err', np.nan),
        'power_ratio': lambda r: np.nan,
        'cddf_lowN':  lambda r: cddf_value_err(r['cddf'], CDDF_LOWN),
        'cddf_highN': lambda r: cddf_value_err(r['cddf'], CDDF_HIGHN),
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
# S_8 collapse test -- S_8 collapse test (master diagnostic)
# =====================================================================

def _matched_s8_gap(curves):
    """RMS gap between the p2 and p1 tracks interpolated onto a shared S_8 grid.

    Comparing at MATCHED S_8 is what makes this fair -- the two scans span
    different S_8 ranges (0.571 vs 0.400), so a raw spread ratio would credit
    Omega_0 with the longer ruler. `curves` maps scan -> (S_8, obs/fid).
    Returns NaN when either track is missing or too short to interpolate.
    """
    if not COSMO_PAIR.issubset(curves):
        return np.nan
    x1, y1 = curves['p1']; x2, y2 = curves['p2']
    lo = max(np.nanmin(x1), np.nanmin(x2))
    hi = min(np.nanmax(x1), np.nanmax(x2))
    m1 = np.isfinite(x1) & np.isfinite(y1)
    m2 = np.isfinite(x2) & np.isfinite(y2)
    if m1.sum() < 2 or m2.sum() < 2 or not hi > lo:
        return np.nan
    xs = np.linspace(lo, hi, 25)
    g1 = np.interp(xs, x1[m1], y1[m1])
    g2 = np.interp(xs, x2[m2], y2[m2])
    return float(np.sqrt(np.mean((g2 - g1) ** 2)))


def s8_collapse_test(records, out_path, snap_label):
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
        gap = _matched_s8_gap(curves)
        if np.isfinite(gap):
            split_score[name] = gap

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
    fig.suptitle(f'S_8 collapse test -- $S_8$ collapse test: degenerate observables overlap '
                 f'({snap_label})', fontsize=13)
    fig.tight_layout()
    _save(fig, out_path)
    return split_score


# =====================================================================
# Path-length geometry test -- geometric path length dX/dz
# =====================================================================

def path_length_geometry_test(records, out_path, snap_label):
    """Left: dX/dz ratio vs parameter (p2 is flat by construction).
    Right: CDDF before/after the dX correction for p1 vs p2 -- only p1 moves.

    Skipped unless some selected scan actually varies Omega_0. On a set that
    holds it fixed (p2 alone, or EX) every point is dX/dz over its own fiducial
    = 1.0 exactly, and the figure asserts "only Omega_0 moves it" over a flat
    line -- a guaranteed null dressed as a result.
    """
    if not any(varies_omega0(records[s]) for s in DEGEN_SCANS):
        print('  [path-length geometry] no selected scan varies Omega_0 -- skipping')
        return
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))

    for scan in DEGEN_SCANS:
        rec = records[scan]
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

    fig.suptitle(f'Path-length geometry test -- geometric discriminant ({snap_label})', fontsize=13)
    fig.tight_layout()
    _save(fig, out_path)


# =====================================================================
# Flux-power scale split -- power-spectrum shape (scale split)
# =====================================================================

def power_scale_split_test(records, out_path, snap_label):
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
        # Only fit a slope where the x-axis is a physical parameter. On EX the
        # x-axis is the order the sims are listed in, and its slope means nothing.
        if m.sum() >= 2 and has_numeric_param(scan):
            slopes[scan] = float(np.polyfit(x[m], y[m], 1)[0])
    ax.axhline(1.0, color='gray', lw=0.8, ls=':')
    ax.axvline(1.0, color='gray', lw=0.8, ls=':')
    ax.set_xlabel('parameter / fiducial'
                  if all(has_numeric_param(s) for s in DEGEN_SCANS)
                  else 'variant (ordinal for categorical sets)')
    ax.set_ylabel(r'(small/large $P_F$ ratio) / fid')
    sub = ', '.join(f'{s} slope={slopes[s]:.2f}' for s in DEGEN_SCANS if s in slopes) \
          or 'no numeric parameter axis -- slope not defined'
    ax.set_title(f'Flux-power scale split -- $P_F(k)$ shape tilt ({snap_label})\n{sub}', fontsize=11)
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    _save(fig, out_path)
    return slopes


# =====================================================================
# Observable-pair map -- observable-space map (joint figure)
# =====================================================================

def observable_pair_map(records, out_path, snap_label):
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
    fig.suptitle(f'Observable-pair map -- observable-space map: separated tracks break the '
                 f'degeneracy ({snap_label})', fontsize=13)
    fig.tight_layout()
    _save(fig, out_path)


# =====================================================================
# CDDF amplitudes -- the two epochs at which each one is usable
# =====================================================================

def cddf_amplitude_figure(records_by_snap, out_path,
                          panels=(('cddf_lowN', 3.0), ('cddf_highN', 0.0))):
    """The two CDDF amplitudes, each at the epoch where it separates the scans.

    The full s8_collapse figure carries all seven observables at one snapshot;
    this is the transpose -- one observable per panel, each at its own redshift,
    so the two results that survive the seed floor can be shown side by side.

    Panels are (observable, target z); the snapshot used is whichever one in
    records_by_snap sits nearest that z, so the figure does not break when the
    snapshot list changes. Where 'ex' is among the scans its fiducial member is
    the seed pair of the p1 fiducial, so its offset from 1.0 IS the seed floor.
    """
    extr = _obs_extractors()
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 4.4),
                             squeeze=False)
    gaps = {}
    for ax, (name, z_target) in zip(axes.ravel(), panels):
        _, lbl, logy = extr[name]
        snap = min(records_by_snap,
                   key=lambda s: abs(records_by_snap[s][DEGEN_SCANS[0]]['z'] - z_target))
        records = records_by_snap[snap]
        curves = {}
        for scan in DEGEN_SCANS:
            rec = records[scan]
            y = _norm_to_fid(rec['obs'][name], rec['fid_idx'])
            yerr = _norm_err_to_fid(rec['obs'][name],
                                    rec.get('obs_err', {}).get(name),
                                    rec['fid_idx'])
            ax.errorbar(rec['S8'], y, yerr=yerr, fmt=SCAN_MARKER[scan] + '-',
                        color=SCAN_COLOR[scan], lw=1.8, ms=6, capsize=3,
                        label=f'{scan} ({SCANS[scan]["label"]})')
            curves[scan] = (rec['S8'], y)
        gap = _matched_s8_gap(curves)
        gaps[name] = {'snap': snap, 'z': records[DEGEN_SCANS[0]]['z'], 'gap': gap}

        ax.axvline(S8_FID, color='gray', lw=0.8, ls=':')
        ax.axhline(1.0,    color='gray', lw=0.8, ls=':')
        if logy:
            ax.set_yscale('log')
        ax.set_xlabel(r'$S_8 = \sigma_8\sqrt{\Omega_m/0.3}$')
        ax.set_ylabel(lbl + ' / fid')
        z = records[DEGEN_SCANS[0]]['z']
        tag = f'  (split={gap:.3f})' if np.isfinite(gap) else ''
        ax.set_title(f'{name} at $z = {z:.2f}$' + tag, fontsize=11)
        ax.grid(alpha=0.3, which='both')
    axes.ravel()[0].legend(fontsize=9, loc='best')
    fig.suptitle('CDDF amplitudes -- each at the epoch where it separates the '
                 'two scans by more than the seed floor', fontsize=12)
    fig.tight_layout()
    _save(fig, out_path)
    return gaps


# =====================================================================
# Evolution-index test -- redshift evolution / growth rate (across snapshots)
# =====================================================================

def evolution_index_test(records_by_snap, out_path, obs_names=('tau_eff', 'mean_flux')):
    """For each scan, plot observable vs z (one line per variant) in its own
    panel, and report the per-variant evolution index d ln(obs) / d ln(1+z).

    The index, not a straight-line d(obs)/dz. tau_eff runs 0.01 to 8 and the mean
    flux 0.95 to 0.0002 across this z range, so a linear fit is dominated by
    whichever endpoint is in the snapshot list and the number moves when the list
    does. The power law is the standard forest parameterisation and is scale-free.

    Expect the answer to be boring: every variant returns an index of 2.28..2.41,
    because raising Omega_0 or sigma_8 scales tau_eff at all z rather than
    changing how it evolves. The information is in the intercept -- amplitude --
    which is exactly what the S_8 collapse test says is degenerate.
    """
    snaps = list(records_by_snap.keys())
    if len(snaps) < 2:
        print('  [evolution-index] need >= 2 snaps for redshift evolution -- skipping')
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
        mem = members(scan)
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(mem)))
        for i, name in enumerate(obs_names):
            ax = axes[i][j]
            for vi, suf in enumerate(mem):
                yv = np.array([records_by_snap[s][scan]['obs'][name][vi]
                               for s in snaps_z], float)
                ev = np.array([records_by_snap[s][scan]
                               .get('obs_err', {})
                               .get(name, [np.nan] * len(mem))[vi]
                               for s in snaps_z], float)
                pv = records_by_snap[snaps_z[0]][scan]['param'][vi]
                ax.errorbar(zarr, yv, yerr=ev if np.any(np.isfinite(ev)) else None,
                            fmt='o-', color=colors[vi], lw=1.6, ms=5, capsize=2,
                            label=f'{suf} ({pv:.2f})')
                # Power-law index, so the number does not depend on which
                # snapshots happen to be in the list. Needs y > 0.
                m = np.isfinite(zarr) & np.isfinite(yv) & (yv > 0) & (zarr > -1)
                if m.sum() >= 2:
                    slopes.setdefault(scan, {}).setdefault(name, {})[suf] = float(
                        np.polyfit(np.log(1.0 + zarr[m]), np.log(yv[m]), 1)[0])
            ax.invert_xaxis()
            ax.set_xlabel('redshift z')
            if name == 'tau_eff':
                ax.set_ylim(-0.5, 3)
                ax.set_ylabel('tau_eff')
                ax.set_title(f'{scan} ({SCANS[scan]["label"]}) -- $\\tau_{{\\rm eff}}(z)$')
            elif name == 'mean_flux':
                ax.set_ylabel('mean flux')
                ax.set_title(f'{scan} ({SCANS[scan]["label"]}) -- $\\langle F \\rangle(z)$')
            ax.grid(alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(title='variant', fontsize=7)
    fig.suptitle('Evolution-index test -- redshift evolution. Reported index is '
                 r'$d\ln(\rm obs)/d\ln(1+z)$; it barely moves across either scan, '
                 'so this is a null test, not a discriminant.', fontsize=12)
    fig.tight_layout()
    _save(fig, out_path)
    return slopes


# =====================================================================
# Entry
# =====================================================================

def main():
    global DEGEN_SCANS
    ap = argparse.ArgumentParser()
    ap.add_argument('--analysis-root', required=True, type=Path,
                    help='root holding the 1P scan dirs (1P_p1_0/, ...)')
    ap.add_argument('--ex-root', type=Path, default=None,
                    help='root holding the EX dirs (EX_0/, ...). Required to '
                         'include ex in --scans: EX lives in a sibling tree, so '
                         'one root cannot reach both and every row silently '
                         'comes back empty if you try.')
    ap.add_argument('--cosmo-csv', required=True, type=Path, nargs='+',
                    help='one or more CosmoAstroSeed tables; pass the 1P and EX '
                         'ones together to run --scans p1,p2,ex')
    ap.add_argument('--scans', default=','.join(DEGEN_SCANS),
                    help='comma-separated scans to overlay (any of '
                         + ','.join(SCANS) + ')')
    ap.add_argument('--snaps', default='snap-080,snap-044',
                    help='comma-separated snap dirs; the first is the primary '
                         'single-snapshot one, all are used for the evolution index')
    ap.add_argument('--out-dir', type=Path, default=Path('plots/degeneracy_test'))
    args = ap.parse_args()

    scans = [s.strip() for s in args.scans.split(',') if s.strip()]
    unknown = [s for s in scans if s not in SCANS]
    if unknown:
        ap.error(f'unknown scan(s) {unknown}; known: {list(SCANS)}')
    DEGEN_SCANS = scans

    # EX_0/ and 1P_p1_0/ are siblings, not children of one root. Resolve each
    # scan to its own tree rather than letting build_scan_frame come back with
    # zero rows and turn the whole run into NaN.
    if 'ex' in scans and args.ex_root is None:
        ap.error("--scans includes 'ex' but --ex-root was not given; EX lives in "
                 'a separate tree from the 1P scans')
    root_for = {s: (args.ex_root if s == 'ex' else args.analysis_root) for s in scans}

    _setup_style()
    cosmo = pd.concat([load_cosmo_table(c) for c in args.cosmo_csv])
    snaps = [s.strip() for s in args.snaps.split(',') if s.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Build records for every (scan, snap).
    records_by_snap = {}
    for snap in snaps:
        records_by_snap[snap] = {
            scan: scan_record(root_for[scan], cosmo, scan, snap)
            for scan in DEGEN_SCANS}

    # A scan that resolved to nothing produces an all-NaN figure that looks like
    # a result. Say so instead.
    for snap, recs in records_by_snap.items():
        for scan, rec in recs.items():
            if rec['fid_idx'] is None or not np.isfinite(rec['z']):
                print(f'  WARNING: {scan} has no usable fiducial row at {snap} '
                      f'under {root_for[scan]} -- its panels will be empty')

    summary = {'snaps': snaps, 'per_snap': {}}

    # Single-snapshot diagnostics for each snap.
    for snap in snaps:
        print(f'\n=== degeneracy diagnostics, {snap} ===')
        recs = records_by_snap[snap]
        d = args.out_dir / snap
        d.mkdir(parents=True, exist_ok=True)
        # Figures are captioned by redshift, not by the directory they came
        # from -- these end up in write-ups where a snapshot number means nothing.
        zs = [recs[s]['z'] for s in DEGEN_SCANS if np.isfinite(recs[s]['z'])]
        label = f'$z = {zs[0]:.2f}$' if zs else snap
        split = s8_collapse_test(recs, d / 's8_collapse.png', label)
        path_length_geometry_test(recs, d / 'path_length_geometry.png', label)
        pslopes = power_scale_split_test(recs, d / 'power_scale_split.png', label)
        observable_pair_map(recs, d / 'observable_pair_map.png', label)
        summary['per_snap'][snap] = {
            'z': {s: recs[s]['z'] for s in DEGEN_SCANS},
            's8_collapse_split': split,        # bigger = more degeneracy-breaking
            'power_scale_split_slope': pslopes,
        }
        with open(d / 'summary.json', 'w') as fh:
            json.dump(summary['per_snap'][snap], fh, indent=2, default=float)

    # The two CDDF amplitudes, each at its own epoch.
    print('\n=== CDDF amplitude figure ===')
    summary['cddf_amplitudes'] = cddf_amplitude_figure(
        records_by_snap, args.out_dir / 'cddf_amplitudes.png')

    # Across-snapshot growth-rate diagnostic.
    print('\n=== evolution-index diagnostic ===')
    evo_index = evolution_index_test(
        records_by_snap, args.out_dir / 'evolution_index.png')
    summary['evolution_index'] = evo_index
    with open(args.out_dir / 'summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2, default=float)

    print(f'\nAll outputs under {args.out_dir.resolve()}')


if __name__ == '__main__':
    sys.exit(main())
