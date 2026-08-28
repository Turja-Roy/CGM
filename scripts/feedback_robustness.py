"""
Feedback robustness of the Lyman-alpha observables (Tier 1, CSV-only).

Question: does extreme feedback move an observable as much as cosmology does?
An observable that responds strongly to Omega_0 / sigma_8 but barely to feedback
is a usable cosmology discriminant. One that moves as much under feedback is
contaminated, and no amount of forest statistics will pull the cosmology back out.

The CAMELS IllustrisTNG EX set is built for exactly this: four sims sharing one
IC seed (13560) and one cosmology (Omega_m=0.3, sigma_8=0.8), differing only in
feedback --

    EX_0  fiducial          EX_2  A_SN1  = 100  (extreme SN)
    EX_1  A_AGN1 = 100      EX_3  all A  = 0    (no feedback)

Metric, per observable and per snapshot:

    S_fb = (max - min over EX_0..EX_3)     / |EX_0|
    S_p1 = (max - min over 1P_p1_n2..p1_2) / |1P_p1_0|      (Omega_0 scan)
    S_p2 = (max - min over 1P_p2_n2..p2_2) / |1P_p2_0|      (sigma_8 scan)

    R_p1 = S_fb / S_p1        R_p2 = S_fb / S_p2

R < 1  -> feedback moves it less than the sampled cosmology range: robust.
R > 1  -> feedback-dominated.

THREE THINGS R IS NOT
---------------------
1. R is RANGE-RELATIVE, not a likelihood. The 1P scan spans a chosen +-range and
   EX spans a deliberately unphysical extreme (100x, and zero). R compares two
   arbitrary lever arms. Use it to rank observables against each other, never as
   a marginalized error.

2. R is only meaningful because both spreads are computed WITHIN one set. EX
   shares seed 13560; the 1P scans share seed 67. The seed cancels in each
   spread. It does NOT cancel between sets: EX_0 and 1P_p1_0 have identical
   cosmology and identical fiducial astrophysics yet their tau_eff differs by
   3.2% at z=4 rising to 13.5% at z=0, purely from IC variance in a 25 Mpc/h box.
   That offset is comparable to the largest feedback signal here, so this script
   never forms an absolute EX/1P quantity -- only ratios of within-set spreads.

3. A spread below the internal 1-sigma error is not a detection. Those points are
   flagged `upper_limit` in the JSON and drawn as open markers.

TNG's kinetic-mode AGN channel only switches on above ~1e8 Msun, and no black
hole in this box crosses that until z~3. EX_1 is therefore bit-identical to EX_0
at snaps 024 and 028 -- real physics, not a pipeline fault. The script counts
distinct members per snapshot and annotates where it is 3 instead of 4, so a
spread carried by two sims is never mistaken for one carried by four.

Consumes only the per-snapshot CSVs that `analyze_spectra.py analyze` writes.
Does not touch spectra or raw snapshots. Run:

    python scripts/feedback_robustness.py \\
        --ex-root output/analysis/IllustrisTNG/EX \\
        --p1-root output/analysis/IllustrisTNG/1P \\
        --cosmo-csv data/IllustrisTNG/1P/CosmoAstroSeed_IllustrisTNG_L25n256_1P.csv \\
        --out-dir plots/ex_robustness

    python scripts/feedback_robustness.py --self-test
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Same scripts/ dir, so a plain import works -- matches degeneracy_test.py.
from hypothesis_test_p1 import load_snap_row, load_cosmo_table, _setup_style, _save
from hypothesis_test_p1 import build_scan_frame, FIDUCIAL, VARIANT_SUFFIXES
from degeneracy_test import (_obs_extractors, _obs_error_extractors,
                             cddf_value, cddf_slope, cddf_value_err, cddf_slope_err)

# The CDDF low-N anchor moves from 13.0 to 13.5.
#
# degeneracy_test anchors cddf_lowN and cddf_slope at log N = 13.0, but the
# production binning starts at log_N_min = 13.0 with 0.2 dex bins, so the first
# BIN CENTRE is 13.111. cddf_value() refuses to extrapolate below x.min(), so
# both observables come back NaN for every simulation -- silently, in the 1P
# figures too. 13.5 is inside the grid and is already the CDDF fit floor
# (see [[cddf-production-defaults]]: fit range 13.5-15), so it keeps the
# low-N intent without extrapolating.
#
# Fixed here rather than in degeneracy_test because changing that module's
# reference column would move already-published 1P figures. Worth fixing there
# separately.
CDDF_LO, CDDF_HI = 13.5, 15.0


def _patched_obs_extractors():
    e = dict(_obs_extractors())
    _, lo_lbl, lo_log = e['cddf_lowN']
    e['cddf_lowN'] = (lambda r: cddf_value(r['cddf'], CDDF_LO),
                      rf'$f(N_{{\rm HI}}{{=}}10^{{{CDDF_LO}}})$', lo_log)
    e['cddf_slope'] = (lambda r: cddf_slope(r['cddf'], CDDF_LO, CDDF_HI),
                       rf'CDDF log-log slope ({CDDF_LO}$\to${CDDF_HI})', False)
    return e


def _patched_obs_error_extractors():
    e = dict(_obs_error_extractors())
    e['cddf_lowN'] = lambda r: cddf_value_err(r['cddf'], CDDF_LO)
    e['cddf_slope'] = lambda r: cddf_slope_err(r['cddf'], CDDF_LO, CDDF_HI)
    return e


EX_SIMS = ['EX_0', 'EX_1', 'EX_2', 'EX_3']
EX_FID = 0
EX_LABEL = {
    'EX_0': 'fiducial',
    'EX_1': r'extreme AGN ($A_{\rm AGN1}{=}100$)',
    'EX_2': r'extreme SN ($A_{\rm SN1}{=}100$)',
    'EX_3': 'no feedback',
}
EX_COLOR = {'EX_0': 'k', 'EX_1': 'C3', 'EX_2': 'C0', 'EX_3': 'C2'}

DEFAULT_SNAPS = ('snap-024 snap-028 snap-032 snap-038 snap-044 '
                 'snap-050 snap-060 snap-072 snap-080 snap-090').split()

COSMO_SCANS = ['p1', 'p2']
SCAN_LABEL = {'p1': r'$\Omega_0$ scan', 'p2': r'$\sigma_8$ scan'}


# =====================================================================
# The metric
# =====================================================================

def frac_spread(vals, fid_idx=0):
    """(max - min) / |fiducial|, over the members of one set.

    NaN if any member is missing: a hole would otherwise shrink the range and
    make the set look more robust than it is. NaN, not inf, on a zero fiducial.
    """
    v = np.asarray(vals, dtype=float)
    if v.size == 0 or not np.all(np.isfinite(v)):
        return np.nan
    fid = v[fid_idx]
    if not np.isfinite(fid) or fid == 0:
        return np.nan
    return float((v.max() - v.min()) / abs(fid))


def ratio(num, den):
    """num/den, NaN-safe and never inf."""
    if not (np.isfinite(num) and np.isfinite(den)) or den == 0:
        return np.nan
    return float(num / den)


def n_distinct(vals):
    """How many members actually differ. Catches EX_1 == EX_0 at high z."""
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    return int(np.unique(v).size)


def is_upper_limit(spread, fid_val, fid_err):
    """True when the spread sits under the fiducial's own 1-sigma error.

    ponytail: compares the range against a single member's error rather than
    propagating a max-minus-min error, which would need the member covariance.
    Deliberately conservative -- it flags marginal points, it does not price them.
    """
    if not np.isfinite(spread):
        return False
    if not (np.isfinite(fid_val) and np.isfinite(fid_err)) or fid_val == 0:
        return False
    return bool(spread < abs(fid_err / fid_val))


# =====================================================================
# Load the EX set into the same shape scan_record() returns for 1P
# =====================================================================

def ex_record(ex_root, snap):
    rows = [load_snap_row(Path(ex_root) / sim / snap) for sim in EX_SIMS]
    extr, errs = _patched_obs_extractors(), _patched_obs_error_extractors()
    return {
        'snap': snap,
        'sims': list(EX_SIMS),
        'rows': rows,
        'z': rows[EX_FID]['redshift'],
        'obs': {n: np.array([fn(r) for r in rows], float)
                for n, (fn, _l, _lg) in extr.items()},
        'obs_err': {n: np.array([fn(r) for r in rows], float)
                    for n, fn in errs.items()},
    }


def cosmo_record(p1_root, cosmo, scan, snap):
    """One 1P scan at one snapshot, in the same shape as ex_record().

    Deliberately NOT degeneracy_test.scan_record: that calls the unpatched
    _obs_extractors internally, so its cddf_lowN and cddf_slope would come back
    NaN at the 13.0 anchor while the EX numerator used 13.5. Numerator and
    denominator have to be measured the same way or R is meaningless.
    """
    rows = build_scan_frame(Path(p1_root), cosmo, scan, snap)
    extr, errs = _patched_obs_extractors(), _patched_obs_error_extractors()
    fid_idx = next((i for i, r in enumerate(rows) if r['suffix'] == FIDUCIAL), None)
    return {
        'scan': scan,
        'snap': snap,
        'rows': rows,
        'fid_idx': fid_idx,
        'z': rows[fid_idx]['redshift'] if fid_idx is not None else np.nan,
        'obs': {n: np.array([fn(r) for r in rows], float)
                for n, (fn, _l, _lg) in extr.items()},
        'obs_err': {n: np.array([fn(r) for r in rows], float)
                    for n, fn in errs.items()},
    }


def build(ex_root, p1_root, cosmo_csv, snaps):
    cosmo = load_cosmo_table(cosmo_csv)
    ex = {s: ex_record(ex_root, s) for s in snaps}
    cos = {sc: {s: cosmo_record(p1_root, cosmo, sc, s) for s in snaps}
           for sc in COSMO_SCANS}
    return ex, cos


def gate(ex, cos, snaps):
    """Refuse to plot a half-synced grid.

    A missing variant silently changes which two members set max - min, so the
    lever arm stops being the same at every redshift. Same gate philosophy as
    shell_scripts/make_comparison_plots.sh.
    """
    missing = []
    for s in snaps:
        for i, sim in enumerate(EX_SIMS):
            if not np.isfinite(ex[s]['obs']['tau_eff'][i]):
                missing.append(f'EX/{sim}/{s}')
        for sc in COSMO_SCANS:
            for r in cos[sc][s]['rows']:
                if not np.isfinite(r['tau_eff']):
                    missing.append(f"1P/{r['label']}/{s}")
    return missing


# =====================================================================
# Reduce to the table everything else reads
# =====================================================================

def robustness_table(ex, cos, snaps):
    extr = _patched_obs_extractors()
    out = {'snaps': list(snaps),
           'z': [float(ex[s]['z']) for s in snaps],
           'observables': {}}
    for name, (_fn, label, _log) in extr.items():
        rec = {'label': label, 'S_fb': [], 'S_p1': [], 'S_p2': [],
               'R_p1': [], 'R_p2': [], 'n_distinct_ex': [], 'upper_limit': []}
        for s in snaps:
            v = ex[s]['obs'][name]
            e = ex[s]['obs_err'][name]
            s_fb = frac_spread(v, EX_FID)
            rec['S_fb'].append(s_fb)
            rec['n_distinct_ex'].append(n_distinct(v))
            rec['upper_limit'].append(is_upper_limit(s_fb, v[EX_FID], e[EX_FID]))
            for sc in COSMO_SCANS:
                c = cos[sc][s]
                s_c = frac_spread(c['obs'][name], c['fid_idx'])
                rec[f'S_{sc}'].append(s_c)
                rec[f'R_{sc}'].append(ratio(s_fb, s_c))
        out['observables'][name] = rec
    return out


# =====================================================================
# Figures
# =====================================================================

def _obs_style():
    names = list(_patched_obs_extractors())
    return {n: (f'C{i}', 'os^vD<>p'[i % 8]) for i, n in enumerate(names)}


def plot_robustness_ratio(tab, out_path):
    z = np.array(tab['z'])
    style = _obs_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, sc in zip(axes, COSMO_SCANS):
        for name, rec in tab['observables'].items():
            r = np.array(rec[f'R_{sc}'], float)
            col, mk = style[name]
            ax.plot(z, r, '-', color=col, lw=1.2, alpha=0.8, zorder=2)
            ul = np.array(rec['upper_limit'], bool)
            ax.plot(z[~ul], r[~ul], mk, color=col, ms=6, label=rec['label'], zorder=3)
            if ul.any():
                ax.plot(z[ul], r[ul], mk, mfc='none', color=col, ms=6, zorder=3)
        ax.axhline(1.0, color='k', ls='--', lw=1.2, zorder=1)
        ax.set_yscale('log')
        ax.set_xlabel('redshift')
        ax.set_title(f'feedback spread / {SCAN_LABEL[sc]} spread')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r'$R = S_{\rm fb}\,/\,S_{\rm cosmo}$')
    axes[1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle('Below the dashed line: feedback moves it less than cosmology does.\n'
                 'Open markers: spread under the internal 1$\\sigma$, treat as upper limits. '
                 'R is range-relative, not a likelihood.', fontsize=9, y=1.06)
    _save(fig, out_path)


def plot_spreads(tab, out_path):
    """The numerator and denominators behind R, so a surprising R is traceable."""
    z = np.array(tab['z'])
    names = list(tab['observables'])
    ncol = 4
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow),
                             squeeze=False, sharex=True)
    for ax, name in zip(axes.ravel(), names):
        rec = tab['observables'][name]
        ax.plot(z, rec['S_fb'], 'ko-', ms=4, label=r'$S_{\rm fb}$ (EX)')
        ax.plot(z, rec['S_p1'], 'o-', color='C0', ms=4, label=r'$S_{p1}$ ($\Omega_0$)')
        ax.plot(z, rec['S_p2'], 's-', color='C3', ms=4, label=r'$S_{p2}$ ($\sigma_8$)')
        ax.set_yscale('log')
        ax.set_title(rec['label'], fontsize=10)
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(names):]:
        ax.axis('off')
    for ax in axes[-1]:
        ax.set_xlabel('redshift')
    for ax in axes[:, 0]:
        ax.set_ylabel('fractional spread')
    axes[0, 0].legend(fontsize=8, frameon=False)
    _save(fig, out_path)


def plot_ex_observables(ex, tab, snaps, out_path):
    """Each observable vs z, four sims overlaid, normalized to EX_0."""
    names = list(tab['observables'])
    z = np.array(tab['z'])
    ncol = 4
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow),
                             squeeze=False, sharex=True)
    for ax, name in zip(axes.ravel(), names):
        rec = tab['observables'][name]
        vals = np.array([ex[s]['obs'][name] for s in snaps], float)   # (nz, 4)
        with np.errstate(divide='ignore', invalid='ignore'):
            norm = vals / vals[:, EX_FID][:, None]
        for i, sim in enumerate(EX_SIMS):
            ax.plot(z, norm[:, i], 'o-', ms=4, color=EX_COLOR[sim],
                    label=EX_LABEL[sim] if name == names[0] else None)
        ax.axhline(1.0, color='k', ls=':', lw=0.8)
        ax.set_title(rec['label'], fontsize=10)
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(names):]:
        ax.axis('off')
    for ax in axes[-1]:
        ax.set_xlabel('redshift')
    for ax in axes[:, 0]:
        ax.set_ylabel('value / EX_0')
    axes[0, 0].legend(fontsize=8, frameon=False)
    # One statement for the whole figure: it is the same snapshots in every panel.
    deg = sorted({s for rec in tab['observables'].values()
                  for s, n in zip(snaps, rec['n_distinct_ex']) if n < len(EX_SIMS)})
    if deg:
        fig.suptitle('EX_1 is bit-identical to EX_0 at ' + ', '.join(deg) +
                     ' (TNG kinetic AGN inactive below ~$10^8\\,M_\\odot$), '
                     'so the spread there is carried by EX_2 and EX_3 alone.',
                     fontsize=9, y=1.02)
    _save(fig, out_path)


def plot_cddf_grid(ex, snaps, out_path):
    ncol = 5
    nrow = int(np.ceil(len(snaps) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.2 * nrow),
                             squeeze=False, sharex=True, sharey=True)
    for ax, snap in zip(axes.ravel(), snaps):
        rec = ex[snap]
        for sim, row in zip(EX_SIMS, rec['rows']):
            c = row['cddf']
            if c is None:
                continue
            m = c['f_N_HI'] > 0
            ax.plot(c['log10_N_HI'][m], c['f_N_HI'][m], '-',
                    color=EX_COLOR[sim], lw=1.3,
                    label=EX_LABEL[sim] if snap == snaps[0] else None)
        ax.set_yscale('log')
        ax.set_title(f"{snap}  (z = {rec['z']:.2f})", fontsize=10)
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(snaps):]:
        ax.axis('off')
    for ax in axes[-1]:
        ax.set_xlabel(r'$\log_{10} N_{\rm HI}$')
    for ax in axes[:, 0]:
        ax.set_ylabel(r'$f(N_{\rm HI})$')
    axes[0, 0].legend(fontsize=8, frameon=False)
    _save(fig, out_path)


# =====================================================================
# Self-test
# =====================================================================

def self_test():
    # equal spreads -> R == 1
    assert ratio(frac_spread([1.0, 1.0, 1.2, 0.8]),
                 frac_spread([2.0, 2.0, 2.4, 1.6])) == 1.0

    # normalized by the FIDUCIAL member, not the mean or the max
    assert abs(frac_spread([2.0, 3.0, 1.0], fid_idx=0) - 1.0) < 1e-12
    assert abs(frac_spread([2.0, 3.0, 1.0], fid_idx=1) - 2.0 / 3.0) < 1e-12

    # a missing member yields NaN rather than a quietly narrower range
    assert np.isnan(frac_spread([1.0, np.nan, 1.5, 0.5]))

    # zero fiducial -> NaN, never inf
    assert np.isnan(frac_spread([0.0, 1.0, 2.0]))
    assert np.isnan(ratio(1.0, 0.0))
    assert np.isnan(ratio(np.nan, 1.0))

    # EX_1 == EX_0 must show up as 3 distinct members, not 4
    assert n_distinct([1.0, 1.0, 2.0, 3.0]) == 3
    assert n_distinct([1.0, 2.0, 3.0, 4.0]) == 4

    # spread under the fiducial 1-sigma is an upper limit; above it is not
    assert is_upper_limit(0.01, 1.0, 0.05)
    assert not is_upper_limit(0.10, 1.0, 0.05)
    assert not is_upper_limit(np.nan, 1.0, 0.05)
    assert not is_upper_limit(0.01, 1.0, np.nan)

    # the real z=0 tau_eff numbers from the 2026-08-24 EX run
    ex0, ex1, ex2, ex3 = 0.0275900, 0.0261500, 0.0299493, 0.0258625
    assert abs(frac_spread([ex0, ex1, ex2, ex3]) - 0.148) < 0.002

    # the patched CDDF anchors must actually be inside the production grid,
    # otherwise cddf_lowN/cddf_slope silently return NaN the way the 13.0
    # anchor does
    assert CDDF_LO > 13.111, 'low-N anchor must clear the first bin centre'
    assert set(_patched_obs_extractors()) == set(_obs_extractors())
    assert set(_patched_obs_error_extractors()) == set(_obs_error_extractors())

    # numerator and denominator must use the SAME extractor set -- an anchor
    # mismatch between EX and 1P silently NaNs every CDDF ratio
    assert _patched_obs_extractors().keys() == _patched_obs_error_extractors().keys()

    print('self-test OK')


# =====================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--ex-root', type=Path,
                    default=Path('output/analysis/IllustrisTNG/EX'))
    ap.add_argument('--p1-root', type=Path,
                    default=Path('output/analysis/IllustrisTNG/1P'))
    ap.add_argument('--cosmo-csv', type=Path,
                    default=Path('data/IllustrisTNG/1P/'
                                 'CosmoAstroSeed_IllustrisTNG_L25n256_1P.csv'))
    ap.add_argument('--snaps', default=','.join(DEFAULT_SNAPS))
    ap.add_argument('--out-dir', type=Path, default=Path('plots/ex_robustness'))
    ap.add_argument('--self-test', action='store_true',
                    help='check the spread arithmetic and exit')
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0

    snaps = [s.strip() for s in args.snaps.split(',') if s.strip()]
    print(f'EX   root: {args.ex_root}')
    print(f'1P   root: {args.p1_root}')
    print(f'snapshots: {len(snaps)}')

    ex, cos = build(args.ex_root, args.p1_root, args.cosmo_csv, snaps)

    missing = gate(ex, cos, snaps)
    if missing:
        print(f'\n{len(missing)} incomplete variant/snap dirs:')
        for m in missing[:20]:
            print(f'  MISSING {m}')
        if len(missing) > 20:
            print(f'  ... and {len(missing) - 20} more')
        print('\nA hole changes which two members set max - min, so the lever arm\n'
              'stops being the same at every redshift. Sync /work down first.')
        return 1
    print(f'grid complete: {len(EX_SIMS)} EX + {5 * len(COSMO_SCANS)} 1P '
          f'x {len(snaps)} snapshots')

    tab = robustness_table(ex, cos, snaps)

    _setup_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_robustness_ratio(tab, args.out_dir / 'robustness_ratio_vs_z.png')
    plot_spreads(tab, args.out_dir / 'spreads_vs_z.png')
    plot_ex_observables(ex, tab, snaps, args.out_dir / 'ex_observables_vs_z.png')
    plot_cddf_grid(ex, snaps, args.out_dir / 'ex_cddf_grid.png')

    jpath = args.out_dir / 'robustness.json'
    jpath.write_text(json.dumps(tab, indent=2, default=float))
    print(f'  saved {jpath}')

    # Rank observables by their worst-case R across redshift: the number to read.
    # Upper-limit points are excluded -- an R built from a spread that never
    # cleared the noise says nothing about robustness, and including it would
    # rank an unmeasurable observable as the safest one on the list.
    print('\nWorst-case R over measured snapshots (lower = more cosmology-robust):')
    rank = []
    for name, rec in tab['observables'].items():
        ul = np.array(rec['upper_limit'], bool)
        r = np.array(rec['R_p1'], float)[~ul]
        r2 = np.array(rec['R_p2'], float)[~ul]
        r = np.concatenate([r, r2])
        r = r[np.isfinite(r)]
        rank.append((r.max() if r.size else np.nan, int((~ul).sum()), name))
    for worst, n_meas, name in sorted(rank, key=lambda t: (np.isnan(t[0]), t[0])):
        if not np.isfinite(worst):
            print(f'  {name:12s} R_max =      n/a   '
                  f'no measured snapshot (all {len(snaps)} under the 1-sigma floor)')
        else:
            flag = 'robust' if worst < 1 else 'feedback-contaminated'
            print(f'  {name:12s} R_max = {worst:8.3f}   {flag}'
                  f'   ({n_meas}/{len(snaps)} snapshots measured)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
