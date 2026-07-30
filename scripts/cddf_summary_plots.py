#!/usr/bin/env python3
"""Cross-snapshot summary figures for the CDDF threshold/implementation test.

`scripts/cddf_threshold_test.py` produces one set of plots per snapshot. The two
decisions it feeds -- fixed cells vs per-feature deblending, and which fit range to
quote -- are cross-redshift judgements, so this puts all snapshots on one axis.

Reads only the CSVs that test already wrote (`cddf_variants.csv`,
`cddf_variants_summary.csv`); it never touches a spectra file, needs no compiled
extension, and runs in seconds on a login node. Missing snapshots are skipped rather
than fatal, in the style of scripts/hypothesis_test_p1.py.

Two figures:

  cddf_summary_decision.png     per redshift: beta vs threshold with the threshold-free
                                definitions (50 km/s cells, whole sightline) as
                                horizontal reference lines, over median feature width vs
                                threshold with the sightline length marked. Where the
                                width curve sits on the sightline length, the
                                per-feature definition has collapsed and no threshold
                                rescues it.

  cddf_beta_vs_fit_floor.png    beta as a continuous function of the fit-range floor,
                                one curve per threshold. Separates "the threshold moves
                                beta" from "the fit range moves beta".

Usage:
    python scripts/cddf_summary_plots.py [--snaps 080 044 014]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Thresholds swept by cddf_threshold_test.py. Kept as a literal rather than imported:
# importing that module pulls in the compiled _analysis_cpp extension, and this script
# is meant to run anywhere the CSVs can be read.
TAU_THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0)

# Fit floors scanned in figure B, against a fixed ceiling.
FLOOR_GRID = np.arange(12.0, 15.01, 0.1)
FIT_CEILING = 16.0

MIN_FIT_BINS = 4


def _fit_beta(log10_N, f_N, counts, floor, ceiling):
    """beta = -slope of log10 f(N) vs log10 N, unweighted, populated bins only.

    Same convention as cddf_threshold_test.fit_beta (which mirrors the C++): strict
    inequalities on the bin centre, no epsilon floor on f(N).
    """
    mask = (log10_N > floor) & (log10_N < ceiling) & (counts > 0) & (f_N > 0)
    if mask.sum() < MIN_FIT_BINS:
        return np.nan
    slope = np.polyfit(log10_N[mask], np.log10(f_N[mask]), 1)[0]
    return float(-slope)


def _read_long_csv(path):
    """Parse cddf_variants.csv: '# key = value' header, then a CSV body.

    Rows for the fake_spectra reference variants have an empty `counts` field, so the
    body cannot go through a plain float conversion.
    """
    meta, header, rows = {}, None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip('\n')
            if line.startswith('#'):
                body = line.lstrip('#').strip()
                if '=' in body:
                    k, v = body.split('=', 1)
                    meta[k.strip()] = v.strip()
                continue
            if not line.strip():
                continue
            if header is None:
                header = line.split(',')
                continue
            rows.append(line.split(','))
    if header is None:
        raise ValueError(f'no CSV header in {path}')

    idx = {name: i for i, name in enumerate(header)}
    out = {}
    for r in rows:
        var = r[idx['variant']]
        rec = out.setdefault(var, {'log10_N': [], 'f_N': [], 'counts': []})
        rec['log10_N'].append(float(r[idx['log10_N']]))
        rec['f_N'].append(float(r[idx['f_N']]))
        c = r[idx['counts']].strip()
        rec['counts'].append(float(c) if c else np.nan)
    for rec in out.values():
        for k in rec:
            rec[k] = np.asarray(rec[k], dtype=float)
    return meta, out


def _read_summary_csv(path):
    """Parse cddf_variants_summary.csv into {variant: {column: value}}."""
    with open(path) as fh:
        lines = [ln.rstrip('\n') for ln in fh if ln.strip()]
    header = lines[0].split(',')
    out = {}
    for ln in lines[1:]:
        parts = ln.split(',')
        rec = {}
        for name, val in zip(header, parts):
            val = val.strip()
            if name == 'variant':
                rec[name] = val
                continue
            try:
                rec[name] = float(val) if val else np.nan
            except ValueError:
                rec[name] = np.nan
        out[rec['variant']] = rec
    return out


def load_snapshot(root, snap):
    """Return None (with a note) if this snapshot was not run."""
    d = Path(root) / f'snap-{snap}'
    long_path = d / 'cddf_variants.csv'
    summary_path = d / 'cddf_variants_summary.csv'
    if not long_path.exists() or not summary_path.exists():
        print(f'  skipping snap-{snap}: no CSVs under {d}')
        return None
    meta, curves = _read_long_csv(long_path)
    summary = _read_summary_csv(summary_path)
    return {
        'snap': snap,
        'redshift': float(meta.get('redshift', 'nan')),
        'n_pixels': float(meta.get('n_pixels', 'nan')),
        'n_sightlines': float(meta.get('n_sightlines', 'nan')),
        'curves': curves,
        'summary': summary,
    }


def _setup_style():
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 8


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


def plot_decision(snaps, out_path, beta_col='beta_13.5_15'):
    """beta and feature width vs threshold, with the threshold-free definitions marked."""
    n = len(snaps)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 7.5), squeeze=False,
                             sharex=True)

    for j, s in enumerate(snaps):
        ax_b, ax_w = axes[0][j], axes[1][j]
        smry = s['summary']

        beta = [smry.get(f'feature_tau{t:g}_sum', {}).get(beta_col, np.nan)
                for t in TAU_THRESHOLDS]
        beta_max = [smry.get(f'feature_tau{t:g}_max', {}).get(beta_col, np.nan)
                    for t in TAU_THRESHOLDS]
        ax_b.plot(TAU_THRESHOLDS, beta, 'o-', color='C0', label='per-feature, sum')
        ax_b.plot(TAU_THRESHOLDS, beta_max, 's--', color='C3', ms=4, alpha=0.7,
                  label='per-feature, max (current)')

        for key, lab, col in (('cpp_cells50', '50 km/s cells (threshold-free)', 'C1'),
                              ('cpp_sightline', 'whole sightline', 'C4')):
            val = smry.get(key, {}).get(beta_col, np.nan)
            if np.isfinite(val):
                ax_b.axhline(val, color=col, ls='-.', lw=1.5, label=lab)

        ax_b.axhspan(1.5, 1.7, color='grey', alpha=0.2, label=r'observed $\beta\approx1.5-1.7$')
        ax_b.axvline(0.5, color='r', ls=':', lw=1)
        ax_b.set_xscale('log')
        ax_b.set_title(f"z = {s['redshift']:.2f}  (snap {s['snap']})")
        if j == 0:
            ax_b.set_ylabel(rf'$\beta$, fit {beta_col.replace("beta_", "").replace("_", "-")}')
            ax_b.legend(loc='best')
        ax_b.grid(alpha=0.3)

        width = [smry.get(f'feature_tau{t:g}_sum', {}).get('median_feature_pixels', np.nan)
                 for t in TAU_THRESHOLDS]
        ax_w.plot(TAU_THRESHOLDS, width, 'o-', color='C0')
        ax_w.axhline(s['n_pixels'], color='k', ls='--', lw=1.5,
                     label='full sightline (definition collapsed)')
        ax_w.axvline(0.5, color='r', ls=':', lw=1)
        ax_w.set_xscale('log')
        ax_w.set_yscale('log')
        ax_w.set_xlabel(r'absorber threshold $\tau_{\rm th}$')
        if j == 0:
            ax_w.set_ylabel('median feature width [pixels]')
        ax_w.legend(loc='best')
        ax_w.grid(alpha=0.3)

    fig.suptitle('CDDF: does the absorber definition survive, and does it agree with '
                 'the threshold-free ones?', y=0.98)
    fig.tight_layout()
    _save(fig, out_path)


def plot_beta_vs_floor(snaps, out_path):
    """beta as a continuous function of the fit floor, one curve per threshold."""
    n = len(snaps)
    # Not sharey: at z=6 beta spans -3 to +1 and would flatten the other panels.
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.4), squeeze=False, sharey=False)

    colours = plt.cm.viridis(np.linspace(0, 0.9, len(TAU_THRESHOLDS)))
    for j, s in enumerate(snaps):
        ax = axes[0][j]
        lowest_populated = np.inf
        for t, c in zip(TAU_THRESHOLDS, colours):
            rec = s['curves'].get(f'feature_tau{t:g}_sum')
            if rec is None:
                continue
            betas = [_fit_beta(rec['log10_N'], rec['f_N'], rec['counts'], f, FIT_CEILING)
                     for f in FLOOR_GRID]
            ax.plot(FLOOR_GRID, betas, '-', color=c, label=rf'$\tau_{{\rm th}}={t:g}$')
            pop = rec['log10_N'][(rec['counts'] > 0) & (rec['f_N'] > 0)]
            if pop.size:
                lowest_populated = min(lowest_populated, float(pop.min()))

        rec = s['curves'].get('cpp_cells50')
        if rec is not None:
            betas = [_fit_beta(rec['log10_N'], rec['f_N'], rec['counts'], f, FIT_CEILING)
                     for f in FLOOR_GRID]
            ax.plot(FLOOR_GRID, betas, 'k--', lw=2, label='50 km/s cells')

        # Left of this line no bin is populated, so lowering the floor changes
        # nothing: the flat section there is an artefact, not stability.
        if np.isfinite(lowest_populated):
            ax.axvspan(FLOOR_GRID[0], lowest_populated, color='k', alpha=0.07)
            ax.axvline(lowest_populated, color='k', ls=':', lw=1.2,
                       label='lowest populated bin\n(flat to the left = no data)')

        ax.axhspan(1.5, 1.7, color='grey', alpha=0.2)
        ax.set_xlabel(r'fit floor $\log_{10} N_{\rm HI}$')
        ax.set_ylabel(r'fitted $\beta$')
        ax.set_title(f"z = {s['redshift']:.2f}  (snap {s['snap']})")
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(ncol=2, fontsize=7)

    fig.suptitle(rf'$\beta$ vs fit floor (ceiling fixed at $\log N = {FIT_CEILING:g}$); '
                 'grey band is an UNSOURCED placeholder', y=1.0)
    fig.tight_layout()
    _save(fig, out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--analysis-root', default='output/cddf_test',
                    help='Root written by cddf_threshold_test.py')
    ap.add_argument('--suite', default='IllustrisTNG')
    ap.add_argument('--sim-set', default='1P')
    ap.add_argument('--sim', default='1P_p1_0')
    ap.add_argument('--snaps', nargs='+', default=['080', '044', '014'],
                    help='Snapshot numbers, in the order to plot (default: low z first)')
    ap.add_argument('--beta-col', default='beta_13.5_15',
                    help='Which fitted range to show in the decision figure')
    ap.add_argument('--out-dir', default=None,
                    help='Default: plots/cddf_test/<suite>/<sim_set>/<sim>/summary')
    args = ap.parse_args()

    rel = Path(args.suite) / args.sim_set / args.sim
    root = Path(args.analysis_root) / rel
    out_dir = Path(args.out_dir) if args.out_dir else Path('plots/cddf_test') / rel / 'summary'

    print(f'Reading {root}')
    snaps = [s for s in (load_snapshot(root, n) for n in args.snaps) if s is not None]
    if not snaps:
        print('No snapshots found. Run scripts/cddf_threshold_test.py first.')
        return 1
    print(f'Loaded {len(snaps)} snapshot(s): '
          + ', '.join(f"snap-{s['snap']} (z={s['redshift']:.2f})" for s in snaps))

    _setup_style()
    plot_decision(snaps, out_dir / 'cddf_summary_decision.png', beta_col=args.beta_col)
    plot_beta_vs_floor(snaps, out_dir / 'cddf_beta_vs_fit_floor.png')
    return 0


if __name__ == '__main__':
    sys.exit(main())
