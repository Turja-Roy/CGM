"""Flux-PDF and optical-depth-PDF overlays.

Two views of flux_pdf.csv / tau_pdf.csv: across redshift for one simulation, and
across variants at one redshift. Both figures are 2 x 2 -- PDFs on top, the same
curves divided by a reference below. The ratio row is the discriminant: raw PDFs
differ so much in normalisation that the cosmology dependence is otherwise
invisible.

CSV-only, no HDF5 needed.

    python scripts/pdf_evolution.py \\
        --analysis-root output/analysis/IllustrisTNG/1P \\
        --sims 1P_p1_0 \\
        --snaps snap-024,snap-032,snap-044,snap-080 \\
        --out-dir plots/pdf_evolution
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors

VARIANT_SUFFIXES = ['n2', 'n1', '0', '1', '2']
FIDUCIAL = '0'


# =====================================================================
# Loading
# =====================================================================

def _parse_headered_csv(path):
    """Read a CSV that begins with '# key = value' lines -> (header dict, frame)."""
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
    return header, pd.read_csv(path, skiprows=data_start)


def load_pdf(snap_dir, kind='flux'):
    """Load one PDF file ('flux', 'tau', 'flux_rescaled').

    Returns (x, density, density_err, header), or None if the file is absent.
    """
    fname = {'flux': 'flux_pdf.csv',
             'tau': 'tau_pdf.csv',
             'flux_rescaled': 'flux_pdf_rescaled.csv'}[kind]
    path = Path(snap_dir) / fname
    if not path.exists():
        return None
    header, df = _parse_headered_csv(path)
    xcol = 'log_tau_bin_center' if kind == 'tau' else 'flux_bin_center'
    if xcol not in df.columns:
        return None
    return (df[xcol].values, df['density'].values,
            df.get('density_err', pd.Series(np.full(len(df), np.nan))).values,
            header)


def redshift_of(snap_dir):
    """Redshift for a snapshot directory, read from the cddf.csv header."""
    path = Path(snap_dir) / 'cddf.csv'
    if not path.exists():
        return np.nan
    header, _ = _parse_headered_csv(path)
    return float(header.get('redshift', np.nan))


# =====================================================================
# Plot helpers
# =====================================================================

def _ratio(y, y_ref):
    """y / y_ref, with zero-reference bins blanked rather than sent to infinity."""
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.where(y_ref > 0, y / y_ref, np.nan)
    return out


def _ratio_err(y, yerr, y_ref, yerr_ref):
    """Fractional errors in quadrature, including the reference's own noise."""
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = np.where(y > 0, yerr / y, np.nan)
        rel_ref = np.where(y_ref > 0, yerr_ref / y_ref, np.nan)
        return np.abs(_ratio(y, y_ref)) * np.sqrt(rel ** 2 + rel_ref ** 2)


def _panel(ax, x, y, yerr, color, label=None, band=True):
    ax.plot(x, y, '-', color=color, lw=1.6, label=label)
    if band and yerr is not None and np.any(np.isfinite(yerr)):
        ax.fill_between(x, np.clip(y - yerr, 0, None), y + yerr,
                        color=color, alpha=0.2, lw=0)


def _finish(fig, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out_path}')


# =====================================================================
# Figure 1: PDF evolution with redshift, one simulation
# =====================================================================

def plot_pdf_redshift_evolution(sim_dir, snaps, out_path, ref_snap=None):
    """Flux and log-tau PDFs at every available redshift, colour-mapped by z.

    Bottom row divides by `ref_snap` (default: the middle redshift).
    """
    sim_dir = Path(sim_dir)
    entries = []
    for snap in snaps:
        d = sim_dir / snap
        z = redshift_of(d)
        flux = load_pdf(d, 'flux')
        tau = load_pdf(d, 'tau')
        if flux is None and tau is None:
            continue
        entries.append({'snap': snap, 'z': z, 'flux': flux, 'tau': tau})

    if not entries:
        print(f'  [pdf-evolution] no PDF CSVs under {sim_dir}')
        return

    entries.sort(key=lambda e: e['z'] if np.isfinite(e['z']) else np.inf)
    zs = np.array([e['z'] for e in entries], float)

    if ref_snap is None:
        ref = entries[len(entries) // 2]
    else:
        ref = next((e for e in entries if e['snap'] == ref_snap), entries[0])

    finite_z = zs[np.isfinite(zs)]
    norm = mcolors.Normalize(vmin=finite_z.min(), vmax=finite_z.max()) \
        if finite_z.size else mcolors.Normalize(0, 1)
    cmap = cm.viridis

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for kind, col, xlabel in (('flux', 0, r'$F = e^{-\tau}$'),
                              ('tau', 1, r'$\log_{10} \tau$')):
        ax_top, ax_bot = axes[0][col], axes[1][col]
        ref_data = ref[kind]
        for e in entries:
            data = e[kind]
            if data is None:
                continue
            x, y, yerr, _ = data
            color = cmap(norm(e['z'])) if np.isfinite(e['z']) else 'gray'
            _panel(ax_top, x, y, yerr, color)
            if ref_data is not None:
                xr, yr, yrerr, _ = ref_data
                if len(xr) == len(x):
                    _panel(ax_bot, x, _ratio(y, yr),
                           _ratio_err(y, yerr, yr, yrerr), color)

        ax_top.set_yscale('log')
        ax_top.set_xlabel(xlabel)
        ax_top.set_ylabel('probability density')
        ax_top.set_title(f'{"Flux" if kind == "flux" else "Optical depth"} PDF')
        ax_top.grid(alpha=0.3, which='both')

        ax_bot.axhline(1.0, color='gray', lw=0.8, ls=':')
        ax_bot.set_yscale('log')
        ax_bot.set_xlabel(xlabel)
        ax_bot.set_ylabel(f'ratio to z = {ref["z"]:.2f}')
        ax_bot.set_title('ratio to reference redshift')
        ax_bot.grid(alpha=0.3, which='both')

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label='redshift z', fraction=0.03, pad=0.02)

    fig.suptitle(f'Flux and optical-depth PDFs vs redshift — {sim_dir.name}\n'
                 f'bands: Poisson error per bin', fontsize=13)
    # No tight_layout: the colourbar is attached to the whole axes grid.
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out_path}')


# =====================================================================
# Figure 2: PDF across variants at one redshift -- the discriminant
# =====================================================================

def plot_pdf_variant_comparison(analysis_root, scan, snap, out_path,
                                param_label=None):
    """Flux and log-tau PDFs for the five variants of one 1P scan at one snapshot.

    The ratio-to-fiducial row answers where the parameter acts: a uniform shift is
    what a mean-flux rescaling removes, structure in the ratio is what survives it.
    """
    analysis_root = Path(analysis_root)
    entries = []
    for suffix in VARIANT_SUFFIXES:
        d = analysis_root / f'1P_{scan}_{suffix}' / snap
        flux = load_pdf(d, 'flux')
        tau = load_pdf(d, 'tau')
        if flux is None and tau is None:
            continue
        entries.append({'suffix': suffix, 'z': redshift_of(d),
                        'flux': flux, 'tau': tau})

    if not entries:
        print(f'  [pdf-variants] no PDF CSVs for {scan} {snap} -- skipping')
        return

    ref = next((e for e in entries if e['suffix'] == FIDUCIAL), entries[0])
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(entries)))
    z = ref['z']

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for kind, col, xlabel in (('flux', 0, r'$F = e^{-\tau}$'),
                              ('tau', 1, r'$\log_{10} \tau$')):
        ax_top, ax_bot = axes[0][col], axes[1][col]
        ref_data = ref[kind]
        for e, c in zip(entries, colors):
            data = e[kind]
            if data is None:
                continue
            x, y, yerr, _ = data
            _panel(ax_top, x, y, yerr, c, label=e['suffix'])
            if ref_data is not None:
                xr, yr, yrerr, _ = ref_data
                if len(xr) == len(x):
                    _panel(ax_bot, x, _ratio(y, yr),
                           _ratio_err(y, yerr, yr, yrerr), c)

        ax_top.set_yscale('log')
        ax_top.set_xlabel(xlabel)
        ax_top.set_ylabel('probability density')
        ax_top.set_title(f'{"Flux" if kind == "flux" else "Optical depth"} PDF')
        ax_top.grid(alpha=0.3, which='both')

        ax_bot.axhline(1.0, color='gray', lw=0.8, ls=':')
        ax_bot.set_xlabel(xlabel)
        ax_bot.set_ylabel('ratio to fiducial')
        ax_bot.set_title('ratio to fiducial variant')
        ax_bot.grid(alpha=0.3, which='both')

    axes[0][0].legend(title=param_label or scan, fontsize=8)
    zlabel = f'z = {z:.2f}' if np.isfinite(z) else snap
    fig.suptitle(f'PDF response to the {param_label or scan} scan — {snap} ({zlabel})\n'
                 f'bands: Poisson error per bin', fontsize=13)
    _finish(fig, out_path)


# =====================================================================
# CLI
# =====================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--analysis-root', required=True,
                    help='e.g. output/analysis/IllustrisTNG/1P')
    ap.add_argument('--sims', default=None,
                    help='comma-separated simulation dirs for the redshift-evolution '
                         'figure (default: every 1P_* dir found)')
    ap.add_argument('--snaps', default=None,
                    help='comma-separated snap dirs, e.g. snap-024,snap-032 '
                         '(default: every snap-* dir found)')
    ap.add_argument('--scans', default='p1,p2',
                    help='scans for the variant-comparison figures (default: p1,p2)')
    ap.add_argument('--out-dir', default='plots/pdf_evolution')
    args = ap.parse_args()

    root = Path(args.analysis_root)
    out_dir = Path(args.out_dir)

    if args.sims:
        sims = [s.strip() for s in args.sims.split(',')]
    else:
        sims = sorted(d.name for d in root.iterdir()
                      if d.is_dir() and d.name.startswith('1P_'))

    if args.snaps:
        snaps = [s.strip() for s in args.snaps.split(',')]
    else:
        seen = set()
        for sim in sims:
            for d in (root / sim).glob('snap-*'):
                seen.add(d.name)
        snaps = sorted(seen)

    print(f'Simulations: {len(sims)}   Snapshots: {len(snaps)}')

    for sim in sims:
        plot_pdf_redshift_evolution(root / sim, snaps,
                                    out_dir / f'{sim}_pdf_vs_z.png')

    labels = {'p1': r'$\Omega_0$', 'p2': r'$\sigma_8$', 'p7': r'$\Omega_b$',
              'p8': r'$h$', 'p9': r'$n_s$'}
    for scan in [s.strip() for s in args.scans.split(',')]:
        for snap in snaps:
            plot_pdf_variant_comparison(
                root, scan, snap,
                out_dir / f'{scan}_{snap}_pdf_variants.png',
                param_label=labels.get(scan))

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
