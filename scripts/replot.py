"""Regenerate the `analyze` plots from the exported CSVs alone.

Reproduced:
    camel_power_spectrum_snap_{N}.png   <- power_spectrum.csv, flux_stats.csv
    camel_cddf_snap_{N}.png             <- cddf.csv
    camel_line_widths_snap_{N}.png      <- line_widths.csv
    camel_temp_density_snap_{N}.png     <- temp_density.csv
    camel_multi_line_comparison_..png   <- metal_lines.csv (>1 line)
    camel_statistics_snap_{N}.png       <- flux_pdf.csv, tau_pdf.csv, flux_stats.csv

Needs the raw snapshot, so opt-in via --only:
    camel_snapshot_diagnostic_...png    <- data/{suite}/{set}/{sim}/snap_{N}.hdf5

Not reproducible:
    camel_sample_spectra_snap_{N}.png   -- needs the per-sightline flux array
    statistics panels 3 and 4           -- need per-sightline flux/tau; drawn as
                                          labelled placeholders so the 2x2
                                          layout still matches analyze's

Usage:
    python scripts/replot.py                       # every snapshot under output/analysis
    python scripts/replot.py output/analysis/IllustrisTNG/1P/1P_p1_0/snap-080
    python scripts/replot.py --pattern '1P_p1_*' --only cddf,power_spectrum
    python scripts/replot.py --root output/cddf_test --out-dir plots/cddf_test
    python scripts/replot.py --dry-run             # list what would be written

    python scripts/replot.py --only snapshot_diagnostic \\
        --pattern 'snap-080' \\
        --snapshot-root /home/turja/cluster_mount/CGM/data
"""

import argparse
import fnmatch
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.config as config
from scripts.plotting import (
    setup_plot_style,
    plot_flux_power_spectrum,
    plot_column_density_distribution,
    plot_line_width_distribution,
    plot_temperature_density_relation,
    plot_multi_line_comparison,
    plot_flux_statistics,
    plot_snapshot_diagnostic,
)

PLOT_TYPES = ['power_spectrum', 'cddf', 'line_widths', 'temp_density',
              'multi_line_comparison', 'statistics', 'snapshot_diagnostic']

# snapshot_diagnostic is the one plot that needs the raw snapshot rather than a
# CSV. It is excluded from the default set and only runs when named in --only:
# PartType0 is gzip-chunked, so the strided read decompresses the whole ~0.5 GB
# of particle data per snapshot -- about 10 minutes each over an sshfs mount.
CSV_ONLY_TYPES = [t for t in PLOT_TYPES if t != 'snapshot_diagnostic']


# =====================================================================
# CSV loading
# =====================================================================

def parse_headered_csv(path):
    """Read a CSV that may begin with '# key = value' lines -> (header, frame).

    Values are coerced to float where possible; trailing units ('37.25 Mpc
    (comoving)') are dropped, keeping the first token only. Same convention as
    scripts.comparison._parse_csv_comment_header and
    scripts.pdf_evolution._parse_headered_csv.
    """
    header = {}
    data_start = 0
    with open(path, 'r') as fh:
        for i, line in enumerate(fh):
            if not line.startswith('#'):
                data_start = i
                break
            body = line.lstrip('#').strip()
            if '=' not in body:
                continue
            key, _, val = body.partition('=')
            tokens = val.strip().split()
            val = tokens[0] if tokens else ''
            try:
                header[key.strip()] = float(val)
            except ValueError:
                header[key.strip()] = val
    return header, pd.read_csv(path, skiprows=data_start)


def _int_or_none(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def load_flux_stats(snap_dir):
    """flux_stats.csv is a two-column statistic,value table."""
    path = Path(snap_dir) / 'flux_stats.csv'
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return dict(zip(df['statistic'], pd.to_numeric(df['value'], errors='coerce')))


def load_power_spectrum(snap_dir, flux_stats):
    """P_F(k) plus the two scalars the plot's info box needs.

    n_sightlines is not in power_spectrum.csv; it comes from flux_stats.csv
    (newer runs) or the cddf.csv header (older ones), resolved by the caller.
    """
    path = Path(snap_dir) / 'power_spectrum.csv'
    if not path.exists():
        return None
    df = pd.read_csv(path)
    k = df['k_s_per_km'].values
    zeros = np.zeros_like(k)
    return {
        'k': k,
        'P_k_mean': df['P_k_mean_km_per_s'].values,
        'P_k_std': df['P_k_std'].values if 'P_k_std' in df else zeros,
        'P_k_err': df['P_k_err'].values if 'P_k_err' in df else zeros,
        'mean_flux': flux_stats.get('mean_flux', np.nan),
        'n_sightlines': None,
    }


def load_cddf(snap_dir):
    """CDDF table plus the config echo, so the plot picks the right units.

    Files written before the config echo existed carry the per-dex Mpc^-1
    quantity and leave norm_mode / dx_mode as None, which the plotting code
    already treats as "not per-dN".
    """
    path = Path(snap_dir) / 'cddf.csv'
    if not path.exists():
        return None
    meta, df = parse_headered_csv(path)
    log10_N = df['log10_N_HI'].values
    bin_centers = df['bin_center'].values if 'bin_center' in df else 10.0 ** log10_N
    if 'delta_log_N' in df:
        delta_log_N = df['delta_log_N'].values
    else:
        d = np.diff(log10_N)
        delta_log_N = np.full_like(log10_N, np.median(d) if len(d) else np.nan)
    f_N = df['f_N_HI'].values
    if len(log10_N):
        bins = 10.0 ** np.append(log10_N - delta_log_N / 2.0,
                                 log10_N[-1] + delta_log_N[-1] / 2.0)
    else:
        bins = np.array([])

    return {
        'log10_N_HI': log10_N,
        'bin_centers': bin_centers,
        'bins': bins,
        'counts': df['counts'].values if 'counts' in df else np.zeros_like(log10_N),
        'f_N': f_N,
        'f_N_HI': f_N,
        'f_N_HI_err': df['f_N_HI_err'].values if 'f_N_HI_err' in df else None,
        'delta_log_N': delta_log_N,
        'beta_fit': meta.get('beta_fit', np.nan),
        'beta_fit_err': meta.get('beta_fit_err', np.nan),
        'beta_fit_weighted': meta.get('beta_fit_weighted', np.nan),
        'beta_fit_raw': meta.get('beta_fit_raw', np.nan),
        'n_absorbers': _int_or_none(meta.get('n_absorbers')) or 0,
        'n_sightlines': _int_or_none(meta.get('n_sightlines')),
        # dx_mode = 1 writes the header key as "X", not "dX".
        'dX': meta.get('dX', meta.get('X', np.nan)),
        'redshift': meta.get('redshift'),
        'norm_mode': _int_or_none(meta.get('norm_mode')),
        'dx_mode': _int_or_none(meta.get('dx_mode')),
        'absorber_mode': _int_or_none(meta.get('absorber_mode')),
        'log_N_min': meta.get('log_N_min'),
        'fit_log_N_min': meta.get('fit_log_N_min'),
        'fit_log_N_max': meta.get('fit_log_N_max'),
        'saturated': str(meta.get('saturated', '')).lower().startswith('true'),
    }


def load_line_widths(snap_dir):
    """b-parameters per absorber; the summary scalars are recomputed, not stored."""
    path = Path(snap_dir) / 'line_widths.csv'
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if 'b_param_km_s' not in df or len(df) == 0:
        return None
    b = df['b_param_km_s'].values
    return {
        'N_HI': df['N_HI'].values if 'N_HI' in df else np.array([]),
        'b_params': b,
        'b_median': float(np.median(b)),
        'b_mean': float(np.mean(b)),
        'b_std': float(np.std(b)),
        'n_absorbers': int(len(b)),
    }


def load_temp_density(snap_dir):
    """T-rho scatter (subsampled at export time) plus the fit from the header.

    n_pixels in the header is the full pre-subsample count, which is what the
    plot's info box reported originally; the histogram uses whatever rows
    survived export.
    """
    path = Path(snap_dir) / 'temp_density.csv'
    if not path.exists():
        return None
    meta, df = parse_headered_csv(path)
    has_cols = 'log_temperature' in df and 'log_density' in df and len(df) > 0
    return {
        'T0': meta.get('T0', np.nan),
        'T0_err': meta.get('T0_err', np.nan),
        'gamma': meta.get('gamma', np.nan),
        'gamma_err': meta.get('gamma_err', np.nan),
        'n_pixels': _int_or_none(meta.get('n_pixels')) or (len(df) if has_cols else 0),
        'log_T': df['log_temperature'].values if has_cols else np.array([]),
        'log_rho': df['log_density'].values if has_cols else np.array([]),
    }


def load_metal_lines(snap_dir):
    """Per-ion summary statistics.

    column_densities is set empty: the per-absorber catalogue was never
    exported, so panel 4 of the multi-line figure comes out blank rather than
    invented.
    """
    path = Path(snap_dir) / 'metal_lines.csv'
    if not path.exists():
        return []
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        rows.append({
            'ion_name': row['ion_name'],
            'n_absorbers': int(row.get('n_absorbers', 0)),
            'dN_dz': float(row.get('dN_dz', np.nan)),
            'covering_fraction': float(row.get('covering_fraction', np.nan)),
            'mean_tau': float(row.get('mean_tau', np.nan)),
            'median_tau': float(row.get('median_tau', np.nan)),
            'column_densities': np.array([]),
        })
    return rows


def load_pdfs(snap_dir):
    """Rebuild the compute_flux_tau_pdf dict from flux_pdf.csv and tau_pdf.csv.

    The keys match what analyze passes to plot_flux_statistics, so the same
    plotting function serves both paths. Returns None when neither file exists;
    a dict missing one half is fine -- the plot renders that panel empty.
    """
    pdf = {}

    flux_path = Path(snap_dir) / 'flux_pdf.csv'
    if flux_path.exists():
        header, df = parse_headered_csv(flux_path)
        if 'flux_bin_center' in df.columns:
            pdf.update({
                'flux_bin_centers': df['flux_bin_center'].values,
                'flux_density': df['density'].values,
                'flux_density_err': (df['density_err'].values
                                     if 'density_err' in df
                                     else np.full(len(df), np.nan)),
                'flux_counts': df['count'].values if 'count' in df else None,
                'n_pixels': _int_or_none(header.get('n_pixels')),
            })
            if 'flux_bin_low' in df and 'flux_bin_high' in df:
                pdf['flux_bin_edges'] = np.append(df['flux_bin_low'].values,
                                                  df['flux_bin_high'].values[-1])

    tau_path = Path(snap_dir) / 'tau_pdf.csv'
    if tau_path.exists():
        header, df = parse_headered_csv(tau_path)
        if 'log_tau_bin_center' in df.columns:
            pdf.update({
                'log_tau_bin_centers': df['log_tau_bin_center'].values,
                'log_tau_density': df['density'].values,
                'log_tau_density_err': (df['density_err'].values
                                        if 'density_err' in df
                                        else np.full(len(df), np.nan)),
                'log_tau_counts': df['count'].values if 'count' in df else None,
                'n_tau_in_grid': _int_or_none(header.get('n_pixels_in_grid')),
                'frac_tau_zero': header.get('frac_tau_zero', np.nan),
                'frac_tau_underflow': header.get('frac_tau_underflow', np.nan),
                'frac_tau_overflow': header.get('frac_tau_overflow', np.nan),
            })
            if 'log_tau_bin_low' in df and 'log_tau_bin_high' in df:
                pdf['log_tau_bin_edges'] = np.append(
                    df['log_tau_bin_low'].values, df['log_tau_bin_high'].values[-1])

    return pdf or None


# =====================================================================
# Driving one snapshot directory
# =====================================================================

def plot_path(snap_dir, root, out_dir, plot_type, suffix=''):
    """Mirror config.get_plot_output_name, but keyed on the analysis directory.

    analyze derives plots/{suite}/{sim_set}/{sim_name}/camel_{type}_snap_{N}.png
    from the spectra path. The analysis tree has the same shape with snap-{N} as
    a final directory, so the plot path is the analysis path minus that level.
    """
    rel = Path(snap_dir).resolve().relative_to(Path(root).resolve())
    snap_num = rel.name.replace('snap-', '').replace('snap_', '')
    return Path(out_dir) / rel.parent / f'camel_{plot_type}_snap_{snap_num}{suffix}.png'


def snapshot_file_for(snap_dir, root, snapshot_root):
    """The raw snapshot matching an analysis directory, or None.

    The analysis tree mirrors the data tree, so
    output/analysis/{suite}/{set}/{sim}/snap-{N} maps onto
    {snapshot_root}/{suite}/{set}/{sim}/snap_{N}.hdf5. snapshot_root may be a
    mounted copy of the cluster's data directory.
    """
    rel = Path(snap_dir).resolve().relative_to(Path(root).resolve())
    snap_num = rel.name.replace('snap-', '').replace('snap_', '')
    path = Path(snapshot_root) / rel.parent / f'snap_{snap_num}.hdf5'
    return path if path.exists() else None


def replot_snapshot(snap_dir, root, out_dir, only, suffix, dry_run,
                    snapshot_root=None, stride=100):
    """Rebuild every requested plot for one snap-{N} directory.

    Each plot is attempted independently: a snapshot missing line_widths.csv
    still gets its CDDF, and a single failing figure does not abort the rest.
    Returns (written, skipped, failed) lists of (plot_type, detail).
    """
    snap_dir = Path(snap_dir)
    written, skipped, failed = [], [], []

    setup_plot_style()

    flux_stats = load_flux_stats(snap_dir)
    cddf = load_cddf(snap_dir)

    # Redshift lives only in the cddf.csv header. NaN renders as "nan" in the
    # titles rather than crashing the format strings.
    redshift = np.nan
    if cddf is not None and cddf.get('redshift') is not None:
        redshift = float(cddf['redshift'])

    n_sightlines = flux_stats.get('n_sightlines')
    if n_sightlines is None or not np.isfinite(n_sightlines):
        n_sightlines = cddf.get('n_sightlines') if cddf else None
    else:
        n_sightlines = int(n_sightlines)

    def run(plot_type, fn):
        if only and plot_type not in only:
            return
        out = plot_path(snap_dir, root, out_dir, plot_type, suffix)
        if dry_run:
            written.append((plot_type, str(out)))
            return
        try:
            result = fn(out)
        except Exception as e:
            failed.append((plot_type, f'{type(e).__name__}: {e}'))
            plt.close('all')
            return
        if result is False:
            skipped.append((plot_type, 'input CSV missing or empty'))
        else:
            written.append((plot_type, str(out)))

    def do_power(out):
        power = load_power_spectrum(snap_dir, flux_stats)
        if power is None:
            return False
        power['n_sightlines'] = n_sightlines if n_sightlines is not None else 'N/A'
        plot_flux_power_spectrum(power, redshift, out)

    def do_cddf(out):
        if cddf is None:
            return False
        plot_column_density_distribution(cddf, redshift, out)

    def do_lwd(out):
        lwd = load_line_widths(snap_dir)
        if lwd is None or lwd['n_absorbers'] == 0:
            return False
        plot_line_width_distribution(lwd, redshift, out)

    def do_tdens(out):
        tdens = load_temp_density(snap_dir)
        if tdens is None or tdens['n_pixels'] < 100:
            return False
        plot_temperature_density_relation(tdens, redshift, out)

    def do_metal(out):
        lines = load_metal_lines(snap_dir)
        if len(lines) <= 1:
            return False
        plot_multi_line_comparison(lines, redshift, out)

    def do_stats(out):
        pdfs = load_pdfs(snap_dir)
        if pdfs is None:
            return False
        # flux and tau left out: panels 3-4 become placeholders. Everything
        # else is identical to the figure analyze draws.
        plot_flux_statistics(pdfs, flux_stats, out)

    def do_diagnostic(out):
        if snapshot_root is None:
            return False
        snapshot = snapshot_file_for(snap_dir, root, snapshot_root)
        if snapshot is None:
            return False
        plot_snapshot_diagnostic(snapshot, out, stride=stride)

    run('power_spectrum', do_power)
    run('cddf', do_cddf)
    run('line_widths', do_lwd)
    run('temp_density', do_tdens)
    run('multi_line_comparison', do_metal)
    run('statistics', do_stats)
    run('snapshot_diagnostic', do_diagnostic)

    return str(snap_dir), written, skipped, failed


# =====================================================================
# Discovery and CLI
# =====================================================================

def find_snapshot_dirs(root, pattern=None):
    """Every directory under root holding at least one analysis CSV.

    Matched on file presence, not on the snap-* naming, so hand-made or
    relocated output directories are picked up too.
    """
    root = Path(root)
    markers = {'cddf.csv', 'power_spectrum.csv', 'flux_stats.csv',
               'line_widths.csv', 'temp_density.csv', 'metal_lines.csv',
               'flux_pdf.csv', 'tau_pdf.csv'}
    dirs = []
    for dirpath, _, filenames in os.walk(root):
        if markers & set(filenames):
            dirs.append(Path(dirpath))
    if pattern:
        dirs = [d for d in dirs
                if fnmatch.fnmatch(str(d), pattern)
                or any(fnmatch.fnmatch(part, pattern) for part in d.parts)]
    return sorted(dirs)


def main():
    parser = argparse.ArgumentParser(
        description='Regenerate analyze plots from the exported CSVs, no spectra needed',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument('dirs', nargs='*',
                        help='snapshot analysis directories; default is every '
                             'one found under --root')
    parser.add_argument('--root', default=str(config.ANALYSIS_OUTPUT_DIR),
                        help='analysis tree to scan (default: output/analysis)')
    parser.add_argument('--out-dir', default=str(config.PLOTS_DIR),
                        help='plot tree to write into (default: plots/). The '
                             'directory layout under --root is mirrored here.')
    parser.add_argument('--pattern',
                        help='glob filter on the directory path or any of its '
                             "components, e.g. '1P_p1_*' or 'snap-080'")
    parser.add_argument('--only',
                        help='comma-separated subset of: ' + ', '.join(PLOT_TYPES)
                             + '. snapshot_diagnostic is excluded by default '
                               'and only runs when named here.')
    parser.add_argument('--snapshot-root', default=None,
                        help='data tree holding the raw snapshots, for '
                             'snapshot_diagnostic only, e.g. '
                             '/home/turja/cluster_mount/CGM/data. Defaults to '
                             'the local data/ directory.')
    parser.add_argument('--stride', type=int, default=100,
                        help='particle subsampling for snapshot_diagnostic '
                             '(default 100). Does not reduce I/O: the data is '
                             'gzip-chunked, so every chunk is read regardless.')
    parser.add_argument('--suffix', default='',
                        help='appended to each filename before .png, e.g. '
                             '_replot, to avoid overwriting the originals')
    parser.add_argument('--workers', type=int, default=1,
                        help='snapshot directories to process in parallel')
    parser.add_argument('--dry-run', action='store_true',
                        help='list the files that would be written, write nothing')
    parser.add_argument('--quiet', action='store_true',
                        help='one line per snapshot instead of one per plot')
    args = parser.parse_args()

    # Default excludes snapshot_diagnostic: it is the only plot that reads a raw
    # snapshot, at roughly 0.5 GB per file.
    only = CSV_ONLY_TYPES
    if args.only:
        only = [t.strip() for t in args.only.split(',') if t.strip()]
        unknown = [t for t in only if t not in PLOT_TYPES]
        if unknown:
            parser.error(f"unknown plot type(s): {', '.join(unknown)}. "
                         f"Valid: {', '.join(PLOT_TYPES)}")

    snapshot_root = args.snapshot_root or str(config.DATA_DIR)
    if 'snapshot_diagnostic' in only and not Path(snapshot_root).is_dir():
        parser.error(f'--snapshot-root does not exist: {snapshot_root}')

    root = Path(args.root).resolve()
    if args.dirs:
        snap_dirs = [Path(d).resolve() for d in args.dirs]
        missing = [d for d in snap_dirs if not d.is_dir()]
        if missing:
            parser.error('not a directory: ' + ', '.join(str(m) for m in missing))
        # An explicit directory outside --root would break the relative-path
        # mirroring, so re-root onto its own parent tree.
        rooted = []
        for d in snap_dirs:
            try:
                d.relative_to(root)
                rooted.append((d, root))
            except ValueError:
                rooted.append((d, d.parent.parent))
    else:
        rooted = [(d, root) for d in find_snapshot_dirs(root, args.pattern)]

    if not rooted:
        print(f'No analysis CSVs found under {root}'
              + (f" matching '{args.pattern}'" if args.pattern else ''))
        return 1

    print('=' * 70)
    print('REPLOTTING FROM ANALYSIS CSVs')
    print('=' * 70)
    print(f'Snapshots:  {len(rooted)}')
    print(f'Root:       {root}')
    print(f'Output:     {Path(args.out_dir).resolve()}')
    print(f'Plots:      {", ".join(only)}')
    if 'snapshot_diagnostic' in only:
        print(f'Snap data:  {snapshot_root}')
    if args.dry_run:
        print('Mode:       DRY RUN (nothing written)')
    print('-' * 70)

    n_written = n_skipped = n_failed = 0
    results = []

    if args.workers > 1 and not args.dry_run:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(replot_snapshot, d, r, args.out_dir,
                                       only, args.suffix, args.dry_run,
                                       snapshot_root, args.stride)
                       for d, r in rooted]
            for future in as_completed(futures):
                results.append(future.result())
    else:
        for d, r in rooted:
            results.append(replot_snapshot(d, r, args.out_dir, only,
                                           args.suffix, args.dry_run,
                                           snapshot_root, args.stride))

    for snap_dir, written, skipped, failed in sorted(results):
        n_written += len(written)
        n_skipped += len(skipped)
        n_failed += len(failed)
        rel = os.path.relpath(snap_dir, root)
        if args.quiet:
            print(f'{rel}: {len(written)} written, {len(skipped)} skipped, '
                  f'{len(failed)} failed')
        else:
            print(f'\n{rel}')
            for plot_type, path in written:
                print(f'  [ok]   {plot_type}: {path}')
            for plot_type, why in skipped:
                print(f'  [skip] {plot_type}: {why}')
            for plot_type, why in failed:
                print(f'  [FAIL] {plot_type}: {why}')

    print('\n' + '=' * 70)
    verb = 'would write' if args.dry_run else 'wrote'
    print(f'{verb} {n_written} plots; {n_skipped} skipped (no input); '
          f'{n_failed} failed')
    print('Never reproducible from CSVs: sample_spectra (per-sightline flux), '
          'snapshot_diagnostic (raw particles),\n'
          'statistics panels 3-4 (per-sightline flux/tau).')
    print('=' * 70)

    return 1 if n_failed else 0


if __name__ == '__main__':
    sys.exit(main())
