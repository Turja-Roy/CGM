#!/usr/bin/env python3
"""CDDF: tau-threshold sensitivity and implementation comparison.

Our CDDF is not fake_spectra's. Ours (src/cpp/analysis/column_density.cpp) calls a
contiguous run of pixels with tau > 0.5 an "absorber" and takes the *peak* per-pixel
colden as its N_HI, normalising by a comoving box length per dex. fake_spectra's
Spectra.column_density_function uses no tau threshold at all: it *sums* per-pixel
colden over either a whole sightline (line=True) or fixed 50 km/s cells
(line=False), and normalises by the dimensionless absorption distance X(z) per
linear dN.

Four things differ at once, and this script measures each separately:

  axis 1  absorber definition   contiguous tau feature / whole sightline / fixed cells
  axis 2  tau threshold         0.05 ... 2.0 (only meaningful for axis 1 = feature)
  axis 3  N per absorber        max over the feature vs sum
  axis 4  normalisation         comoving Mpc per dex vs X(z) per linear dN

The C++ takes all four as runtime options whose defaults reproduce the historical
production behaviour, so every variant below comes from the same code path that
produces cddf.csv -- not from a reimplementation.

Three validation gates run before any result is reported (see run_gates):

  G1 regression  the "current production" variant reproduces the committed
                 output/analysis/.../cddf.csv for this snapshot
  G2 equivalence C++ absorber_mode=1 == fake_spectra line=True
  G3 equivalence C++ absorber_mode=2 == fake_spectra line=False, close=50

G2/G3 are exact up to a known 0.265% offset: fake_spectra's absorption_distance uses
c = 2.99e10 cm/s, we use 2.99792458e10 (unitsystem.py:31-42 vs constants.h).

Usage:
    python scripts/cddf_threshold_test.py --spectra <file.hdf5> [--out-dir ...] [--plot-dir ...]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.config as config
from scripts.analysis_cpp import compute_column_density_distribution as cpp_cddf


# fake_spectra's default CDDF grid: edges = 10**arange(13, 23, 0.2), i.e. 50 edges
# spanning 13.0 to 22.8 in log10 N, hence 49 bins of 0.2 dex.
FS_LOG_N_MIN = 13.0
FS_LOG_N_MAX = 22.8
FS_N_BINS = 49
FS_DLOG_N = 0.2

# Production grid (column_density.cpp defaults).
PROD_LOG_N_MIN = 12.0
PROD_LOG_N_MAX = 22.0
PROD_N_BINS = 50

TAU_THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0)

# beta is fitted over several ranges: the production range, a range whose floor sits
# safely above the tau>0.5 completeness limit, and one pushed to higher N.
FIT_RANGES = (
    (12.0, 14.4771212547196624),  # production: 1e12 < N < 3e14
    (13.5, 14.5),
    (13.5, 15.0),
    (14.0, 16.0),
)

# fake_spectra unitsystem.py:19 uses a 3-digit c; constants.h uses the exact value.
# f(N) ~ 1/X ~ c, so our f(N) is larger by exactly this ratio.
C_FAKE_SPECTRA = 2.99e10
C_EXACT = 2.99792458e10
C_RATIO = C_EXACT / C_FAKE_SPECTRA


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------

def load_spectra(path):
    """Load tau, colden and the header values the CDDF needs.

    tau/colden are read straight into float32 buffers: the datasets are float64 on
    disk (~2 GB each at 10000 x 24727) and the C++ takes float32 anyway, so a
    read-then-cast would double the peak footprint for nothing.
    """
    out = {}
    with h5py.File(path, 'r') as f:
        tau_path = 'tau/H/1/1215'
        colden_path = 'colden/H/1'
        if tau_path not in f:
            raise KeyError(f'{tau_path} not in {path}')
        if colden_path not in f:
            raise KeyError(
                f'{colden_path} not in {path}: every variant here needs per-pixel '
                'column densities, and so does fake_spectra')

        dset = f[tau_path]
        tau = np.empty(dset.shape, dtype=np.float32)
        dset.read_direct(tau)

        dset = f[colden_path]
        colden = np.empty(dset.shape, dtype=np.float32)
        dset.read_direct(colden)

        h = f['Header'].attrs
        out['redshift'] = float(h.get('redshift', h.get('Redshift')))
        out['box_size_ckpc_h'] = float(h.get('box', h.get('BoxSize')))
        out['hubble'] = float(h.get('hubble', h.get('HubbleParam')))
        out['omega_m'] = float(h.get('omegam', h.get('Omega0')))
        out['velocity_spacing'] = float(h['dvbin'])
        out['discarded'] = int(h['discarded']) if 'discarded' in h else 0

    out['tau'] = tau
    out['colden'] = colden
    out['n_sightlines'], out['n_pixels'] = tau.shape
    return out


def open_fake_spectra(path):
    """Construct a fake_spectra Spectra from the savefile alone, no snapshot.

    No simulation snapshot is needed: Spectra swallows the missing snapshot
    (spectra.py:136-155) and colden is lazy-loaded from the savefile
    (load_savefile:454-457 -> _really_load_array:359-372). Three things bite:
      - res must be None; the default res=1. trips the dvbin assert (spectra.py:233-238)
      - self.units is not stored in the savefile, so it falls back to the default
        UnitSystem (kpc/h) -- correct for CAMELS, but asserted rather than assumed
      - column_density_function divides by NumLos + discarded, so discarded must be 0
        for the comparison against our n_sightlines to be apples to apples

    One object is reused for both CDDF calls: it caches colden as float64 (~2 GB at
    10000 x 24727), so a second object would double that for nothing.
    """
    from scripts.fake_spectra_fix import apply_fake_spectra_bugfixes
    apply_fake_spectra_bugfixes()
    from fake_spectra.spectra import Spectra

    path = Path(path)
    snap_num = config.extract_simulation_info(str(path))['snap_num']
    # base deliberately points at a directory with no snapshot in it: the CDDF is a
    # pure post-processing operation on the savefile.
    sp = Spectra(int(snap_num), str(path.parent), None, None,
                 savefile=str(path), reload_file=False, res=None, quiet=True)

    assert np.isclose(sp.units.UnitLength_in_cm, 3.085678e21, rtol=1e-6), (
        f'unexpected UnitLength_in_cm {sp.units.UnitLength_in_cm}: the savefile does '
        'not store the unit system, so absorption_distance would be silently wrong')
    assert sp.discarded == 0, f'discarded={sp.discarded}, expected 0'
    return sp


def fake_spectra_cddf(sp, line, close=50.0):
    """fake_spectra's column_density_function on the same grid as our C++ variants."""
    centers, f_N = sp.column_density_function(
        elem='H', ion=1, dlogN=FS_DLOG_N, minN=FS_LOG_N_MIN,
        maxN=FS_LOG_N_MAX + FS_DLOG_N, line=line, close=close, dX=True)
    return {
        'bin_centers': np.asarray(centers, dtype=float),
        'f_N': np.asarray(f_N, dtype=float),
        'NumLos': int(sp.NumLos),
        'X': float(sp.units.absorption_distance(sp.box, sp.red)),
        'dvbin': float(sp.dvbin),
    }


# ----------------------------------------------------------------------------
# fitting
# ----------------------------------------------------------------------------

def fit_beta(bin_centers, f_N, counts, fit_log_N_min, fit_log_N_max):
    """beta = -slope of log10 f(N) vs log10 N by unweighted least squares.

    Matches the C++ fit (column_density.cpp): strict inequalities on the bin centre,
    only populated bins, unweighted. Reproducing it here lets one variant be refitted
    over several ranges without another pass over the pixels.

    One deliberate difference: the C++ needs more than 5 populated bins, this needs 4,
    so that a narrow range like log N in [13.5, 14.5] (5 bins of 0.2 dex) still
    produces a number instead of a NaN.
    """
    N_lo = 10.0 ** fit_log_N_min
    N_hi = 10.0 ** fit_log_N_max
    mask = (bin_centers > N_lo) & (bin_centers < N_hi) & (counts > 0) & (f_N > 0)
    if mask.sum() < 4:
        return np.nan, int(mask.sum())
    x = np.log10(bin_centers[mask])
    y = np.log10(f_N[mask])
    slope = np.polyfit(x, y, 1)[0]
    return float(-slope), int(mask.sum())


# ----------------------------------------------------------------------------
# variants
# ----------------------------------------------------------------------------

def run_variant(data, name, **opts):
    """Run one C++ CDDF variant and attach the refits and diagnostics."""
    res = cpp_cddf(
        data['tau'], data['velocity_spacing'],
        colden=data['colden'],
        redshift=data['redshift'],
        box_size_ckpc_h=data['box_size_ckpc_h'],
        hubble=data['hubble'],
        omega_m=data['omega_m'],
        **opts)
    res['variant'] = name
    res['threshold'] = opts.get('threshold', np.nan)
    res['betas'] = {
        (lo, hi): fit_beta(res['bin_centers'], res['f_N'], res['counts'], lo, hi)
        for lo, hi in FIT_RANGES
    }
    px = res.get('feature_pixels')
    if px is not None and px.size:
        res['median_feature_pixels'] = float(np.median(px))
        res['mean_feature_pixels'] = float(np.mean(px))
        res['max_feature_pixels'] = int(np.max(px))
        res['frac_pixels_absorbing'] = float(
            px.sum() / (data['n_sightlines'] * data['n_pixels']))
    else:
        res['median_feature_pixels'] = np.nan
        res['mean_feature_pixels'] = np.nan
        res['max_feature_pixels'] = 0
        res['frac_pixels_absorbing'] = np.nan
    return res


def production_opts():
    """Exactly the options cmd_analyze uses today."""
    return dict(threshold=config.TAU_THRESHOLD_HI, absorber_mode=0, colden_mode=0,
                dx_mode=0, norm_mode=0, log_N_min=PROD_LOG_N_MIN,
                log_N_max=PROD_LOG_N_MAX, n_bins=PROD_N_BINS,
                fit_log_N_min=FIT_RANGES[0][0], fit_log_N_max=FIT_RANGES[0][1],
                min_N_gate=1e12)


def corrected_opts(threshold, colden_mode=1):
    """Per-feature absorbers, but summed colden, X(z), and per linear dN."""
    return dict(threshold=threshold, absorber_mode=0, colden_mode=colden_mode,
                dx_mode=1, norm_mode=1, log_N_min=FS_LOG_N_MIN,
                log_N_max=FS_LOG_N_MAX, n_bins=FS_N_BINS,
                fit_log_N_min=FIT_RANGES[0][0], fit_log_N_max=FIT_RANGES[0][1],
                min_N_gate=0.0)


def fake_spectra_equivalent_opts(absorber_mode, cell_dv=50.0):
    """C++ configured to be arithmetically identical to fake_spectra."""
    return dict(threshold=0.0, absorber_mode=absorber_mode, cell_dv=cell_dv,
                colden_mode=1, dx_mode=1, norm_mode=1,
                log_N_min=FS_LOG_N_MIN, log_N_max=FS_LOG_N_MAX, n_bins=FS_N_BINS,
                fit_log_N_min=FIT_RANGES[0][0], fit_log_N_max=FIT_RANGES[0][1],
                min_N_gate=0.0)


def build_variants(data):
    """All variants, keyed by name, in report order."""
    variants = {}

    print('\n[B] current production defaults')
    variants['production'] = run_variant(data, 'production', **production_opts())

    print('[A/C] threshold sweep, corrected normalisation')
    for th in TAU_THRESHOLDS:
        for cm, cm_name in ((1, 'sum'), (0, 'max')):
            name = f'feature_tau{th:g}_{cm_name}'
            print(f'  {name}')
            variants[name] = run_variant(data, name, **corrected_opts(th, colden_mode=cm))

    print('[D] fake_spectra-equivalent C++')
    variants['cpp_sightline'] = run_variant(
        data, 'cpp_sightline', **fake_spectra_equivalent_opts(1))
    variants['cpp_cells50'] = run_variant(
        data, 'cpp_cells50', **fake_spectra_equivalent_opts(2, cell_dv=50.0))

    return variants


# ----------------------------------------------------------------------------
# validation gates
# ----------------------------------------------------------------------------

def _read_cddf_csv(path):
    """Read a committed cddf.csv: '# key = value' header then a plain CSV body."""
    meta, rows = {}, []
    with open(path) as fh:
        header = None
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
    cols = {name: np.array([r[i] for r in rows], dtype=float)
            for i, name in enumerate(header)}
    return meta, cols


def gate_regression(variants, spectra_path):
    """G1: the production variant must reproduce the committed cddf.csv."""
    from scripts.data_export import get_analysis_output_dir
    csv_path = Path(get_analysis_output_dir(str(spectra_path))) / 'cddf.csv'
    if not csv_path.exists():
        return ('SKIP', f'no committed cddf.csv at {csv_path}')

    meta, cols = _read_cddf_csv(csv_path)
    v = variants['production']

    n_los_ref = int(float(meta['n_sightlines']))
    if n_los_ref != v['n_sightlines']:
        return ('SKIP', f'{csv_path} has n_sightlines = {n_los_ref} but this run used '
                        f"{v['n_sightlines']}: not comparable")

    msgs = []
    ok = True

    n_ref = int(float(meta['n_absorbers']))
    if n_ref != v['n_absorbers']:
        ok = False
        msgs.append(f"n_absorbers {v['n_absorbers']} != {n_ref}")

    # f_N must be exact; beta_fit is allowed ~1e-4 because the fit no longer adds a
    # 1e-10 floor to f(N) before taking its log (see column_density.cpp).
    beta_ref = float(meta['beta_fit'])
    if not np.isclose(beta_ref, v['beta_fit'], rtol=1e-4, atol=1e-5):
        ok = False
        msgs.append(f"beta_fit {v['beta_fit']:.6f} != {beta_ref:.6f}")
    else:
        msgs.append(f"beta_fit {v['beta_fit']:.6f} vs {beta_ref:.6f}")

    f_ref = cols['f_N_HI']
    if f_ref.size != v['f_N'].size:
        ok = False
        msgs.append(f"f_N size {v['f_N'].size} != {f_ref.size}")
    else:
        nz = (f_ref > 0) | (v['f_N'] > 0)
        if nz.any():
            dev = np.max(np.abs(v['f_N'][nz] - f_ref[nz]) /
                         np.maximum(f_ref[nz], 1e-300))
            msgs.append(f'max relative f_N deviation {dev:.2e}')
            if dev > 1e-5:
                ok = False

    return ('PASS' if ok else 'FAIL', f'{csv_path}: ' + '; '.join(msgs))


def _compare_f_N(ours, theirs, expected_ratio, rtol):
    """Compare two f(N) arrays that should agree up to a known scalar ratio."""
    if ours.size != theirs.size:
        return False, f'size {ours.size} != {theirs.size}'
    nz = (ours > 0) & (theirs > 0)
    n_only_ours = int(np.sum((ours > 0) & (theirs <= 0)))
    n_only_theirs = int(np.sum((ours <= 0) & (theirs > 0)))
    if not nz.any():
        return False, 'no bin populated in both'
    ratio = ours[nz] / theirs[nz]
    dev = np.max(np.abs(ratio / expected_ratio - 1.0))
    msg = (f'{nz.sum()} shared bins, ratio/{expected_ratio:.6f} - 1 max {dev:.2e}, '
           f'bins only in ours {n_only_ours}, only in theirs {n_only_theirs}')
    ok = bool(dev <= rtol and n_only_ours == 0 and n_only_theirs == 0)
    return ok, msg


def gate_fake_spectra(variants, spectra_path, data):
    """G2/G3: the fake_spectra-equivalent C++ modes must match fake_spectra."""
    results = []
    try:
        sp = open_fake_spectra(spectra_path)
        fs_line = fake_spectra_cddf(sp, line=True)
        fs_cell = fake_spectra_cddf(sp, line=False, close=50.0)
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        return [('G2 sightline', 'ERROR', repr(exc)),
                ('G3 cells50', 'ERROR', repr(exc))]

    if fs_line['NumLos'] != data['n_sightlines']:
        results.append(('G2 sightline', 'FAIL',
                        f"NumLos {fs_line['NumLos']} != {data['n_sightlines']}"))
        return results

    # A ratio of exactly 1 in counts is expected; f(N) differs only by c.
    for gate, key, fs in (('G2 sightline', 'cpp_sightline', fs_line),
                          ('G3 cells50', 'cpp_cells50', fs_cell)):
        v = variants[key]
        ok, msg = _compare_f_N(v['f_N'], fs['f_N'], C_RATIO, rtol=1e-6)
        x_msg = (f"X ours {v['X_absorption']:.6e} vs theirs {fs['X']:.6e} "
                 f"(ratio {fs['X'] / v['X_absorption']:.6f}, expected {C_RATIO:.6f})")
        results.append((gate, 'PASS' if ok else 'FAIL', f'{msg}; {x_msg}'))
    return results, fs_line, fs_cell


def run_gates(variants, spectra_path, data):
    print('\n' + '=' * 70)
    print('VALIDATION GATES')
    print('=' * 70)

    outcomes = []
    status, msg = gate_regression(variants, spectra_path)
    outcomes.append(('G1 regression', status, msg))

    fs_line = fs_cell = None
    fs_out = gate_fake_spectra(variants, spectra_path, data)
    if isinstance(fs_out, tuple) and len(fs_out) == 3:
        fs_results, fs_line, fs_cell = fs_out
    else:
        fs_results = fs_out
    outcomes.extend(fs_results)

    for gate, status, msg in outcomes:
        print(f'  [{status}] {gate}: {msg}')

    failed = [g for g, s, _ in outcomes if s in ('FAIL', 'ERROR')]
    return failed, fs_line, fs_cell


# ----------------------------------------------------------------------------
# output
# ----------------------------------------------------------------------------

def write_csvs(variants, fs_line, fs_cell, data, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    long_path = out_dir / 'cddf_variants.csv'
    with open(long_path, 'w') as fh:
        fh.write(f"# redshift = {data['redshift']:.6f}\n")
        fh.write(f"# n_sightlines = {data['n_sightlines']}\n")
        fh.write(f"# n_pixels = {data['n_pixels']}\n")
        fh.write('# f_N units: [Mpc^-1] for norm_mode=0, [cm^2] for norm_mode=1\n')
        fh.write('variant,absorber_mode,tau_threshold,colden_mode,dx_mode,norm_mode,'
                 'log10_N,f_N,counts\n')
        for name, v in variants.items():
            for i in range(v['f_N'].size):
                fh.write(f"{name},{v['absorber_mode']},{v.get('threshold', np.nan)},"
                         f"{v['colden_mode']},{v['dx_mode']},{v['norm_mode']},"
                         f"{np.log10(v['bin_centers'][i]):.6f},{v['f_N'][i]:.6e},"
                         f"{int(v['counts'][i])}\n")
        for name, fs in (('fake_spectra_line', fs_line), ('fake_spectra_cells50', fs_cell)):
            if fs is None:
                continue
            for i in range(fs['f_N'].size):
                fh.write(f'{name},,,1,1,1,{np.log10(fs["bin_centers"][i]):.6f},'
                         f'{fs["f_N"][i]:.6e},\n')
    print(f'  wrote {long_path}')

    summary_path = out_dir / 'cddf_variants_summary.csv'
    beta_cols = [f'beta_{lo:g}_{hi:g}' for lo, hi in FIT_RANGES]
    with open(summary_path, 'w') as fh:
        fh.write('variant,absorber_mode,tau_threshold,colden_mode,dx_mode,norm_mode,'
                 'n_absorbers,n_features_total,dX_used,dX_comoving_mpc,X_absorption,'
                 'median_feature_pixels,mean_feature_pixels,max_feature_pixels,'
                 'frac_pixels_absorbing,beta_cpp,' + ','.join(beta_cols) + '\n')
        for name, v in variants.items():
            betas = ','.join(f"{v['betas'][r][0]:.6f}" if np.isfinite(v['betas'][r][0])
                             else 'nan' for r in FIT_RANGES)
            fh.write(f"{name},{v['absorber_mode']},{v.get('threshold', '')},"
                     f"{v['colden_mode']},{v['dx_mode']},{v['norm_mode']},"
                     f"{v['n_absorbers']},{v['n_features_total']},{v['dX']:.6e},"
                     f"{v['dX_comoving_mpc']:.6e},{v['X_absorption']:.6e},"
                     f"{v['median_feature_pixels']:.1f},{v['mean_feature_pixels']:.2f},"
                     f"{v['max_feature_pixels']},{v['frac_pixels_absorbing']:.6e},"
                     f"{v['beta_fit']:.6f}," + betas + '\n')
    print(f'  wrote {summary_path}')


# ----------------------------------------------------------------------------
# plots
# ----------------------------------------------------------------------------

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


def _pos(v):
    """log-log plottable subset of one variant."""
    m = v['f_N'] > 0
    return v['bin_centers'][m], v['f_N'][m]


def plot_threshold_sweep(variants, fs_cell, out_path, z_label):
    fig, ax = plt.subplots(figsize=(7, 5))
    colours = plt.cm.viridis(np.linspace(0, 0.9, len(TAU_THRESHOLDS)))
    for th, c in zip(TAU_THRESHOLDS, colours):
        v = variants[f'feature_tau{th:g}_sum']
        x, y = _pos(v)
        ax.loglog(x, y, '-o', ms=3, color=c, label=rf'$\tau_{{\rm th}}={th:g}$')
    if fs_cell is not None:
        m = fs_cell['f_N'] > 0
        ax.loglog(fs_cell['bin_centers'][m], fs_cell['f_N'][m], 'k--', lw=2,
                  label='fake_spectra (50 km/s cells)')
    ax.set_xlabel(r'$N_{\rm HI}$ [cm$^{-2}$]')
    ax.set_ylabel(r'$f(N_{\rm HI})$ [cm$^2$]')
    ax.set_title(f'CDDF vs absorber threshold, {z_label}\n'
                 '(summed colden, absorption distance, per linear dN)')
    ax.legend(ncol=2)
    ax.grid(alpha=0.3, which='both')
    _save(fig, out_path)


def plot_beta_vs_threshold(variants, out_path, z_label):
    fig, ax = plt.subplots(figsize=(7, 5))
    for (lo, hi), marker in zip(FIT_RANGES, ('o', 's', '^', 'v')):
        betas = [variants[f'feature_tau{th:g}_sum']['betas'][(lo, hi)][0]
                 for th in TAU_THRESHOLDS]
        ax.plot(TAU_THRESHOLDS, betas, marker + '-',
                label=rf'fit $\log N \in [{lo:g}, {hi:g}]$')
    ax.axhspan(1.5, 1.7, color='grey', alpha=0.25,
               label=r'observed $\beta \approx 1.5-1.7$')
    ax.axvline(config.TAU_THRESHOLD_HI, color='r', ls=':',
               label=rf'production $\tau_{{\rm th}}={config.TAU_THRESHOLD_HI:g}$')
    ax.set_xscale('log')
    ax.set_xlabel(r'absorber threshold $\tau_{\rm th}$')
    ax.set_ylabel(r'fitted $\beta$')
    ax.set_title(f'Power-law slope vs threshold and fit floor, {z_label}')
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, out_path)


def plot_absorber_counts(variants, out_path, z_label):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    n_abs = [variants[f'feature_tau{th:g}_sum']['n_absorbers'] for th in TAU_THRESHOLDS]
    med_px = [variants[f'feature_tau{th:g}_sum']['median_feature_pixels']
              for th in TAU_THRESHOLDS]
    mean_px = [variants[f'feature_tau{th:g}_sum']['mean_feature_pixels']
               for th in TAU_THRESHOLDS]
    n_los = variants['production']['n_sightlines']

    ax1.plot(TAU_THRESHOLDS, n_abs, 'o-')
    ax1.axhline(n_los, color='k', ls='--', label='one per sightline (fully blended)')
    ax1.set_yscale('log')
    ax1.set_ylabel('absorbers found')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_title(f'Absorber census vs threshold, {z_label}')

    ax2.plot(TAU_THRESHOLDS, med_px, 'o-', label='median')
    ax2.plot(TAU_THRESHOLDS, mean_px, 's-', label='mean')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel(r'absorber threshold $\tau_{\rm th}$')
    ax2.set_ylabel('feature width [pixels]')
    ax2.legend()
    ax2.grid(alpha=0.3)
    for ax in (ax1, ax2):
        ax.axvline(config.TAU_THRESHOLD_HI, color='r', ls=':')
    _save(fig, out_path)


def plot_max_vs_sum(variants, out_path, z_label):
    th = config.TAU_THRESHOLD_HI
    v_sum = variants[f'feature_tau{th:g}_sum']
    v_max = variants[f'feature_tau{th:g}_max']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    for v, lab, style in ((v_sum, 'sum over feature (fake_spectra definition)', '-o'),
                          (v_max, 'max over feature (current)', '-s')):
        x, y = _pos(v)
        ax1.loglog(x, y, style, ms=3, label=lab)
    ax1.set_xlabel(r'$N_{\rm HI}$ [cm$^{-2}$]')
    ax1.set_ylabel(r'$f(N_{\rm HI})$ [cm$^2$]')
    ax1.set_title(rf'$\tau_{{\rm th}}={th:g}$: colden reduction')
    ax1.legend()
    ax1.grid(alpha=0.3, which='both')

    # Per-absorber paired comparison: N_HI_alt of the sum variant is the max.
    n_sum = v_sum['N_HI']
    n_max = v_sum.get('N_HI_alt')
    if n_max is not None and n_sum.size:
        good = (n_sum > 0) & (n_max > 0)
        shift = np.log10(n_sum[good]) - np.log10(n_max[good])
        ax2.hist(shift, bins=80, color='C2')
        ax2.axvline(np.median(shift), color='k', ls='--',
                    label=f'median {np.median(shift):.2f} dex')
        ax2.set_xlabel(r'$\log_{10}(N_{\rm sum}/N_{\rm max})$ per absorber')
        ax2.set_ylabel('absorbers')
        ax2.set_title('How much max underestimates')
        ax2.legend()
        ax2.grid(alpha=0.3)
    _save(fig, out_path)


def plot_three_way(variants, fs_line, fs_cell, out_path, z_label):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    v = variants['production']
    x, y = _pos(v)
    ax1.loglog(x, y, '-o', ms=3, color='C3',
               label=r'current C++ ($\tau>0.5$, max, Mpc, per dex)')
    ax1.set_xlabel(r'$N_{\rm HI}$ [cm$^{-2}$]')
    ax1.set_ylabel(r'$f$ [Mpc$^{-1}$ dex$^{-1}$]')
    ax1.set_title('current production units')
    ax1.legend()
    ax1.grid(alpha=0.3, which='both')

    th = config.TAU_THRESHOLD_HI
    for key, lab, style, col in (
            (f'feature_tau{th:g}_sum',
             rf'corrected C++ ($\tau>{th:g}$, sum, $X(z)$, per $dN$)', '-o', 'C0'),
            ('cpp_cells50', 'C++ 50 km/s cells', '-s', 'C1'),
            ('cpp_sightline', 'C++ whole sightline', '-^', 'C4')):
        x, y = _pos(variants[key])
        ax2.loglog(x, y, style, ms=3, color=col, label=lab)
    for fs, lab, col in ((fs_cell, 'fake_spectra line=False', 'k'),
                         (fs_line, 'fake_spectra line=True', 'grey')):
        if fs is None:
            continue
        m = fs['f_N'] > 0
        ax2.loglog(fs['bin_centers'][m], fs['f_N'][m], '--', lw=2, color=col, label=lab)
    ax2.set_xlabel(r'$N_{\rm HI}$ [cm$^{-2}$]')
    ax2.set_ylabel(r'$f(N_{\rm HI})$ [cm$^2$]')
    ax2.set_title('corrected normalisation, three absorber definitions')
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3, which='both')

    fig.suptitle(f'CDDF implementation comparison, {z_label}')
    _save(fig, out_path)


def plot_completeness(variants, out_path, z_label):
    """Where the threshold cuts, empirically: peak tau vs summed N of each feature."""
    v = variants['feature_tau0.05_sum']
    n_sum = v['N_HI']
    peak = v.get('peak_tau')
    if peak is None or n_sum.size == 0:
        return
    good = (n_sum > 0) & (peak > 0)
    if good.sum() == 0:
        return
    x = np.log10(n_sum[good])
    y = np.log10(peak[good])

    fig, ax = plt.subplots(figsize=(7, 5))
    hb = ax.hexbin(x, y, gridsize=70, bins='log', cmap='viridis', mincnt=1)
    fig.colorbar(hb, ax=ax, label='absorbers per cell')
    for th, col in ((0.5, 'r'), (1.0, 'orange')):
        ax.axhline(np.log10(th), color=col, ls='--',
                   label=rf'$\tau_{{\rm th}}={th:g}$')
    ax.set_xlabel(r'$\log_{10} N_{\rm HI}$ (summed over the feature)')
    ax.set_ylabel(r'$\log_{10}(\rm peak\ \tau)$')
    ax.set_title(f'Empirical completeness limit of the threshold, {z_label}\n'
                 r'(features found with $\tau_{\rm th}=0.05$)')
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, out_path)


def make_plots(variants, fs_line, fs_cell, plot_dir, z_label):
    _setup_style()
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_threshold_sweep(variants, fs_cell, plot_dir / 'cddf_threshold_sweep.png', z_label)
    plot_beta_vs_threshold(variants, plot_dir / 'cddf_beta_vs_threshold.png', z_label)
    plot_absorber_counts(variants, plot_dir / 'cddf_absorber_census.png', z_label)
    plot_max_vs_sum(variants, plot_dir / 'cddf_max_vs_sum.png', z_label)
    plot_three_way(variants, fs_line, fs_cell, plot_dir / 'cddf_three_way.png', z_label)
    plot_completeness(variants, plot_dir / 'cddf_completeness.png', z_label)


# ----------------------------------------------------------------------------

def print_summary(variants, data):
    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print(f"z = {data['redshift']:.4f}   n_sightlines = {data['n_sightlines']}   "
          f"n_pixels = {data['n_pixels']}")
    v0 = variants['production']
    print(f"dX (comoving Mpc, current) = {v0['dX_comoving_mpc']:.6f}    "
          f"X(z) (absorption distance) = {v0['X_absorption']:.6e}    "
          f"ratio = {v0['dX_comoving_mpc'] / v0['X_absorption']:.1f}")

    hdr = (f"{'variant':<26}{'n_abs':>10}{'med px':>8}" +
           ''.join(f'{f"b[{lo:g},{hi:g}]":>14}' for lo, hi in FIT_RANGES))
    print('\n' + hdr)
    print('-' * len(hdr))
    for name, v in variants.items():
        betas = ''.join(f"{v['betas'][r][0]:>14.3f}" if np.isfinite(v['betas'][r][0])
                        else f"{'nan':>14}" for r in FIT_RANGES)
        print(f"{name:<26}{v['n_absorbers']:>10}"
              f"{v['median_feature_pixels']:>8.0f}{betas}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--spectra', required=True, help='Path to a spectra HDF5 file')
    ap.add_argument('--out-dir', default=None,
                    help='Root for CSV output (default: output/cddf_test)')
    ap.add_argument('--plot-dir', default=None,
                    help='Root for plots (default: plots/cddf_test)')
    ap.add_argument('--skip-gates', action='store_true',
                    help='Report gate outcomes but do not exit non-zero on failure')
    args = ap.parse_args()

    spectra_path = Path(args.spectra)
    if not spectra_path.exists():
        print(f'Error: file not found: {spectra_path}')
        return 1

    info = config.extract_simulation_info(str(spectra_path))
    rel = Path(info['suite']) / info['sim_set'] / info['sim_name'] / f"snap-{info['snap_num']}"
    out_dir = Path(args.out_dir or (config.OUTPUT_DIR / 'cddf_test')) / rel
    plot_dir = Path(args.plot_dir or (config.PLOTS_DIR / 'cddf_test')) / rel

    print('=' * 70)
    print('CDDF THRESHOLD AND IMPLEMENTATION TEST')
    print('=' * 70)
    print(f'Spectra: {spectra_path}')
    print(f'Threads (OpenMP): {os.environ.get("OMP_NUM_THREADS", "unset")}')

    print('\nLoading tau and colden...')
    data = load_spectra(spectra_path)
    print(f"  z = {data['redshift']:.4f}, box = {data['box_size_ckpc_h']:g} ckpc/h, "
          f"h = {data['hubble']:g}, dvbin = {data['velocity_spacing']:.6f} km/s")
    print(f"  tau {data['tau'].shape} float32, colden {data['colden'].shape} float32")

    variants = build_variants(data)

    failed, fs_line, fs_cell = run_gates(variants, spectra_path, data)

    z_label = f"z = {data['redshift']:.2f} ({info['sim_name']}, snap {info['snap_num']})"
    print('\nWriting CSVs...')
    write_csvs(variants, fs_line, fs_cell, data, out_dir)
    print('\nPlotting...')
    make_plots(variants, fs_line, fs_cell, plot_dir, z_label)

    print_summary(variants, data)

    if failed and not args.skip_gates:
        print(f'\nFAILED GATES: {", ".join(failed)}')
        print('Results above are written but must not be trusted until these pass.')
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
