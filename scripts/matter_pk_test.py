"""
Dark-matter P(k) shape test for the Omega_0 -- sigma_8 degeneracy.

Mechanism being tested: sigma_8 rescales the linear matter power spectrum by a
constant factor (amplitude), so P_var(k)/P_fid(k) is flat in k. Omega_0 shifts
the matter-radiation equality scale k_eq ~ Omega_m h^2, which changes the SHAPE
of P(k): the ratio P_var(k)/P_fid(k) tilts and has a turnover near k_eq. The DM
field is the clean test of this -- no forest, thermal, or UVB confound (unlike
the flux-power scale split in degeneracy_test.py, which only separated the two
at z >~ 4).

Method, per (scan, variant, snapshot):
  1. Read PartType1 coordinates (equal-mass DM particles -> density = number
     count) and the box size from the snapshot header.
  2. CIC-deposit onto an Ngrid^3 mesh, form the overdensity delta = n/nbar - 1.
  3. FFT, |delta_k|^2 -> P(k); deconvolve the CIC window; subtract shot noise
     P_shot = V_box / N_part.
  4. Bin in |k| (log bins, k_f .. k_Nyquist).

Outputs, per snapshot:
  - P(k) for all variants, p1 and p2 in separate panels.
  - the shape diagnostic P_var(k)/P_fid(k): flat => sigma_8-like, tilted => Omega_0-like.
  - dimensionless Delta^2(k) = k^3 P(k) / (2 pi^2).
  - a tilt number: slope of ln[P_var/P_fid] vs ln k over a fixed band, per variant.

Units: coordinates and BoxSize are in ckpc/h; converted to cMpc/h (factor 1e-3),
so k is in h/Mpc.

Standalone (reads raw HDF5 under data/...), only the CosmoAstroSeed CSV is shared
with the other scripts.

Run:
    python scripts/matter_pk_test.py \\
        --data-root data/IllustrisTNG/1P \\
        --cosmo-csv data/IllustrisTNG/1P/CosmoAstroSeed_IllustrisTNG_L25n256_1P.csv \\
        --snaps 080,024 \\
        --ngrid 256 \\
        --out-dir plots/matter_pk_test
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

# Same scan registry the CSV-side tests use, so a set is defined in one place.
from hypothesis_test_p1 import SCANS, FIDUCIAL, members, load_cosmo_table

# module-level, rebound once by main() from --scans, matching
# degeneracy_test. One configuration per process, so a global is enough.
DEGEN_SCANS = ['p1', 'p2']

# Which particle types make up the field. Feedback moves baryons, so the
# total-matter field is where AGN suppression actually shows; DM alone still
# responds through backreaction, but several times more weakly.
FIELD_PARTS = {'dm': (1,), 'total': (0, 1, 4, 5)}
FIELD_LABEL = {'dm': 'DM', 'total': 'total matter'}


def _S8(omega0, sigma8):
    return sigma8 * np.sqrt(omega0 / 0.3)


# =====================================================================
# Power spectrum estimator
# =====================================================================

def cic_deposit(pos, ngrid, boxsize, weights=None, out=None):
    """Cloud-in-cell assignment. pos in same units as boxsize.

    weights=None deposits counts (equal-mass particles); otherwise it deposits
    mass. out lets several particle types accumulate onto one grid, so a
    multi-type field never holds all its coordinates in memory at once.
    """
    grid = np.zeros((ngrid, ngrid, ngrid), dtype=np.float64) if out is None else out
    x = (pos / boxsize) * ngrid                 # cell coordinates [0, ngrid)
    i = np.floor(x).astype(np.int64)
    d = x - i                                    # fractional offset in [0,1)
    i0 = i % ngrid
    i1 = (i + 1) % ngrid
    wx = [1.0 - d[:, 0], d[:, 0]]
    wy = [1.0 - d[:, 1], d[:, 1]]
    wz = [1.0 - d[:, 2], d[:, 2]]
    ix = [i0[:, 0], i1[:, 0]]
    iy = [i0[:, 1], i1[:, 1]]
    iz = [i0[:, 2], i1[:, 2]]
    gflat = grid.reshape(-1)
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                w = wx[a] * wy[b] * wz[c]
                if weights is not None:
                    w = w * weights
                flat = (ix[a] * ngrid + iy[b]) * ngrid + iz[c]
                gflat += np.bincount(flat, weights=w, minlength=gflat.size)
    return grid


def deposit_chunks(chunks, ngrid, boxsize_mpc):
    """CIC-deposit an iterable of (pos, weights) onto one grid.

    Returns (grid, wsum, w2sum). weights=None means equal-mass, so its
    contribution to both sums is just the particle count. The two sums are what
    the weighted shot-noise term needs.
    """
    if len(chunks) > 1 and any((w is None) for _, w in chunks):
        raise ValueError('a multi-chunk field must weight every chunk: mixing '
                         'unweighted counts with masses corrupts wsum')
    grid = np.zeros((ngrid, ngrid, ngrid), dtype=np.float64)
    wsum = w2sum = 0.0
    for pos, w in chunks:
        cic_deposit(pos, ngrid, boxsize_mpc, weights=w, out=grid)
        if w is None:
            wsum += pos.shape[0]
            w2sum += pos.shape[0]
        else:
            wsum += float(w.sum())
            w2sum += float((w.astype(np.float64) ** 2).sum())
    return grid, wsum, w2sum


def power_spectrum(pos, ngrid, boxsize_mpc, nkbins=40, weights=None):
    """Single-chunk convenience wrapper around the grid estimator."""
    grid, wsum, w2sum = deposit_chunks([(pos, weights)], ngrid, boxsize_mpc)
    return power_spectrum_from_grid(grid, wsum, w2sum, ngrid, boxsize_mpc, nkbins)


def power_spectrum_from_grid(grid, wsum, w2sum, ngrid, boxsize_mpc, nkbins=40):
    """Return (k_centers [h/Mpc], P(k) [(Mpc/h)^3], n_modes) from a deposited grid.

    CIC deconvolution and shot-noise subtraction applied.
    """
    nbar = wsum / ngrid ** 3
    delta = grid / nbar - 1.0

    dk = np.fft.rfftn(delta)
    vol = boxsize_mpc ** 3
    # |delta_k|^2 normalized to a power spectrum estimate
    pk3d = (np.abs(dk) ** 2) * vol / (ngrid ** 6)

    kf = 2.0 * np.pi / boxsize_mpc
    kx = np.fft.fftfreq(ngrid, d=1.0 / ngrid) * kf
    ky = kx
    kz = np.fft.rfftfreq(ngrid, d=1.0 / ngrid) * kf
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    kmag = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)

    # CIC window deconvolution. The CIC assignment window is
    # W_CIC = prod_i sinc^2(pi k_i / (2 k_Ny)) (one sinc per dimension is NGP;
    # CIC is its square), so P must be divided by W_CIC^2 = prod sinc^4.
    # Note this factor cancels exactly in the P_var/P_fid ratios (same grid,
    # same k), so the ratio-based tilt results are unaffected by the choice.
    kny = np.pi * ngrid / boxsize_mpc
    def sinc(t):
        return np.sinc(t / np.pi)            # np.sinc(x) = sin(pi x)/(pi x)
    Wx = sinc(np.pi * KX / (2.0 * kny))
    Wy = sinc(np.pi * KY / (2.0 * kny))
    Wz = sinc(np.pi * KZ / (2.0 * kny))
    W = Wx * Wy * Wz
    W[W == 0] = 1.0
    pk3d /= W ** 4                             # deconvolve: P_obs = W_CIC^2 P_true

    # Shot noise for a weighted field: V * sum(w^2) / (sum w)^2, which reduces
    # to the equal-mass V / N when every weight is the same.
    p_shot = vol * w2sum / wsum ** 2
    pk3d -= p_shot

    # radial binning (log) from k_f to k_Ny
    kmin = kf
    kmax = kny
    bins = np.logspace(np.log10(kmin), np.log10(kmax), nkbins + 1)
    kflat = kmag.ravel()
    pflat = pk3d.ravel()
    good = kflat > 0
    kflat, pflat = kflat[good], pflat[good]
    which = np.digitize(kflat, bins)
    kc, Pk, nm = [], [], []
    for b in range(1, nkbins + 1):
        m = which == b
        if m.sum() == 0:
            continue
        kc.append(kflat[m].mean())
        Pk.append(pflat[m].mean())
        nm.append(int(m.sum()))
    return np.array(kc), np.array(Pk), np.array(nm)


# =====================================================================
# I/O
# =====================================================================

def read_field(snapshot, parts):
    """Return (chunks, box_Mpc_per_h, redshift, counts).

    chunks is a list of (coords, masses) -- one entry per particle type, kept
    separate so the deposit can consume them one at a time. masses is None only
    when the whole field is one equal-mass type (DM alone, whose mass lives in
    the header MassTable): a multi-type field always carries real masses, so
    the weight sums never mix particle counts with masses. counts is
    {ptype: n} for the run log, the cheapest way to catch a field that
    silently lost a component.
    """
    chunks, counts = [], {}
    with h5py.File(snapshot, 'r') as f:
        box = float(f['Header'].attrs['BoxSize']) * 1e-3     # ckpc/h -> cMpc/h
        z = float(f['Header'].attrs['Redshift'])
        mtable = np.asarray(f['Header'].attrs['MassTable'], dtype=np.float64)
        for i in parts:
            key = f'PartType{i}'
            if key not in f or 'Coordinates' not in f[key]:
                continue
            pos = f[key]['Coordinates'][:].astype(np.float64) * 1e-3
            if pos.shape[0] == 0:
                continue
            pos = np.mod(pos, box)               # wrap into [0, box)
            if mtable[i] > 0:
                mass = None if len(parts) == 1 else np.full(pos.shape[0], mtable[i])
            else:
                mass = f[key]['Masses'][:].astype(np.float64)
            chunks.append((pos, mass))
            counts[i] = pos.shape[0]
    return chunks, box, z, counts


# =====================================================================
# Driver per snapshot
# =====================================================================

def compute_snapshot(data_root, cosmo, scan, snapnum, ngrid, nkbins, field='dm'):
    """Return dict suffix -> {k, P, omega0, sigma8, S8, z} for one field."""
    parts = FIELD_PARTS[field]
    out = {}
    for suf in members(scan):
        label = SCANS[scan]['name_fmt'].format(s=suf)
        snap = data_root / label / f'snap_{snapnum}.hdf5'
        if not snap.exists():
            print(f'  [skip] {snap} missing')
            continue
        chunks, box, z, counts = read_field(snap, parts)
        if not chunks:
            print(f'  [skip] {snap}: no particles for field {field}')
            continue
        grid, wsum, w2sum = deposit_chunks(chunks, ngrid, box)
        del chunks
        k, P, nm = power_spectrum_from_grid(grid, wsum, w2sum, ngrid, box,
                                            nkbins=nkbins)
        del grid
        om = float(cosmo.loc[label, 'Omega0']) if label in cosmo.index else np.nan
        s8 = float(cosmo.loc[label, 'sigma8']) if label in cosmo.index else np.nan
        col = SCANS[scan]['column']
        pv = (float(cosmo.loc[label, col])
              if col is not None and label in cosmo.index else np.nan)
        out[suf] = {'k': k, 'P': P, 'nmodes': nm, 'label': label,
                    'omega0': om, 'sigma8': s8, 'S8': _S8(om, s8), 'z': z,
                    'param': pv, 'counts': counts}
        npart = ', '.join(f'PartType{i}={n}' for i, n in sorted(counts.items()))
        print(f'  {label} snap_{snapnum} [{field}]: z={z:.3f}, nk={len(k)}, '
              f'Omega0={om}, sigma8={s8}, {npart}')
    return out


def _variant_label(scan, suf, r):
    """Legend text: the varied parameter where there is one, the sim name where
    there is not (EX varies feedback, which is not a single number)."""
    if SCANS[scan]['column'] is None:
        return r.get('label', suf)
    return f"{suf} ({r['param']:.2f})"


def tilt_slope(k, ratio, klo=0.3, khi=5.0):
    """Slope of ln(ratio) vs ln(k) over [klo, khi] h/Mpc. ~0 => amplitude only
    (sigma_8); nonzero => shape change (Omega_0)."""
    m = (k >= klo) & (k <= khi) & np.isfinite(ratio) & (ratio > 0)
    if m.sum() < 2:
        return np.nan
    return float(np.polyfit(np.log(k[m]), np.log(ratio[m]), 1)[0])


def plot_snapshot(results, snapnum, out_dir, field='dm'):
    """results: scan -> {suffix -> data}. Make P(k) and shape-ratio figures."""
    flabel = FIELD_LABEL[field]
    # --- P(k) panels, one per scan ---
    fig, axes = plt.subplots(1, len(DEGEN_SCANS),
                             figsize=(6.5 * len(DEGEN_SCANS), 5.2), squeeze=False)
    for ax, scan in zip(axes[0], DEGEN_SCANS):
        d = results[scan]
        mem = members(scan)
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(mem)))
        for suf, c in zip(mem, colors):
            if suf not in d:
                continue
            r = d[suf]
            m = r['P'] > 0
            ax.loglog(r['k'][m], r['P'][m], '-', color=c, lw=1.6,
                      label=_variant_label(scan, suf, r))
        ax.set_xlabel(r'$k$ [$h$/Mpc]')
        ax.set_ylabel(r'$P(k)$ [(Mpc/$h$)$^3$]')
        ax.set_title(f'{flabel} $P(k)$ -- {scan} ({SCANS[scan]["label"]})')
        ax.grid(alpha=0.3, which='both')
        ax.legend(fontsize=8, title='variant')
    z = _results_z(results)
    fig.suptitle(f'snap_{snapnum}  (z={z:.2f}) -- {flabel}')
    fig.tight_layout()
    _save(fig, out_dir / f'Pk_snap_{snapnum}.png')

    # --- shape diagnostic: P_var / P_fid ---
    fig, axes = plt.subplots(1, len(DEGEN_SCANS),
                             figsize=(6.5 * len(DEGEN_SCANS), 5.2), squeeze=False)
    tilts = {}
    for ax, scan in zip(axes[0], DEGEN_SCANS):
        d = results[scan]
        if FIDUCIAL not in d:
            continue
        kf_, Pf = d[FIDUCIAL]['k'], d[FIDUCIAL]['P']
        mem = members(scan)
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(mem)))
        for suf, c in zip(mem, colors):
            if suf not in d:
                continue
            r = d[suf]
            # interpolate fiducial onto this k grid (same grid anyway)
            ratio = r['P'] / np.interp(r['k'], kf_, Pf)
            ax.semilogx(r['k'], ratio, '-', color=c, lw=1.6,
                        label=_variant_label(scan, suf, r))
            tilts.setdefault(scan, {})[suf] = tilt_slope(r['k'], ratio)
        ax.axhline(1.0, color='gray', lw=0.8, ls=':')
        ax.set_xlabel(r'$k$ [$h$/Mpc]')
        ax.set_ylabel(r'$P_{\rm var}(k)/P_{\rm fid}(k)$')
        sub = ', '.join(f"{s}={tilts.get(scan,{}).get(s,np.nan):+.2f}"
                        for s in tilts.get(scan, {}))
        ax.set_title(f'shape ratio -- {scan} ({SCANS[scan]["label"]})\n'
                     f'tilt slope: {sub}', fontsize=9)
        ax.grid(alpha=0.3, which='both')
        ax.legend(fontsize=8, title='variant')
    fig.suptitle(f'Shape diagnostic snap_{snapnum} (z={z:.2f}, {flabel}): '
                 f'flat=$\\sigma_8$-like, tilted=$\\Omega_0$-like')
    fig.tight_layout()
    _save(fig, out_dir / f'Pk_ratio_snap_{snapnum}.png')

    return tilts


def _results_z(results):
    for d in results.values():
        for r in d.values():
            return r['z']
    return np.nan


def plot_field_ratio(res_total, res_dm, snapnum, out_dir):
    """P_total(k) / P_DM(k) per variant. Baryons trace the same large-scale
    structure, so this sits above 1 at low k; feedback pushes gas out of haloes,
    so it drops below 1 at high k, and more so the stronger the feedback."""
    fig, axes = plt.subplots(1, len(DEGEN_SCANS),
                             figsize=(6.5 * len(DEGEN_SCANS), 5.2), squeeze=False)
    out = {}
    for ax, scan in zip(axes[0], DEGEN_SCANS):
        mem = members(scan)
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(mem)))
        for suf, c in zip(mem, colors):
            if suf not in res_total.get(scan, {}) or suf not in res_dm.get(scan, {}):
                continue
            rt, rd = res_total[scan][suf], res_dm[scan][suf]
            ratio = rt['P'] / np.interp(rt['k'], rd['k'], rd['P'])
            ax.semilogx(rt['k'], ratio, '-', color=c, lw=1.6,
                        label=_variant_label(scan, suf, rt))
            out.setdefault(scan, {})[suf] = tilt_slope(rt['k'], ratio)
        ax.axhline(1.0, color='gray', lw=0.8, ls=':')
        ax.set_xlabel(r'$k$ [$h$/Mpc]')
        ax.set_ylabel(r'$P_{\rm total}(k)/P_{\rm DM}(k)$')
        ax.set_title(f'baryonic backreaction -- {scan} ({SCANS[scan]["label"]})')
        ax.grid(alpha=0.3, which='both')
        ax.legend(fontsize=8, title='variant')
    fig.suptitle(f'snap_{snapnum}: total matter vs DM-only')
    fig.tight_layout()
    _save(fig, out_dir / f'Pk_total_over_dm_snap_{snapnum}.png')
    return out


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


# =====================================================================
# Entry
# =====================================================================

def self_test():
    """The weighted path is the only non-trivial new logic here: check it
    reduces to the equal-mass estimator it replaced."""
    rng = np.random.default_rng(0)
    ngrid, box, n = 32, 100.0, 5000
    pos = rng.random((n, 3)) * box

    k0, P0, nm0 = power_spectrum(pos, ngrid, box, nkbins=12)
    k1, P1, nm1 = power_spectrum(pos, ngrid, box, nkbins=12,
                                 weights=np.full(n, 3.7))
    assert np.allclose(k0, k1) and np.array_equal(nm0, nm1)
    # equal weights of any size must give the same P(k): the weight cancels in
    # delta = rho/rhobar - 1 and in V*sum(w^2)/sum(w)^2
    # P(k) crosses zero once shot noise is subtracted, so compare against the
    # spectrum's own scale rather than each point's magnitude
    assert np.allclose(P0, P1, rtol=1e-8, atol=1e-8 * np.abs(P0).max()), \
        np.abs(P0 - P1).max()

    # the weighted shot-noise term must reduce to V/N in that limit
    _, wsum, w2sum = deposit_chunks([(pos, np.full(n, 3.7))], ngrid, box)
    assert np.isclose(box ** 3 * w2sum / wsum ** 2, box ** 3 / n, rtol=1e-12)
    # splitting one population into chunks must not change the sums or the grid
    g1, ws, w2s = deposit_chunks([(pos, np.ones(n))], ngrid, box)
    g2, ws2, w2s2 = deposit_chunks([(pos[:1000], np.ones(1000)),
                                    (pos[1000:], np.ones(n - 1000))], ngrid, box)
    assert (ws, w2s) == (ws2, w2s2) == (float(n), float(n))
    assert np.allclose(g1, g2)

    # mixing an unweighted chunk into a multi-type field would add counts to a
    # mass sum, so it must be refused rather than silently mis-normalized
    try:
        deposit_chunks([(pos, None), (pos, np.ones(n))], ngrid, box)
    except ValueError:
        pass
    else:
        raise AssertionError('mixed weighted/unweighted chunks must raise')

    # a field with no members must not be silently treated as populated
    assert FIELD_PARTS['total'] == (0, 1, 4, 5) and FIELD_PARTS['dm'] == (1,)
    print('self-test OK')
    return 0


def main():
    global DEGEN_SCANS
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=Path,
                    help='dir holding <run label>/snap_NNN.hdf5, e.g. '
                         'data/IllustrisTNG/1P or data/IllustrisTNG/EX')
    ap.add_argument('--cosmo-csv', type=Path, nargs='+',
                    help='one or more CosmoAstroSeed tables')
    ap.add_argument('--snaps', default='080',
                    help='comma-separated snapshot numbers, e.g. 080,024')
    ap.add_argument('--ngrid', type=int, default=256)
    ap.add_argument('--nkbins', type=int, default=40)
    ap.add_argument('--out-dir', type=Path, default=Path('plots/matter_pk_test'))
    ap.add_argument('--scans', default=','.join(DEGEN_SCANS))
    ap.add_argument('--field', default='dm', choices=['dm', 'total', 'both'],
                    help="'dm' reproduces the original PartType1-only run; "
                         "'total' is gas+DM+stars+BH, where feedback actually "
                         "moves the power")
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if args.data_root is None or args.cosmo_csv is None:
        ap.error('--data-root and --cosmo-csv are required unless --self-test')

    scans = [s.strip() for s in args.scans.split(',') if s.strip()]
    unknown = [s for s in scans if s not in SCANS]
    if unknown:
        ap.error(f'unknown scan(s) {unknown}; known: {list(SCANS)}')
    DEGEN_SCANS = scans
    fields = ['dm', 'total'] if args.field == 'both' else [args.field]

    plt.rcParams['figure.dpi'] = 150
    cosmo = pd.concat([load_cosmo_table(c) for c in args.cosmo_csv])
    snaps = [s.strip() for s in args.snaps.split(',') if s.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for snapnum in snaps:
        per_field = {}
        for field in fields:
            print(f'\n=== matter P(k), snap_{snapnum}, field={field} ===')
            results = {}
            for scan in scans:
                print(f' scan {scan}')
                results[scan] = compute_snapshot(
                    args.data_root, cosmo, scan, snapnum, args.ngrid,
                    args.nkbins, field=field)
            if not any(results[s] for s in scans):
                print('  no data for this snap, skipping plots')
                continue
            fdir = args.out_dir / field if len(fields) > 1 else args.out_dir
            per_field[field] = {
                'results': results,
                'tilts': plot_snapshot(results, snapnum, fdir, field=field),
            }
        if not per_field:
            continue
        entry = {'z': _results_z(next(iter(per_field.values()))['results']),
                 'tilt_slopes': {f: v['tilts'] for f, v in per_field.items()}}
        if 'total' in per_field and 'dm' in per_field:
            entry['total_over_dm_tilt'] = plot_field_ratio(
                per_field['total']['results'], per_field['dm']['results'],
                snapnum, args.out_dir)
        summary[snapnum] = entry

    with open(args.out_dir / 'summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f'\nAll outputs under {args.out_dir.resolve()}')


if __name__ == '__main__':
    sys.exit(main())
