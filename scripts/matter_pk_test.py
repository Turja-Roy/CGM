"""
Dark-matter P(k) shape test for the Omega_0 -- sigma_8 degeneracy.

Mechanism being tested: sigma_8 rescales the linear matter power spectrum by a
constant factor (amplitude), so P_var(k)/P_fid(k) is flat in k. Omega_0 shifts
the matter-radiation equality scale k_eq ~ Omega_m h^2, which changes the SHAPE
of P(k): the ratio P_var(k)/P_fid(k) tilts and has a turnover near k_eq. The DM
field is the clean test of this -- no forest, thermal, or UVB confound (unlike
the flux-power D3 diagnostic in degeneracy_test.py, which only separated the two
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
import h5py
import matplotlib.pyplot as plt

VARIANT_SUFFIXES = ['n2', 'n1', '0', '1', '2']
FIDUCIAL = '0'
DEGEN_SCANS = ['p1', 'p2']
SCAN_LABEL  = {'p1': r'$\Omega_0$', 'p2': r'$\sigma_8$'}
SCAN_COLOR  = {'p1': 'C0', 'p2': 'C3'}


def _S8(omega0, sigma8):
    return sigma8 * np.sqrt(omega0 / 0.3)


# =====================================================================
# Power spectrum estimator
# =====================================================================

def cic_deposit(pos, ngrid, boxsize):
    """Cloud-in-cell mass assignment. pos in same units as boxsize.
    Returns number-density grid (counts per cell)."""
    grid = np.zeros((ngrid, ngrid, ngrid), dtype=np.float64)
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
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                w = wx[a] * wy[b] * wz[c]
                np.add.at(grid, (ix[a], iy[b], iz[c]), w)
    return grid


def power_spectrum(pos, ngrid, boxsize_mpc, nkbins=40):
    """Return (k_centers [h/Mpc], P(k) [(Mpc/h)^3], n_modes) for a particle set.

    CIC deconvolution and shot-noise subtraction applied.
    """
    npart = pos.shape[0]
    grid = cic_deposit(pos, ngrid, boxsize_mpc)
    nbar = npart / ngrid ** 3
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

    # shot noise
    p_shot = vol / npart
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

def read_dm(snapshot):
    """Return (coords_Mpc_per_h, boxsize_Mpc_per_h, redshift)."""
    with h5py.File(snapshot, 'r') as f:
        box = f['Header'].attrs['BoxSize'] * 1e-3            # ckpc/h -> cMpc/h
        z = float(f['Header'].attrs['Redshift'])
        pos = f['PartType1/Coordinates'][:].astype(np.float64) * 1e-3
    # wrap into [0, box)
    pos = np.mod(pos, box)
    return pos, float(box), z


def load_cosmo(cosmo_csv):
    import pandas as pd
    df = pd.read_csv(cosmo_csv).set_index('Name')
    return df


# =====================================================================
# Driver per snapshot
# =====================================================================

def compute_snapshot(data_root, cosmo, scan, snapnum, ngrid, nkbins):
    """Return dict suffix -> {k, P, omega0, sigma8, S8, z}."""
    out = {}
    for suf in VARIANT_SUFFIXES:
        label = f'1P_{scan}_{suf}'
        snap = data_root / label / f'snap_{snapnum}.hdf5'
        if not snap.exists():
            print(f'  [skip] {snap} missing')
            continue
        pos, box, z = read_dm(snap)
        k, P, nm = power_spectrum(pos, ngrid, box, nkbins=nkbins)
        om = float(cosmo.loc[label, 'Omega0']) if label in cosmo.index else np.nan
        s8 = float(cosmo.loc[label, 'sigma8']) if label in cosmo.index else np.nan
        out[suf] = {'k': k, 'P': P, 'nmodes': nm,
                    'omega0': om, 'sigma8': s8, 'S8': _S8(om, s8), 'z': z}
        print(f'  {label} snap_{snapnum}: z={z:.3f}, nk={len(k)}, '
              f'Omega0={om}, sigma8={s8}')
    return out


def tilt_slope(k, ratio, klo=0.3, khi=5.0):
    """Slope of ln(ratio) vs ln(k) over [klo, khi] h/Mpc. ~0 => amplitude only
    (sigma_8); nonzero => shape change (Omega_0)."""
    m = (k >= klo) & (k <= khi) & np.isfinite(ratio) & (ratio > 0)
    if m.sum() < 2:
        return np.nan
    return float(np.polyfit(np.log(k[m]), np.log(ratio[m]), 1)[0])


def plot_snapshot(results, snapnum, out_dir):
    """results: scan -> {suffix -> data}. Make P(k), ratio, Delta^2 figures."""
    # --- P(k) panels, one per scan ---
    fig, axes = plt.subplots(1, len(DEGEN_SCANS),
                             figsize=(6.5 * len(DEGEN_SCANS), 5.2), squeeze=False)
    for ax, scan in zip(axes[0], DEGEN_SCANS):
        d = results[scan]
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(VARIANT_SUFFIXES)))
        for suf, c in zip(VARIANT_SUFFIXES, colors):
            if suf not in d:
                continue
            r = d[suf]
            m = r['P'] > 0
            ax.loglog(r['k'][m], r['P'][m], '-', color=c, lw=1.6,
                      label=f"{suf} ({r['omega0'] if scan=='p1' else r['sigma8']:.2f})")
        ax.set_xlabel(r'$k$ [$h$/Mpc]')
        ax.set_ylabel(r'$P(k)$ [(Mpc/$h$)$^3$]')
        ax.set_title(f'DM $P(k)$ -- {scan} ({SCAN_LABEL[scan]})')
        ax.grid(alpha=0.3, which='both')
        ax.legend(fontsize=8, title='variant')
    z = next(iter(results['p1'].values()))['z']
    fig.suptitle(f'snap_{snapnum}  (z={z:.2f})')
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
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(VARIANT_SUFFIXES)))
        for suf, c in zip(VARIANT_SUFFIXES, colors):
            if suf not in d:
                continue
            r = d[suf]
            # interpolate fiducial onto this k grid (same grid anyway)
            ratio = r['P'] / np.interp(r['k'], kf_, Pf)
            ax.semilogx(r['k'], ratio, '-', color=c, lw=1.6,
                        label=f"{suf}")
            tilts.setdefault(scan, {})[suf] = tilt_slope(r['k'], ratio)
        ax.axhline(1.0, color='gray', lw=0.8, ls=':')
        ax.set_xlabel(r'$k$ [$h$/Mpc]')
        ax.set_ylabel(r'$P_{\rm var}(k)/P_{\rm fid}(k)$')
        sub = ', '.join(f"{s}={tilts.get(scan,{}).get(s,np.nan):+.2f}"
                        for s in tilts.get(scan, {}))
        ax.set_title(f'shape ratio -- {scan} ({SCAN_LABEL[scan]})\n'
                     f'tilt slope: {sub}', fontsize=9)
        ax.grid(alpha=0.3, which='both')
        ax.legend(fontsize=8, title='variant')
    fig.suptitle(f'Shape diagnostic snap_{snapnum} (z={z:.2f}): '
                 f'flat=$\\sigma_8$-like, tilted=$\\Omega_0$-like')
    fig.tight_layout()
    _save(fig, out_dir / f'Pk_ratio_snap_{snapnum}.png')

    return tilts


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


# =====================================================================
# Entry
# =====================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True, type=Path,
                    help='dir holding 1P_pX_{suffix}/snap_NNN.hdf5')
    ap.add_argument('--cosmo-csv', required=True, type=Path)
    ap.add_argument('--snaps', default='080',
                    help='comma-separated snapshot numbers, e.g. 080,024')
    ap.add_argument('--ngrid', type=int, default=256)
    ap.add_argument('--nkbins', type=int, default=40)
    ap.add_argument('--out-dir', type=Path, default=Path('plots/matter_pk_test'))
    ap.add_argument('--scans', default='p1,p2')
    args = ap.parse_args()

    plt.rcParams['figure.dpi'] = 150
    cosmo = load_cosmo(args.cosmo_csv)
    snaps = [s.strip() for s in args.snaps.split(',') if s.strip()]
    scans = [s.strip() for s in args.scans.split(',') if s.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for snapnum in snaps:
        print(f'\n=== matter P(k), snap_{snapnum} ===')
        results = {}
        for scan in scans:
            print(f' scan {scan}')
            results[scan] = compute_snapshot(
                args.data_root, cosmo, scan, snapnum, args.ngrid, args.nkbins)
        if not any(results[s] for s in scans):
            print('  no data for this snap, skipping plots')
            continue
        tilts = plot_snapshot(results, snapnum, args.out_dir)
        summary[snapnum] = {
            'z': next(iter(next(v for v in results.values() if v).values()))['z'],
            'tilt_slopes': tilts,
        }

    with open(args.out_dir / 'summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f'\nAll outputs under {args.out_dir.resolve()}')


if __name__ == '__main__':
    sys.exit(main())
