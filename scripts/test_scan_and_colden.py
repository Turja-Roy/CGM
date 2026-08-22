"""Self-checks for two things that fail silently.

1. E(z) and dX/dz must use the real Omega_0, not the scanned parameter. They
   coincide on p1; on p2 using param_value would inject a fake 1/H(z) trend
   into the sigma_8 scan -- the artefact the decomposition exists to rule out.
2. An absorber's column density is the SUM over its pixels (colden_mode = 1).

Run from the repo root: python scripts/test_scan_and_colden.py
Needs no spectra and no analysis output.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analysis import (  # noqa: E402
    TAU_TO_COLDEN_CONSTANT,
    compute_metal_line_statistics,
)
from scripts.hypothesis_test_p1 import (  # noqa: E402
    build_scan_frame,
    load_cosmo_table,
)

COSMO_CSV = Path('data/IllustrisTNG/1P/CosmoAstroSeed_IllustrisTNG_L25n256_1P.csv')


def test_omega0_is_not_the_scanned_parameter():
    cosmo = load_cosmo_table(COSMO_CSV)
    root = Path('output/analysis/IllustrisTNG/1P')

    p1 = build_scan_frame(root, cosmo, 'p1', 'snap-044')
    p2 = build_scan_frame(root, cosmo, 'p2', 'snap-044')

    # p1 scans Omega_0 itself, so the two agree and both vary.
    om_p1 = [r['omega0'] for r in p1]
    assert om_p1 == [r['param_value'] for r in p1], om_p1
    assert len(set(om_p1)) == len(om_p1), f'p1 Omega_0 should vary: {om_p1}'

    # p2 scans sigma_8 with Omega_0 pinned at the fiducial.
    om_p2 = [r['omega0'] for r in p2]
    s8_p2 = [r['param_value'] for r in p2]
    assert len(set(om_p2)) == 1, f'p2 Omega_0 should be constant: {om_p2}'
    assert len(set(s8_p2)) == len(s8_p2), f'p2 sigma_8 should vary: {s8_p2}'
    assert om_p2[0] != s8_p2[0], 'p2 would silently reuse sigma_8 as Omega_0'

    assert p1[0]['scan'] == 'p1' and p2[0]['scan'] == 'p2'
    assert p1[0]['param_label'] != p2[0]['param_label']


def test_absorber_colden_is_a_sum():
    # One sightline, one three-pixel absorber, flanked by transparent pixels.
    tau = np.array([[0.0, 1.0, 2.0, 3.0, 0.0]])
    colden = np.array([[0.0, 1e12, 2e12, 3e12, 0.0]])

    stats = compute_metal_line_statistics(tau, velocity_spacing=10.0,
                                          threshold=0.5, colden=colden)
    N = stats['column_densities']
    assert len(N) == 1, N
    assert np.isclose(N[0], 6e12), f'expected the sum 6e12, got {N[0]}'

    # Fallback path (no colden): tau-integral with the C++ constant, not 1e13.
    stats_fb = compute_metal_line_statistics(tau, velocity_spacing=10.0,
                                             threshold=0.5, colden=None)
    expected = TAU_TO_COLDEN_CONSTANT * 6.0 * 10.0
    assert np.isclose(stats_fb['column_densities'][0], expected), \
        stats_fb['column_densities']


def test_absorber_running_to_the_edge_is_counted():
    tau = np.array([[0.0, 1.0, 2.0]])
    colden = np.array([[0.0, 1e12, 4e12]])
    stats = compute_metal_line_statistics(tau, velocity_spacing=10.0,
                                          threshold=0.5, colden=colden)
    assert np.isclose(stats['column_densities'][0], 5e12), \
        stats['column_densities']


if __name__ == '__main__':
    test_omega0_is_not_the_scanned_parameter()
    test_absorber_colden_is_a_sum()
    test_absorber_running_to_the_edge_is_counted()
    print('all checks passed')
