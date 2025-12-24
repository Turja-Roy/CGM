"""
CAMEL Lyman-alpha Spectra Analysis Pipeline - Core modules.

This package contains all the core functionality for analyzing absorption
spectra from CAMEL simulations.

Modules:
    config            - Configuration and constants
    hdf5_io          - HDF5 file I/O operations
    fake_spectra_fix - Python 3.13 compatibility fixes for fake_spectra
    analysis         - Statistical analysis functions
    plotting         - Plotting and visualization
    comparison       - Multi-simulation comparison and evolution tracking
    commands         - CLI command implementations
"""

# Convenience imports for commonly used functions
from .config import *
from .hdf5_io import load_snapshot_metadata, explore_hdf5_structure
from .fake_spectra_fix import apply_fake_spectra_bugfixes
from .analysis import (
    compute_flux_statistics,
    compute_power_spectrum,
    compute_column_density_distribution,
    compute_effective_optical_depth,
    compute_line_width_distribution,
    compute_temperature_density_relation,
)
from .plotting import (
    setup_plot_style,
    create_sample_spectra_plot,
    plot_flux_power_spectrum,
    plot_column_density_distribution,
)
from .comparison import (
    load_spectra_results,
    compare_simulations,
    track_redshift_evolution,
)
