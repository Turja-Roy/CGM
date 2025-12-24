import os
from pathlib import Path

# Project root directory (go up one level from scripts/)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
PLOTS_DIR = PROJECT_ROOT / "plots"
OUTPUT_DIR = PROJECT_ROOT / "output"
ARCHIVE_DIR = PROJECT_ROOT / "archive"

# Ensure output directories exist
PLOTS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# CAMEL simulation suites available
CAMEL_SUITES = {
    'IllustrisTNG': DATA_DIR / "IllustrisTNG",
    'Simba': DATA_DIR / "Simba",
    'Astrid': DATA_DIR / "Astrid",
}


def get_available_datasets(suite='IllustrisTNG'):
    datasets = {}

    if suite not in CAMEL_SUITES:
        return datasets

    suite_path = CAMEL_SUITES[suite]

    if not suite_path.exists():
        return datasets

    # Look for set directories (e.g., LH, 1P, CV, etc.)
    for set_dir in suite_path.iterdir():
        if not set_dir.is_dir():
            continue

        sim_set = set_dir.name

        # Look inside each set directory for simulation directories (e.g., LH_0, LH_1, etc.)
        for sim_dir in set_dir.iterdir():
            if not sim_dir.is_dir():
                continue

            dir_name = sim_dir.name

            # Look for directories matching pattern: SET_NUM (e.g., LH_0, LH_1, 1P_0)
            if '_' in dir_name and dir_name.startswith(sim_set + '_'):
                parts = dir_name.split('_')
                if len(parts) == 2:
                    _, sim_num = parts

                    # Verify this directory contains simulation data
                    # Check for snapshots, spectra files, or SPECTRA subdirectories
                    has_data = (list(sim_dir.glob("snap_*.hdf5")) or
                                list(sim_dir.glob("camel_*_spectra_snap_*.hdf5")) or
                                list(sim_dir.glob("SPECTRA_*")))

                    if has_data:
                        if sim_set not in datasets:
                            datasets[sim_set] = []
                        datasets[sim_set].append(sim_num)

    # Sort the simulation numbers for each set
    for sim_set in datasets:
        datasets[sim_set] = sorted(
            datasets[sim_set], key=lambda x: int(x) if x.isdigit() else x)

    return datasets


def get_available_snapshots(suite='IllustrisTNG', sim_set='LH', sim_num='0'):
    snapshots = {}

    if suite not in CAMEL_SUITES:
        return {}

    sim_name = sim_set + '_' + sim_num
    suite_path = CAMEL_SUITES[suite] / sim_set / sim_name

    if not suite_path.exists():
        return snapshots

    # Look for snapshot files in simulation directory
    for snap_file in suite_path.glob("snap_*.hdf5"):
        snap_num = int(snap_file.stem.split('_')[-1])
        snapshots[snap_num] = snap_file

    # Look for spectra files directly in simulation directory
    for spectra_file in suite_path.glob("camel_*_spectra_snap_*.hdf5"):
        # Extract snapshot number from filename
        parts = spectra_file.stem.split('_')
        if 'snap' in parts:
            snap_idx = parts.index('snap')
            if snap_idx + 1 < len(parts):
                try:
                    # Get the part after 'snap', which should be the number
                    snap_part = parts[snap_idx + 1]
                    # Handle case where there might be additional suffixes like 'n100'
                    snap_num = int(snap_part)
                    # Store as spectra file (distinguish from snapshot)
                    key = f"spectra_{snap_num}"
                    snapshots[key] = spectra_file
                except ValueError:
                    continue

    # Also check SPECTRA_* subdirectories for generated spectra
    for spectra_dir in suite_path.glob("SPECTRA_*"):
        for spectra_file in spectra_dir.glob("camel_*_spectra_snap_*.hdf5"):
            # Extract snapshot number from filename
            parts = spectra_file.stem.split('_')
            if 'snap' in parts:
                snap_idx = parts.index('snap')
                if snap_idx + 1 < len(parts):
                    try:
                        snap_part = parts[snap_idx + 1]
                        snap_num = int(snap_part)
                        # Store as spectra file (distinguish from snapshot)
                        key = f"spectra_{snap_num}"
                        snapshots[key] = spectra_file
                    except ValueError:
                        continue

    return snapshots


# Print summary of all available simulation data
def list_all_available_data():
    print("=" * 70)
    print("AVAILABLE SIMULATION DATA")
    print("=" * 70)

    for suite_name, suite_path in CAMEL_SUITES.items():
        if not suite_path.exists():
            continue

        print(f"\n{suite_name}:")

        # Dynamically discover all available datasets
        datasets = get_available_datasets(suite_name)

        if not datasets:
            print(f"  No datasets found")
            continue

        for sim_set in sorted(datasets.keys()):
            print(f"  {sim_set}:")

            for sim_num in datasets[sim_set]:
                snapshots = get_available_snapshots(
                    suite_name, sim_set, sim_num)

                if not snapshots:
                    continue

                # Separate simulation snapshots from spectra files
                sim_snaps = {k: v for k, v in snapshots.items()
                             if isinstance(k, int)}
                spectra_files = {k: v for k, v in snapshots.items()
                                 if isinstance(k, str)}

                if sim_snaps or spectra_files:
                    print(f"    {sim_set}_{sim_num}:")

                    if sim_snaps:
                        snap_nums = sorted(sim_snaps.keys())
                        print(f"      Snapshots: {snap_nums}")

                    if spectra_files:
                        spectra_nums = sorted([int(k.split('_')[1])
                                              for k in spectra_files.keys()])
                        print(f"      Spectra:   {spectra_nums}")

    print("=" * 70)


# ============================================================================
# DEFAULT ANALYSIS PARAMETERS
# ============================================================================

# Spectra generation parameters
DEFAULT_NUM_SIGHTLINES = 100        # Number of random sightlines
DEFAULT_PIXEL_RESOLUTION = 0.1      # Resolution in km/s
# Size of simulation box in Mpc/h (for CAMEL)
DEFAULT_BOX_SIZE = 25.0
DEFAULT_RANDOM_SEED = None          # Random seed for reproducibility

# Wavelength range for Lyman-alpha (in Angstroms)
LYMAN_ALPHA_WAVELENGTH = 1215.67    # Rest wavelength
WAVELENGTH_MIN = 1180.0             # Minimum wavelength
WAVELENGTH_MAX = 1250.0             # Maximum wavelength

# Redshift parameters
REDSHIFT_MIN = 2.0                  # Minimum redshift of interest
REDSHIFT_MAX = 6.0                  # Maximum redshift of interest

# Analysis parameters
FLUX_BINS = 50                      # Number of bins for flux histograms
TAU_MAX = 10.0                      # Maximum optical depth to plot
POWER_SPECTRUM_KMIN = 0.001         # Minimum k for power spectrum (s/km)
POWER_SPECTRUM_KMAX = 0.1           # Maximum k for power spectrum (s/km)


# Spectral line definitions: (element, ion_level, wavelength_angstrom, name)
# Format used by fake_spectra: spec.get_tau(element, ion_level, wavelength)
SPECTRAL_LINES = {
    'lya':  ('H',  1, 1215, 'Lyman-alpha'),
    'lyb':  ('H',  1, 1025, 'Lyman-beta'),
    'heii': ('He', 2, 303,  'HeII-303'),
    'civ':  ('C',  4, 1548, 'CIV-1548'),
    'ovi':  ('O',  6, 1031, 'OVI-1031'),
    'mgii': ('Mg', 2, 2796, 'MgII-2796'),
    'siiv': ('Si', 4, 1393, 'SiIV-1393'),
}


# Get spectral line information
def get_line_info(line_code):
    return SPECTRAL_LINES.get(line_code.lower())


# Parse comma-separated list of spectral lines
def parse_line_list(line_string):
    lines = [l.strip().lower() for l in line_string.split(',')]

    # Validate all lines exist
    invalid = [l for l in lines if l not in SPECTRAL_LINES]
    if invalid:
        return None

    return lines


# PLOTTING CONFIGURATION
# Figure sizes (width, height in inches)
FIGSIZE_SINGLE = (10, 6)            # Single panel plots
FIGSIZE_DOUBLE = (14, 6)            # Two-panel plots
FIGSIZE_QUAD = (12, 10)             # Four-panel plots

# DPI for saved figures
PLOT_DPI = 300

# Color schemes
COLORS_QUALITATIVE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Line styles
LINESTYLE_OBSERVED = '--'
LINESTYLE_SIMULATED = '-'
LINEWIDTH_DEFAULT = 1.5

# Font sizes
FONTSIZE_TITLE = 14
FONTSIZE_LABEL = 12
FONTSIZE_TICK = 10
FONTSIZE_LEGEND = 10


# Convert comoving length to proper length
def comoving_to_proper(comoving_length, scale_factor):
    return comoving_length * scale_factor


# Convert redshift to scale factor
def redshift_to_scale_factor(redshift):
    return 1.0 / (1.0 + redshift)


# Convert scale factor to redshift
def scale_factor_to_redshift(scale_factor):
    return (1.0 / scale_factor) - 1.0


# FILE NAMING CONVENTIONS
# Generate standard output filename for spectra
def get_snapshot_output_name(snapshot_path, lines=None, num_sightlines=None):
    snapshot_path = Path(snapshot_path)

    # Extract snapshot number
    snap_num = None
    parts = snapshot_path.stem.split('_')
    for i, part in enumerate(parts):
        if part == 'snap' and i + 1 < len(parts):
            snap_num = parts[i + 1]
            break

    if snap_num is None:
        snap_num = 'unknown'

    if lines is None:
        lines = ['lya']  # Default
    elif isinstance(lines, str):
        lines = [lines]

    # Build filename with line codes
    line_str = '_'.join(lines)
    filename = f"camel_{line_str}_spectra_snap_{snap_num}"

    if num_sightlines is not None:
        filename += f"_n{num_sightlines}"

    filename += ".hdf5"

    return filename


# Generate standard plot filename
def get_plot_output_name(snapshot_path, plot_type, extension='png'):
    snapshot_path = Path(snapshot_path)

    # Extract simulation set and number from path
    sim_name = None
    if len(snapshot_path.parts) >= 2:
        sim_name = snapshot_path.parts[-2]
    
    # Extract snapshot number
    snap_num = 'unknown'
    parts = snapshot_path.stem.split('_')
    for i, part in enumerate(parts):
        if part == 'snap' and i + 1 < len(parts):
            snap_num = parts[i + 1]
            break

    filename = f"camel_{plot_type}_snap_{snap_num}.{extension}"

    # Create subdirectory for this simulation
    if sim_name:
        plot_subdir = PLOTS_DIR / sim_name
        plot_subdir.mkdir(exist_ok=True, parents=True)
        return plot_subdir / filename
    else:
        return PLOTS_DIR / filename


def print_config_summary():
    """
    Print summary of current configuration
    """
    print("\n" + "=" * 70)
    print("CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"Project root:     {PROJECT_ROOT}")
    print(f"Data directory:   {DATA_DIR}")
    print(f"Plots directory:  {PLOTS_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    print("Default Parameters:")
    print(f"  Sightlines:     {DEFAULT_NUM_SIGHTLINES}")
    print(f"  Resolution:     {DEFAULT_PIXEL_RESOLUTION} km/s")
    print(f"  Box size:       {DEFAULT_BOX_SIZE} Mpc/h")
    print(f"  Random seed:    {DEFAULT_RANDOM_SEED}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    print_config_summary()
    list_all_available_data()
