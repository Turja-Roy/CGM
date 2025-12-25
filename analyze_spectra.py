import sys
import argparse

import scripts.config as config
from scripts.commands import (
    cmd_list,
    cmd_explore,
    cmd_generate,
    cmd_analyze,
    cmd_compare,
    cmd_evolve,
    cmd_pipeline,
    cmd_halo,
    cmd_cgm,
)


def main():
    parser = argparse.ArgumentParser(
        description='CAMEL Lyman-alpha Spectra Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available data
  python analyze_spectra.py list

  # Explore HDF5 file
  python analyze_spectra.py explore data/.../camel_lya_spectra_snap_080.hdf5

  # Generate spectra
  python analyze_spectra.py generate data/snap_080.hdf5 -n 200

  # Analyze existing spectra
  python analyze_spectra.py analyze data/.../camel_lya_spectra_snap_080.hdf5

  # Compare multiple simulations (same redshift, different physics parameters)
  python analyze_spectra.py compare file1.hdf5 file2.hdf5 file3.hdf5 -l "LH_0,LH_80,LH_832"

  # Track redshift evolution (different snapshots from same or different simulations)
  python analyze_spectra.py evolve snap_080.hdf5 snap_085.hdf5 snap_090.hdf5

  # Full pipeline
  python analyze_spectra.py pipeline data/snap_080.hdf5 -n 100 --res 0.05

Configuration:
  Edit config.py to change default parameters and paths

Documentation:
  See README.md for detailed usage instructions
        """
    )

    subparsers = parser.add_subparsers(
        dest='command', help='Available commands')
    subparsers.required = True

    parser_list = subparsers.add_parser(
        'list', help='List all available simulation data')
    parser_list.set_defaults(func=cmd_list)

    parser_explore = subparsers.add_parser(
        'explore', help='Explore HDF5 file structure')
    parser_explore.add_argument(
        'spectra_file', help='Path to HDF5 file to explore')
    parser_explore.set_defaults(func=cmd_explore)

    parser_gen = subparsers.add_parser(
        'generate', help='Generate spectra from snapshot')
    parser_gen.add_argument(
        'snapshot', help='Path to simulation snapshot HDF5 file')
    parser_gen.add_argument('-n', '--sightlines', type=int, default=config.DEFAULT_NUM_SIGHTLINES,
                            help=f'Number of sightlines (default: {config.DEFAULT_NUM_SIGHTLINES})')
    parser_gen.add_argument('-r', '--res', type=float, default=config.DEFAULT_PIXEL_RESOLUTION,
                            help=f'Velocity resolution in km/s (default: {config.DEFAULT_PIXEL_RESOLUTION})')
    parser_gen.add_argument('--line', type=str, default='lya',
                            help='Spectral line(s) to compute: lya,lyb,heii,civ,ovi,mgii,siiv (comma-separated, default: lya)')
    parser_gen.add_argument('--seed', type=int, default=42,
                            help='Random seed (default: 42)')
    parser_gen.add_argument('-o', '--output', type=str, default=None,
                            help='Output file path (default: auto-generated)')
    parser_gen.set_defaults(func=cmd_generate)

    parser_analyze = subparsers.add_parser(
        'analyze', help='Analyze existing spectra file')
    parser_analyze.add_argument(
        'spectra_file', help='Path to spectra HDF5 file')
    parser_analyze.add_argument('--line', type=str, default=None,
                                help='Spectral line to analyze (auto-detect if not specified)')
    parser_analyze.add_argument('--cd-method', type=str, default='simple',
                                choices=['simple', 'vpfit', 'hybrid'],
                                help='Column density calculation method: '
                                'simple (pixel optical depth), '
                                'vpfit (Voigt profile fitting), '
                                'hybrid (simple detection + VPFIT fitting)')
    parser_analyze.add_argument('--max-sightlines', type=int, default=None,
                                help='Maximum number of sightlines to analyze (for memory-limited systems)')
    parser_analyze.set_defaults(func=cmd_analyze)

    parser_compare = subparsers.add_parser('compare',
                                           help='Compare multiple simulation runs (same z, different parameters)')
    parser_compare.add_argument('spectra_files', nargs='+',
                                help='Paths to spectra HDF5 files to compare')
    parser_compare.add_argument('-l', '--labels', type=str, default=None,
                                help='Comma-separated labels for simulations (default: auto-generated)')
    parser_compare.add_argument('-o', '--output', type=str, default=None,
                                help='Output plot path (default: plots/simulation_comparison.png)')
    parser_compare.set_defaults(func=cmd_compare)

    parser_evolve = subparsers.add_parser('evolve',
                                          help='Track how observables evolve with redshift across different snapshots')
    parser_evolve.add_argument('spectra_files', nargs='+',
                               help='Paths to spectra HDF5 files from different snapshots (different redshifts)')
    parser_evolve.add_argument('-l', '--labels', type=str, default=None,
                               help='Comma-separated labels for snapshots (default: auto-generated)')
    parser_evolve.add_argument('-o', '--output', type=str, default=None,
                               help='Output plot path (default: plots/redshift_evolution.png)')
    parser_evolve.set_defaults(func=cmd_evolve)

    parser_pipeline = subparsers.add_parser('pipeline',
                                            help='Full pipeline: generate + analyze')
    parser_pipeline.add_argument(
        'snapshot', help='Path to simulation snapshot HDF5 file')
    parser_pipeline.add_argument('-n', '--sightlines', type=int, default=config.DEFAULT_NUM_SIGHTLINES,
                                 help=f'Number of sightlines (default: {config.DEFAULT_NUM_SIGHTLINES})')
    parser_pipeline.add_argument('-r', '--res', type=float, default=config.DEFAULT_PIXEL_RESOLUTION,
                                 help=f'Velocity resolution in km/s (default: {config.DEFAULT_PIXEL_RESOLUTION})')
    parser_pipeline.add_argument('--line', type=str, default='lya',
                                 help='Spectral line(s) to compute (comma-separated, default: lya)')
    parser_pipeline.add_argument('--seed', type=int, default=42,
                                 help='Random seed (default: 42)')
    parser_pipeline.add_argument('-o', '--output', type=str, default=None,
                                 help='Output file path (default: auto-generated)')
    parser_pipeline.set_defaults(func=cmd_pipeline)

    parser_halo = subparsers.add_parser(
        'halo', help='Analyze individual galaxy halos and their CGM')
    parser_halo.add_argument('snapshot', help='Path to snapshot HDF5')
    parser_halo.add_argument('--mass-range', nargs=2, type=float, 
        default=[11.0, 12.5], metavar=('MIN', 'MAX'),
        help='Halo mass range in log10(M_sun) (default: 11.0 12.5)')
    parser_halo.add_argument('--halo-id', type=int, default=None,
        help='Analyze specific halo by ID')
    parser_halo.add_argument('--n-halos', type=int, default=10,
        help='Number of top massive halos to analyze (default: 10)')
    parser_halo.add_argument('--isolated-only', action='store_true',
        help='Only analyze isolated halos')
    parser_halo.add_argument('--output-dir', type=str, default=None,
        help='Output directory for plots (default: plots/<sim_name>/halos/)')
    parser_halo.add_argument('--slice-thickness', type=float, default=1000,
        help='Projection slice thickness in ckpc/h (default: 1000)')
    parser_halo.add_argument('--plot-type', type=str, default='summary',
        choices=['projection', 'temperature', 'radial', 'summary'],
        help='Type of plot to generate (default: summary)')
    parser_halo.set_defaults(func=cmd_halo)

    parser_cgm = subparsers.add_parser(
        'cgm', help='Generate CGM-targeted spectra around halos')
    parser_cgm.add_argument('snapshot', help='Path to snapshot HDF5')
    parser_cgm.add_argument('--mass-range', nargs=2, type=float,
        default=[11.0, 12.5], metavar=('MIN', 'MAX'),
        help='Halo mass range in log10(M_sun) (default: 11.0 12.5)')
    parser_cgm.add_argument('--n-halos', type=int, default=None,
        help='Number of top massive halos to use (default: all halos in mass range)')
    parser_cgm.add_argument('--impact-params', type=str,
        default='0.25,0.5,0.75,1.0,1.25',
        help='Impact parameters in Rvir, comma-separated (default: 0.25,0.5,0.75,1.0,1.25)')
    parser_cgm.add_argument('--n-per-bin', type=int, default=100,
        help='Number of sightlines per impact parameter bin (default: 100)')
    parser_cgm.add_argument('--azimuthal', type=int, default=8,
        help='Number of azimuthal angle samples (default: 8)')
    parser_cgm.add_argument('--line', type=str, default='lya',
        help='Spectral lines (comma-separated, default: lya)')
    parser_cgm.add_argument('--output', type=str, default=None,
        help='Output spectra filename (default: auto-generated)')
    parser_cgm.add_argument('--isolated-only', action='store_true',
        help='Only use isolated halos')
    parser_cgm.set_defaults(func=cmd_cgm)

    if len(sys.argv) == 1:
        parser.print_help()
        return 1

    args = parser.parse_args()

    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
