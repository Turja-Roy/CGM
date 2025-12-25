import os
import argparse

import scripts.config as config
from scripts.commands.generate import cmd_generate
from scripts.commands.analyze import cmd_analyze


# Full pipeline: generate spectra from snapshot and analyze them
def cmd_pipeline(args):
    # Step 1: Generate spectra
    print("PIPELINE STAGE 1/2: GENERATING SPECTRA")
    print("=" * 70)

    ret = cmd_generate(args)
    if ret != 0:
        return ret

    # Determine output file that was generated
    if args.output:
        spectra_file = args.output
    else:
        lines = config.parse_line_list(
            args.line if hasattr(args, 'line') else 'lya')
        spectra_file = config.get_snapshot_output_name(
            args.snapshot,
            lines=lines,
            num_sightlines=args.sightlines
        )
        # It's saved in the same directory as snapshot
        snapshot_dir = os.path.dirname(args.snapshot)
        if snapshot_dir:
            spectra_file = os.path.join(
                snapshot_dir, os.path.basename(spectra_file))

    print("\n\n")
    print("PIPELINE STAGE 2/2: ANALYZING SPECTRA")
    print("=" * 70)

    # Step 2: Analyze
    args_analyze = argparse.Namespace(spectra_file=spectra_file)
    ret = cmd_analyze(args_analyze)

    if ret == 0:
        print("\n\n")
        print("=" * 70)
        print("PIPELINE COMPLETE!")
        print("=" * 70)
        print(f"Spectra file: {spectra_file}")
        print(f"Plots saved:  {config.PLOTS_DIR}")

    return ret
