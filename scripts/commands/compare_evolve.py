import scripts.config as config
from scripts.comparison import compare_simulations, track_redshift_evolution
# Compare multiple simulations
def cmd_compare(args):
    print("=" * 70)
    print("SIMULATION COMPARISON")
    print("=" * 70)

    # Parse labels if provided
    labels = None
    if args.labels:
        labels = [l.strip() for l in args.labels.split(',')]
        if len(labels) != len(args.spectra_files):
            print(f"Error: Number of labels ({
                  len(labels)}) doesn't match number of files ({len(args.spectra_files)})")
            return 1

    # Default output path
    output_path = args.output if args.output else config.PLOTS_DIR / \
        "simulation_comparison.png"

    # Run comparison
    try:
        comparison = compare_simulations(
            args.spectra_files,
            labels=labels,
            output_path=output_path
        )

        if comparison is None:
            return 1

        print("\n" + "=" * 70)
        print("COMPARISON COMPLETE")
        print("=" * 70)
        print(f"\nCompared {comparison['n_simulations']} simulations:")
        for i, label in enumerate(comparison['labels']):
            res = comparison['results'][i]
            print(f"[{i+1}] {label}: z={res['redshift']:.3f}, "
                f"τ_eff={res['tau_eff']['tau_eff']:.4f}, "
                f"<F>={res['flux_stats']['mean_flux']:.4f}")

        print(f"\nComparison plot: {output_path}")
        print("=" * 70)

        return 0

    except Exception as e:
        print(f"\nError during comparison: {e}")
        return 1


# Track redshift evolution
def cmd_evolve(args):
    print("=" * 70)
    print("REDSHIFT EVOLUTION TRACKING")
    print("=" * 70)

    # Parse labels if provided
    labels = None
    if args.labels:
        labels = [l.strip() for l in args.labels.split(',')]
        if len(labels) != len(args.spectra_files):
            print(f"Error: Number of labels ({
                  len(labels)}) doesn't match number of files ({len(args.spectra_files)})")
            return 1

    # Default output path
    output_path = args.output if args.output else config.PLOTS_DIR / "redshift_evolution.png"

    # Run evolution tracking
    try:
        evolution = track_redshift_evolution(
            args.spectra_files,
            labels=labels,
            output_path=output_path
        )

        if evolution is None:
            return 1

        print("\n" + "=" * 70)
        print("EVOLUTION TRACKING COMPLETE")
        print("=" * 70)
        print(f"\nTracked {evolution['n_snapshots']} snapshots:")
        print(f"Redshift range: z = {
              evolution['redshift_range'][0]:.3f} - {evolution['redshift_range'][1]:.3f}")
        print(f"\nKey trends:")

        # Show trend in tau_eff
        tau_start, tau_end = evolution['tau_eff'][0], evolution['tau_eff'][-1]
        tau_change = ((tau_end - tau_start) / tau_start) * 100
        print(f"tau_eff: {tau_start:.4f} -> {tau_end:.4f} ({tau_change:+.1f}%)")

        # Show trend in mean flux
        flux_start, flux_end = evolution['mean_flux'][0], evolution['mean_flux'][-1]
        flux_change = ((flux_end - flux_start) / flux_start) * 100
        print(f"<F>:   {flux_start:.4f} → {
              flux_end:.4f} ({flux_change:+.1f}%)")

        # Show T-ρ trends if available
        if 'T0' in evolution and len(evolution['T0']) > 0:
            T0_start, T0_end = evolution['T0'][0], evolution['T0'][-1]
            gamma_start, gamma_end = evolution['gamma'][0], evolution['gamma'][-1]
            print(f"T_0:   {T0_start:.0f} K -> {T0_end:.0f} K")
            print(f"gamma: {gamma_start:.3f} -> {gamma_end:.3f}")

        print(f"\nEvolution plot: {output_path}")
        print("=" * 70)

        return 0

    except Exception as e:
        print(f"\nError during evolution tracking: {e}")
        return 1
