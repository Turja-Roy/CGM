# Breaking the Ω₀–σ₈ Degeneracy with the Lyman-α Forest

Ω₀ (matter density) and σ₈ (fluctuation amplitude) are degenerate in one-point
cosmological statistics: both raise the amplitude of density fluctuations, so most
summary observables respond only to the combination S₈ = σ₈√(Ω_m/0.3). This repository
is the analysis pipeline for a study of whether the Lyman-α forest — and the underlying
dark-matter field — carries information that separates the two.

The data are the **CAMELS IllustrisTNG L25n256 1P set**, where `p1` scans Ω₀ (0.1–0.5)
and `p2` scans σ₈ (0.6–1.0) about a common fiducial, all else fixed. Synthetic forest
spectra are generated with `fake_spectra` on a **shared sightline set** across every
variant (so scan-to-scan differences carry no sample variance), and reduced to τ_eff,
mean flux, flux and optical-depth PDFs, the column density distribution function
f(N_HI), the flux power spectrum P_F(k), the Doppler-width distribution b(N_HI), and
the temperature–density relation. The dark-matter P(k) is measured separately from the
particle field. The **EX set** (fixed cosmology, extreme AGN/SN feedback) brackets the
feedback axis to test whether any discriminant found is robust against it.

---

## Setup

```bash
pip install -r requirements.txt
./compile.sh        # pybind11 C++ extensions; needs cmake, gcc, eigen, fftw3
```

The C++ extensions (`src/cpp/`) carry the power spectrum, CDDF, line widths, T–ρ,
flux stats, and halo selection. They are threaded with OpenMP — set `OMP_NUM_THREADS`.
There is no MPI.

`scripts/fake_spectra_fix.py` patches `fake_spectra` for Python 3.13+; it is imported
automatically before `fake_spectra` and must stay that way.

## Data

```bash
python downloader.py --suite IllustrisTNG --set 1P --sim p1_0 --snapshot 80
python downloader.py --groups data/IllustrisTNG/1P/1P_p1_0/snap_080.hdf5
```

Layout is `data/<suite>/<set>/<sim_name>/snap_XXX.hdf5`. Sets: `LH`, `1P`, `CV`, `EX`.

Snapshot numbers are not redshifts and the published CAMELS table is wrong — read
`Header/Redshift`. Production set: **024, 028, 032, 038, 044, 050, 060, 072, 080, 090**
(z = 4.0, 3.5, 3.0, 2.46, 2.0, 1.6, 1.05, 0.54, 0.27, 0.0).

## Pipeline

```bash
# 1. one shared sightline set for the whole scan
python analyze_spectra.py generate-sightlines scan80 -n 10000 --seed 42

# 2. spectra per variant
python analyze_spectra.py generate 'data/IllustrisTNG/1P/1P_p1_*/snap_080.hdf5' \
    --sightlines-from output/sightlines/scan80.hdf5 --line lya

# 3. reduce to plots + CSVs
python analyze_spectra.py analyze 'spectra/IllustrisTNG/1P/1P_p1_*/camel_*_spectra_snap_080_*.hdf5'

# 4. overlay the variants
python analyze_spectra.py compare 'spectra/IllustrisTNG/1P/1P_p1_*/camel_*_spectra_snap_080_*.hdf5' \
    --param Omega_m --fiducial 1P_0 --name omega_scan_z027
```

Other subcommands: `list`, `explore`, `evolve` (redshift tracks), `diagnose`,
`pipeline` (generate + analyze), `halo`, `cgm` (halo-targeted sightlines at fixed
impact parameter). Run `python analyze_spectra.py <cmd> -h` for flags.

Lines available for `--line` are in `config.SPECTRAL_LINES` (`lya`, `lyb`, `heii`,
`civ`, `ovi`, `mgii`, `siiv`, plus `lya_h` — all hydrogen treated as neutral, i.e. the
Gunn-Peterson optical depth).

## Outputs

`analyze` writes both a figure set and a CSV set, keyed by
`<suite>/<sim_set>/<sim_name>/snap-XXX`:

- `plots/…/` — sample spectra, flux statistics, P_F(k), CDDF, line widths, T–ρ,
  multi-line comparison
- `output/analysis/…/` — `analysis_results.json` plus `power_spectrum.csv`, `cddf.csv`,
  `flux_stats.csv`, `flux_pdf.csv`, `tau_pdf.csv`, `line_widths.csv`,
  `temp_density.csv`, `metal_lines.csv`

Everything downstream reads the CSVs. The generated spectra HDF5s are intermediates and
can be deleted once `analyze` has run — only `evolve` and `diagnose` still need the raw
τ arrays.

## Science scripts

All CSV-only unless noted; each takes `--analysis-root`, `--cosmo-csv`, `--snaps`,
`--out-dir`.

| script | what it does |
|---|---|
| `scripts/degeneracy_test.py` | the S₈ collapse test and the growth/geometry/shape discriminants — the core result |
| `scripts/hypothesis_test_p1.py` | tests the "less Ω₀ → less feedback → more HI" explanation of the p1 CDDF/τ_eff inversion |
| `scripts/matter_pk_test.py` | dark-matter P(k) shape test (k_eq tilt vs flat rescaling). Reads raw snapshots |
| `scripts/pdf_evolution.py` | flux- and τ-PDF overlays across redshift and across variants |
| `scripts/replot.py` | regenerates the `analyze` figures from the CSVs alone |
| `scripts/test_scan_and_colden.py` | self-checks: E(z) uses the real Ω₀, absorber N_HI is a sum |



## References

- CAMELS — Villaescusa-Navarro et al. (2021)
- `fake_spectra` — Bird et al. (2015)
