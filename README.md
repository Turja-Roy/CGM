### Analysis Capabilities

**Basic IGM Analysis:**
- Flux statistics (mean, median, standard deviation)
- Effective optical depth τ_eff
- Flux power spectrum P_F(k)
- Column density distribution f(N_HI)

**Advanced IGM Physics:**
- Line width distribution b(N_HI) and temperature inference
- Temperature-density relation T(ρ) - IGM equation of state

**Multi-Line & Comparative Analysis:**
- Metal line statistics (CIV, OVI, MgII, SiIV)
- Multi-simulation comparisons
- Redshift evolution tracking

### Code Organization

- **`analyze_spectra.py`**  - Main CLI with 7 subcommands (1002 lines)
- **`utils.py`**            - Analysis functions and plotting (1730+ lines)
- **`config.py`**           - Central configuration and spectral line database (373 lines)
- **`batch_process.py`**    - Batch processing framework for multiple files

### Prerequisites
**Dependencies**
   ```bash
   pip install h5py numpy matplotlib scipy fake_spectra
   ```

### List Available Data
```bash
python analyze_spectra.py list
```

### Explore an HDF5 File
```bash
python analyze_spectra.py explore data/IllustrisTNG/LH/LH_0/snap_080.hdf5
```

### Generate Spectra from Snapshot
```bash
# Generate 10,000 sightlines with default Lyα
python analyze_spectra.py generate data/snap_080.hdf5 -n 10000

# Generate multiple spectral lines (Lyα, CIV, OVI)
python analyze_spectra.py generate data/snap_080.hdf5 -n 10000 --line lya,civ,ovi

# Custom resolution and output location
python analyze_spectra.py generate data/snap_080.hdf5 -n 5000 --res 0.05 -o my_spectra.hdf5
```

### Analyze Existing Spectra
```bash
python analyze_spectra.py analyze spectra_file.hdf5
```

- Flux statistics and sample spectra plots
- Effective optical depth τ_eff
- Flux power spectrum P_F(k)
- Column density distribution f(N_HI)
- Line width distribution b(N_HI) and temperatures (if T/ρ data available)
- Temperature-density relation T(ρ) - IGM equation of state (if data available)
- Metal line statistics (if multiple lines in file)
- Detailed console summary with all statistics

### Compare Multiple Simulations

```bash
# Compare 3 simulations at same redshift
python analyze_spectra.py compare \
  data/LH_0/spectra.hdf5 \
  data/LH_80/spectra.hdf5 \
  data/LH_832/spectra.hdf5 \
  -l "Fiducial,Omega_m+,Sigma_8-" \
  -o plots/simulation_comparison.png
```

Side-by-side comparison with:
- Flux power spectrum P_F(k) overlay
- Effective optical depth comparison
- Mean flux comparison
- Column density distributions with power-law fits
- IGM equation of state (T₀ and γ) if available

### Track Redshift Evolution

```bash
# Track evolution across 3 redshifts
python analyze_spectra.py evolve \
  spectra_z2.0.hdf5 \
  spectra_z2.5.hdf5 \
  spectra_z3.0.hdf5 \
  -l "z=2.0,z=2.5,z=3.0" \
  -o plots/evolution.png
```

Evolution of:
- Effective optical depth τ_eff(z)
- Mean flux <F>(z)
- Number of absorbers vs redshift
- Temperature-density parameters T₀(z) and γ(z)

### Full Pipeline (Generate + Analyze)

```bash
# One command to generate spectra AND run full analysis
python analyze_spectra.py pipeline data/snap_080.hdf5 -n 10000 --res 0.1
```

## Analysis Features

### Tier 1: Basic IGM Analysis

#### Flux Statistics
- **Mean flux** `<F>` - Average transmission through IGM
- **Median flux** - Robust central value
- **Standard deviation** - Flux variability
- **Flux distribution** - Histogram showing absorption patterns

**Physics:** Basic tracer of total HI column density along sightline.

#### Effective Optical Depth τ_eff
```
τ_eff = -ln(<F>)
```
- Measures total absorption averaged over all sightlines
- Direct observable from quasar spectra
- Redshift evolution τ_eff(z) constrains thermal history and cosmology

**Typical values:** τ_eff ~ 0.1-0.5 at z=2-4

#### Flux Power Spectrum P_F(k)
```
P_F(k) = <|δ_F(k)|²>  where  δ_F = (F - <F>) / <F>
```
- **1D power spectrum** in velocity space
- Captures fluctuations at different scales
- Sensitive to matter power spectrum, thermal broadening, and pressure smoothing
- **Key scales:** 
  - Large scales (k < 0.01 km⁻¹ s): Cosmology (σ₈, Ω_m)
  - Small scales (k > 0.1 km⁻¹ s): IGM physics (temperature, pressure)

**Implementation:** FFT-based with proper normalization and binning.

#### Column Density Distribution f(N_HI)
```
f(N_HI) = d²N / (dN_HI dX)
```
- **Number of absorbers per column density per absorption distance**
- Power-law at low N_HI: f(N_HI) ∝ N_HI^(-β) with β ~ 1.5-1.7
- Turnover at high N_HI due to self-shielding and Lyman Limit Systems
- **Absorption distance:** dX/dz = (1+z)² × (Ω_m(1+z)³ + Ω_Λ)^(-1/2) × H₀/100

**Typical range:** 10^12 - 10^17 cm⁻²

**Available Methods:**
- **Simple (pixel optical depth):** Fast approximation using N_HI ≈ 1.13×10¹⁴ × Σ(τ) × Δv
- **VPFIT (Voigt profile fitting):** Proper fitting of Voigt profiles for accurate column densities
- **Hybrid:** Simple detection + VPFIT refinement

**When to use VPFIT:** For publication-quality column densities and b-parameters

#### Line Width Distribution b(N_HI)
- **Doppler b-parameter** from Voigt profile fitting
- **Relates to temperature:** T = 1.28×10⁴ × b² K (for thermal broadening)
- **b(N_HI) correlation:** Tests for non-thermal broadening (bulk flows, turbulence)

**Analysis:**
- Fits Voigt profiles to absorption features
- Extracts b-parameters and column densities
- Computes mean b and temperature statistics
- Plots b-parameter histogram and b(N_HI) scatter with binned medians

**Typical values:** b ~ 20-40 km/s → T ~ 10⁴ - 2×10⁴ K

**Physics:** 
- Pure thermal: points lie on b(N_HI) = const
- Additional broadening: increased scatter or correlation

#### Temperature-Density Relation T(ρ)
```
T = T₀ × (ρ/ρ̄)^(γ-1)
```
- **IGM equation of state**
- **T₀:** Temperature at mean density (reflects heating/cooling balance)
- **γ:** Polytropic index (measures entropy distribution)

**Physical interpretation:**
- γ = 1: Isothermal (no pressure support)
- γ ~ 1.3-1.6: Photo-ionization equilibrium
- γ > 1.6: Recent HeII reionization heating

**Analysis:**
- Extracts temperature and density from spectra
- Filters by optical depth (0.1 < τ < 2.0) to isolate IGM
- Fits power-law in log-log space using robust median binning
- Returns T₀, γ with bootstrap uncertainties

**Expected evolution:**
- T₀ rises at z ~ 3 due to HeII reionization
- γ inverts from > 1.6 to < 1.3 after heating

---

#### Metal Line Analysis
Analyzes weak metal absorption lines in addition to HI Lyα:
- **CIV 1548Å** - Highly ionized carbon (traces hot gas T ~ 10⁵ K)
- **OVI 1031Å** - Oxygen quintuple ionized (WHIM tracer)
- **MgII 2796Å** - Singly ionized magnesium (cool gas T ~ 10⁴ K)
- **SiIV 1393Å** - Silicon triply ionized (intermediate ionization)

**Statistics computed:**
- **Number of absorbers** (lower threshold τ > 0.05)
- **dN/dz** - Incidence rate of absorbers per unit redshift
- **Covering fraction** - Fraction of sightlines with absorption
- **Mean optical depth** - Average absorption strength
- **Column density distribution** - Analogous to f(N_HI)

**Output:** 4-panel comparison plot with dN/dz, covering fraction, mean τ, and column density histograms.

**Physics:** Metal lines trace enrichment history and multiphase IGM/CGM.

#### Simulation Comparison

Compare multiple simulations at same redshift (e.g., parameter variations):

**5-panel publication plot:**
1. **Flux power spectrum P_F(k)** - Overlay with error bands
2. **Effective optical depth τ_eff** - Bar chart comparison
3. **Mean flux <F>** - Bar chart comparison
4. **Column density distributions f(N_HI)** - Overlay with power-law slopes
5. **IGM equation of state** - T₀ and γ comparison (if available)

**Use cases:**
- Compare CAMEL Latin Hypercube variations (Ω_m, σ₈, feedback, etc.)
- Test different baryonic physics models
- Validate against observations

#### Redshift Evolution Tracking

Track observables across multiple redshifts from same simulation:

**4-panel evolution plot:**
1. **τ_eff(z)** - Evolution with error bars
2. **Mean flux <F>(z)** - Transmission evolution
3. **Number of absorbers vs z** - Absorption line statistics
4. **T₀(z) and γ(z)** - IGM thermal history (dual y-axis)

**Shows percentage changes** in key observables.

**Use cases:**
- Thermal history of IGM
- HeII reionization signatures
- Cosmological parameter constraints from evolution

---

## Example workflow for reference

### Example 1: Full Analysis of Single Snapshot
```bash
# Activate environment
source venv/bin/activate

# Generate spectra with temperature/density data
python analyze_spectra.py generate data/snap_080.hdf5 -n 10000

# Run comprehensive Tier 1-3 analysis
python analyze_spectra.py analyze camel_lya_spectra_snap_080.hdf5

# Check output
ls plots/
```

**Output plots:**
- `sample_spectra_*.png` - Example absorption spectra
- `flux_power_spectrum_*.png` - P_F(k) with error bands
- `column_density_*.png` - f(N_HI) with power-law fit
- `line_width_distribution_*.png` - b-parameters and b(N_HI)
- `temperature_density_*.png` - T(ρ) relation with fit
- (Plus other diagnostic plots)

### Example 2: Compare CAMEL Latin Hypercube Variations

```bash
# Generate spectra for 3 LH variations at z~2
for lh in LH_0 LH_80 LH_832; do
    python analyze_spectra.py generate \
        data/IllustrisTNG/LH/$lh/snap_086.hdf5 \
        -n 10000 \
        -o spectra_${lh}.hdf5
done

# Compare all three
python analyze_spectra.py compare \
    spectra_LH_0.hdf5 \
    spectra_LH_80.hdf5 \
    spectra_LH_832.hdf5 \
    -l "Fiducial,Variation_80,Variation_832" \
    -o plots/LH_comparison.png
```

**Identifies:** How parameter changes (Ω_m, σ₈, feedback) affect Lyα forest.

### Example 3: Redshift Evolution Study

```bash
# Generate spectra at multiple redshifts
for snap in 080 083 086 089 092; do
    python analyze_spectra.py generate \
        data/snap_${snap}.hdf5 \
        -n 10000 \
        -o spectra_snap_${snap}.hdf5
done

# Track evolution
python analyze_spectra.py evolve \
    spectra_snap_080.hdf5 \
    spectra_snap_083.hdf5 \
    spectra_snap_086.hdf5 \
    spectra_snap_089.hdf5 \
    spectra_snap_092.hdf5 \
    -l "z=0.1,z=0.5,z=1.0,z=1.5,z=2.0" \
    -o plots/redshift_evolution.png
```

**Shows:** τ_eff(z), <F>(z), thermal history T₀(z) and γ(z).

### Example 4: Multi-Line Metal Analysis

```bash
# Generate spectra with multiple lines
python analyze_spectra.py generate data/snap_086.hdf5 \
    -n 10000 \
    --line lya,civ,ovi,mgii,siiv \
    -o multi_line_spectra.hdf5

# Analyze (automatically detects all lines)
python analyze_spectra.py analyze multi_line_spectra.hdf5
```

**Output:** Multi-line comparison plot with dN/dz, covering fractions, optical depths.

### Example 5: Batch Processing Multiple Snapshots

```bash
# Process all snapshots in directory
python batch_process.py generate \
    data/IllustrisTNG/LH/LH_0/ \
    -n 10000 \
    --workers 4

# Analyze all generated spectra
python batch_process.py analyze \
    data/IllustrisTNG/LH/LH_0/SPECTRA_*/
```

**Parallel processing** with multiple workers for faster turnaround.

## Command Reference

### `list`
**Purpose:** List all available simulation snapshots and spectra files  
**Usage:** `python analyze_spectra.py list`  
**Output:** Prints structured summary of data directory contents

### `explore`
**Purpose:** Explore HDF5 file structure (snapshots or spectra)  
**Usage:** `python analyze_spectra.py explore <file.hdf5> [-d DEPTH]`  
**Options:**
- `-d, --depth`: Maximum tree depth (default: 3)

**Example:**
```bash
python analyze_spectra.py explore data/snap_080.hdf5 -d 2
```

### `generate`
**Purpose:** Generate synthetic spectra from simulation snapshot  
**Usage:** `python analyze_spectra.py generate <snapshot.hdf5> [options]`  
**Options:**
- `-n, --sightlines`: Number of sightlines (default: 100)
- `-r, --res`: Velocity resolution in km/s (default: 0.1)
- `--seed`: Random seed for reproducibility (default: 42)
- `--line`: Spectral lines to generate (default: lya)
  - Single: `--line lya`
  - Multiple: `--line lya,civ,ovi`
  - Available: lya, civ, ovi, mgii, siiv, and more (see `config.SPECTRAL_LINES`)
- `-o, --output`: Output file path (default: auto-generated)

**Example:**
```bash
python analyze_spectra.py generate data/snap_080.hdf5 -n 10000 --line lya,civ,ovi
```

**Note:** Automatically saves temperature and density data for IGM analysis.

### `analyze`
**Purpose:** Comprehensive Tier 1-3 analysis of spectra file  
**Usage:** `python analyze_spectra.py analyze <spectra.hdf5> [options]`  

**Options:**
- `--line`: Spectral line to analyze (auto-detect if not specified)
- `--cd-method`: Column density method - 'simple', 'vpfit', or 'hybrid' (default: simple)

**Performs:**
1. Load spectra and metadata
2. Compute flux statistics
3. Compute effective optical depth τ_eff
4. Compute flux power spectrum P_F(k)
5. Compute column density distribution f(N_HI) [method configurable]
6. Compute line width distribution b(N_HI) (if T/ρ data available)
7. Compute temperature-density relation T(ρ) (if T/ρ data available)
8. Analyze metal lines (if multiple lines present)
9. Generate all plots
10. Print comprehensive summary

**Output:** Saves 8-12 plots to `plots/` directory

**Example:**
```bash
# Use VPFIT for accurate column densities
python analyze_spectra.py analyze spectra.hdf5 --cd-method vpfit

# Hybrid method (recommended for production)
python analyze_spectra.py analyze spectra.hdf5 --cd-method hybrid
```

### `compare`
**Purpose:** Compare multiple simulations side-by-side  
**Usage:** `python analyze_spectra.py compare <file1> <file2> [file3...] [options]`  
**Options:**
- `-l, --labels`: Comma-separated labels (default: File1, File2, ...)
- `-o, --output`: Output plot path (default: plots/simulation_comparison.png)

**Example:**
```bash
python analyze_spectra.py compare \
  spectra_LH_0.hdf5 \
  spectra_LH_80.hdf5 \
  spectra_LH_832.hdf5 \
  -l "Fiducial,Omega_m+,Sigma_8-" \
  -o plots/LH_comparison.png
```

**Output:** 5-panel comparison plot (P_F(k), τ_eff, <F>, f(N_HI), T-ρ)

### `evolve`
**Purpose:** Track redshift evolution of observables  
**Usage:** `python analyze_spectra.py evolve <file1> <file2> [file3...] [options]`  
**Options:**
- `-l, --labels`: Comma-separated redshift labels (default: z1, z2, ...)
- `-o, --output`: Output plot path (default: plots/redshift_evolution.png)

**Example:**
```bash
python analyze_spectra.py evolve \
  spectra_z2.0.hdf5 \
  spectra_z2.5.hdf5 \
  spectra_z3.0.hdf5 \
  -l "z=2.0,z=2.5,z=3.0" \
  -o plots/evolution.png
```

**Output:** 4-panel evolution plot (τ_eff(z), <F>(z), N_absorbers(z), T₀/γ(z))

### `pipeline`
**Purpose:** Full pipeline - generate and analyze in one command  
**Usage:** `python analyze_spectra.py pipeline <snapshot.hdf5> [options]`  
**Options:** Same as `generate` command

**Example:**
```bash
python analyze_spectra.py pipeline data/snap_080.hdf5 -n 10000 --res 0.1
```

**Performs:**
1. Generate spectra from snapshot
2. Run full Tier 1-3 analysis
3. Save all plots

---

## Batch Processing & HPC

### Using batch_process.py

Process multiple files in parallel:

```bash
# Generate spectra for all snapshots in directory
python batch_process.py generate data/IllustrisTNG/LH/LH_0/ \
  -n 10000 \
  --workers 4

# Analyze all spectra files in directory
python batch_process.py analyze data/IllustrisTNG/LH/LH_0/SPECTRA_*/ \
  --workers 4
```

**Options:**
- `--workers`: Number of parallel processes (default: 4)
- `--pattern`: File pattern to match (default: *.hdf5)

### SLURM Scripts for HPC

Located in `slurm_templates/`:

**1. Generate spectra (single snapshot):**
```bash
sbatch slurm_templates/generate_spectra.sbatch data/snap_080.hdf5 10000
```

**2. Analyze spectra (single file):**
```bash
sbatch slurm_templates/analyze_spectra.sbatch spectra_file.hdf5
```

**3. Full pipeline (entire simulation suite):**
```bash
sbatch slurm_templates/batch_pipeline.sbatch data/IllustrisTNG/LH/LH_0/
```

**Customize for your cluster:**
- Edit `#SBATCH` directives (partition, time, memory, CPUs)
- Modify module loads (e.g., `module load python/3.13`)
- Adjust array job sizes for parallel processing

**Example customization:**
```bash
#SBATCH --partition=regular
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --array=0-99  # Process 100 snapshots in parallel
```

---

## Output Files

### Generated Spectra (HDF5)

**New format** (with temperature/density data):
```
camel_lya_spectra_snap_XXX.hdf5
├── Header/
│   ├── redshift              # Snapshot redshift
│   ├── hubble                # Hubble parameter H(z)
│   ├── box_size              # Simulation box size (comoving kpc/h)
│   └── ...
├── tau/                      # Optical depth (new multi-line format)
│   ├── H/1/1215              # HI Lyα (element/ion/wavelength)
│   ├── C/4/1548              # CIV 1548Å
│   └── O/6/1031              # OVI 1031Å
├── temperature/              # Temperature field (NEW in current version)
│   └── H/1/                  # Per element/ion
├── density_weight_density/   # Density field (NEW in current version)
│   └── H/1/                  # Per element/ion
├── flux                      # Transmitted flux [n_sightlines, n_pixels]
├── wavelength                # Wavelength array [n_pixels] (Angstroms)
└── ...
```

**Legacy format** (older files):
```
├── tau                       # Direct optical depth array
├── flux
└── wavelength
```

**Note:** Code handles both formats automatically.

### Plot Files

All plots saved to `plots/` directory with descriptive names:

**Tier 1 plots:**
- `sample_spectra_*.png` - Example absorption spectra (5-10 sightlines)
- `flux_statistics_*.png` - Flux and tau histograms
- `flux_power_spectrum_*.png` - P_F(k) with error bars/bands
- `column_density_distribution_*.png` - f(N_HI) with power-law fit

**Tier 2 plots:**
- `line_width_distribution_*.png` - b-parameter histogram and b(N_HI) correlation
- `temperature_density_relation_*.png` - T vs ρ/ρ̄ with power-law fit

**Tier 3 plots:**
- `multi_line_comparison_*.png` - 4-panel metal line statistics
- `simulation_comparison.png` - 5-panel comparison of simulations
- `redshift_evolution.png` - 4-panel evolution tracking

**Plus additional diagnostic plots:** optical depth maps, detailed statistics, etc.

---

## Configuration

Edit `config.py` to customize:

### Paths
```python
DATA_DIR = Path(__file__).parent / 'data'
PLOTS_DIR = Path(__file__).parent / 'plots'
OUTPUT_DIR = Path(__file__).parent / 'output'
```

### Default Parameters
```python
DEFAULT_SIGHTLINES = 100
DEFAULT_RESOLUTION = 0.1  # km/s
DEFAULT_BOX_SIZE = 25000  # ckpc/h
DEFAULT_SEED = 42
```

### Spectral Lines Database
```python
SPECTRAL_LINES = {
    'lya': {'wavelength': 1215.67, 'element': 'H', 'ion': 1},
    'civ': {'wavelength': 1548.19, 'element': 'C', 'ion': 4},
    'ovi': {'wavelength': 1031.93, 'element': 'O', 'ion': 6},
    'mgii': {'wavelength': 2796.35, 'element': 'Mg', 'ion': 2},
    'siiv': {'wavelength': 1393.76, 'element': 'Si', 'ion': 4},
    # ... more lines
}
```

### Physical Constants
```python
HUBBLE_CONSTANT = 0.6774  # h = H0 / (100 km/s/Mpc)
SPEED_OF_LIGHT = 299792.458  # km/s
```

### Plotting Settings
```python
PLOT_DPI = 150
FIGURE_SIZE = (12, 8)
COLOR_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', ...]
```

---

## Common Workflows

### 1. Quick Test Analysis (10 minutes)

```bash
# Generate small test dataset
python analyze_spectra.py generate data/snap_080.hdf5 -n 100 -o /tmp/test.hdf5

# Run analysis
python analyze_spectra.py analyze /tmp/test.hdf5

# View plots
ls plots/
```

### 2. Production Analysis (1-2 hours)

```bash
# Generate 10,000 sightlines with multiple lines
python analyze_spectra.py generate data/snap_080.hdf5 \
  -n 10000 \
  --line lya,civ,ovi \
  -o production_spectra.hdf5

# Full Tier 1-3 analysis
python analyze_spectra.py analyze production_spectra.hdf5
```

**Output:** Comprehensive plots and statistics for publication.

### 3. Parameter Study (LH Suite)

```bash
# Generate spectra for 3 parameter variations
for lh in 0 80 832; do
    python analyze_spectra.py generate \
        data/IllustrisTNG/LH/LH_${lh}/snap_086.hdf5 \
        -n 10000 \
        -o spectra_LH_${lh}.hdf5
done

# Compare all variations
python analyze_spectra.py compare \
    spectra_LH_0.hdf5 \
    spectra_LH_80.hdf5 \
    spectra_LH_832.hdf5 \
    -l "Fiducial,Var80,Var832" \
    -o plots/parameter_study.png
```

**Purpose:** Identify which cosmological/astrophysical parameters affect Lyα forest.

### 4. Redshift Evolution Study (Overnight)

```bash
# Generate spectra at 5 redshifts (use HPC for faster processing)
snapshots=(080 083 086 089 092)
for i in "${!snapshots[@]}"; do
    snap=${snapshots[$i]}
    python analyze_spectra.py generate \
        data/snap_${snap}.hdf5 \
        -n 10000 \
        -o spectra_snap_${snap}.hdf5
done

# Track evolution
python analyze_spectra.py evolve \
    spectra_snap_080.hdf5 \
    spectra_snap_083.hdf5 \
    spectra_snap_086.hdf5 \
    spectra_snap_089.hdf5 \
    spectra_snap_092.hdf5 \
    -l "z=0.1,z=0.5,z=1.0,z=1.5,z=2.0" \
    -o plots/thermal_history.png
```

**Scientific output:** IGM thermal history, HeII reionization signatures.

### 5. HPC Production Run (Full Suite)

```bash
# Copy SLURM template and customize
cp slurm_templates/batch_pipeline.sbatch my_pipeline.sbatch

# Edit job parameters
nano my_pipeline.sbatch

# Submit job array for entire LH suite
sbatch my_pipeline.sbatch data/IllustrisTNG/LH/
```

**Processes:** All snapshots, all variations, overnight on cluster.

---

## Script Integration & Python API

You can import and use the analysis functions directly in custom scripts:

```python
#!/usr/bin/env python3
import h5py
import numpy as np
from pathlib import Path

# Import our modules
import config
import utils

# Load snapshot metadata
metadata = utils.load_snapshot_metadata('data/snap_080.hdf5')
print(f"Redshift: {metadata['redshift']:.3f}")
print(f"Box size: {metadata['box_size']:.1f} ckpc/h")

# Load and analyze spectra
with h5py.File('spectra.hdf5', 'r') as f:
    flux = f['flux'][:]
    tau = f['tau/H/1/1215'][:]  # HI Lyα optical depth
    wavelength = f['wavelength'][:]
    redshift = f['Header'].attrs['redshift']
    
    # Compute statistics
    stats = utils.compute_flux_statistics(flux, tau)
    print("\nFlux Statistics:")
    print(utils.format_stats_table(stats))
    
    # Compute power spectrum
    k_bins, Pk, Pk_err = utils.compute_power_spectrum(flux, res=0.1)
    
    # Compute column density distribution
    result = utils.compute_column_density_distribution(
        tau, wavelength, redshift, res=0.1
    )
    
    # If temperature/density data available
    if 'temperature/H/1' in f:
        temp = f['temperature/H/1'][:]
        dens = f['density_weight_density/H/1'][:]
        
        # Compute T-ρ relation
        T0, gamma, T0_err, gamma_err = utils.compute_temperature_density_relation(
            temp, dens, tau, threshold_low=0.1, threshold_high=2.0
        )
        print(f"\nIGM Equation of State:")
        print(f"  T0 = {T0:.0f} ± {T0_err:.0f} K")
        print(f"  γ = {gamma:.3f} ± {gamma_err:.3f}")
```

### Available Functions

**Tier 1 Analysis:**
- `compute_flux_statistics(flux, tau)` → dict with mean, median, std
- `compute_effective_optical_depth(flux)` → τ_eff
- `compute_power_spectrum(flux, res)` → k_bins, Pk, Pk_err
- `compute_column_density_distribution(tau, wavelength, z, res)` → dict

**Tier 2 Analysis:**
- `compute_line_width_distribution(tau, wavelength, z)` → N_HI, b_params, temperatures
- `compute_temperature_density_relation(T, ρ, tau)` → T0, gamma, errors

**Tier 3 Analysis:**
- `compute_metal_line_statistics(tau, wavelength, z, line_name)` → dict
- `load_spectra_results(filepath)` → comprehensive results dict
- `compare_simulations(filepaths, labels)` → comparison plot
- `track_redshift_evolution(filepaths, labels)` → evolution plot

**Plotting Functions:**
- `plot_sample_spectra(flux, wavelength, z, save_path)`
- `plot_flux_power_spectrum(k_bins, Pk, Pk_err, save_path)`
- `plot_column_density_distribution(result, save_path)`
- `plot_line_width_distribution(N_HI, b_params, temps, save_path)`
- `plot_temperature_density_relation(T, rho, T0, gamma, save_path)`
- `plot_multi_line_comparison(lines_data, save_path)`

---

## Column Density Methods

### Simple Method (Default)
- **Algorithm**: Pixel optical depth summation
- **Formula**: N_HI ≈ 1.13×10¹⁴ × Σ(τ) × Δv cm⁻²
- **Pros**: Fast, works on any spectra
- **Cons**: Approximate, no error estimates
- **Use when**: Quick analysis, large datasets, testing

### VPFIT Method (Accurate)
- **Algorithm**: Voigt profile fitting with VoigtFit
- **Features**: Proper Voigt profiles, column density uncertainties, b-parameters
- **Pros**: Publication-quality, accurate physics
- **Cons**: Slower, requires VoigtFit installation
- **Use when**: Final analysis, publications, high accuracy needed

### Hybrid Method (Recommended)
- **Algorithm**: Simple detection + VPFIT refinement
- **Pros**: Fast detection, accurate fitting, robust
- **Cons**: Requires VoigtFit
- **Use when**: Production analysis, best of both worlds

**Usage:**
```bash
# Simple method (default)
python analyze_spectra.py analyze spectra.hdf5

# VPFIT method
python analyze_spectra.py analyze spectra.hdf5 --cd-method vpfit

# Hybrid method (recommended)
python analyze_spectra.py analyze spectra.hdf5 --cd-method hybrid
```

---

## Troubleshooting

### File not found errors
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/snap_080.hdf5'
```

**Solutions:**
- Check data is in `data/` directory: `python analyze_spectra.py list`
- Use absolute paths: `python analyze_spectra.py generate /full/path/to/snap_080.hdf5`
- Verify file exists: `ls -lh data/snap_080.hdf5`

### ImportError for fake_spectra
```
ModuleNotFoundError: No module named 'fake_spectra'
```

**Solution:**
```bash
source venv/bin/activate
pip install fake_spectra
```

### Python 3.13 compatibility issues
```
OverflowError: Python int too large to convert to C ulong
```

**Solution:** This is automatically fixed by importing `utils` (monkey patches applied). Ensure:
```python
import utils  # Applies bugfixes automatically
import fake_spectra  # Now safe to use
```

If issues persist, downgrade to Python 3.11:
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Memory issues with large datasets
```
MemoryError: Unable to allocate array
```

**Solutions:**
- Reduce sightlines: `python analyze_spectra.py generate snap.hdf5 -n 1000` (instead of 10000)
- Process in batches using `batch_process.py`
- Use HPC with more memory: Edit SLURM script `#SBATCH --mem=128GB`

### Missing temperature/density data
```
Warning: Temperature/density data not found. Skipping Tier 2 analysis.
```

**Cause:** Spectra file generated with older version of code (before lines 281-295 of analyze_spectra.py were added).

**Solution:** Re-generate spectra with current version:
```bash
python analyze_spectra.py generate data/snap_080.hdf5 -n 10000 -o new_spectra.hdf5
```

### No metal lines detected
```
Info: Metal line analysis skipped - only 1 spectral line found.
```

**Cause:** File only contains HI Lyα.

**Solution:** Generate with multiple lines:
```bash
python analyze_spectra.py generate data/snap_080.hdf5 -n 10000 --line lya,civ,ovi
```

### Plots look empty or strange
```
Warning: Very few absorbers found (N=2). Results may be unreliable.
```

**Causes:**
- Too few sightlines (use -n 1000 or more)
- Wrong redshift (metal lines weak at z < 1)
- Incorrect threshold (HI uses 0.5, metals use 0.05)

**Solutions:**
- Increase sightlines: `-n 10000`
- Use higher redshift snapshot (z > 2 for metals)
- Check console output for statistics

### HDF5 structure errors
```
KeyError: 'tau/H/1/1215'
```

**Cause:** Old vs new HDF5 format.

**Solution:** Code handles both automatically. If error persists:
```bash
# Check file structure
python analyze_spectra.py explore spectra.hdf5 -d 3

# Re-generate with current version
python analyze_spectra.py generate data/snap.hdf5 -o new_spectra.hdf5
```

---

## Performance Notes

### Timing Estimates

**Spectra generation** (depends on snapshot size and number of sightlines):
- 100 sightlines: ~30 seconds
- 1,000 sightlines: ~3 minutes
- 10,000 sightlines: ~5-10 minutes
- Multiple lines (3-5): ~2-3× longer

**Analysis** (depends on number of sightlines):
- Tier 1 only: ~30 seconds (1000 sightlines) to ~2 minutes (10000 sightlines)
- Tier 1+2+3: ~1-3 minutes (includes FFT, Voigt fitting, T-ρ analysis)

**Bottlenecks:**
- Power spectrum: FFT on all sightlines (most expensive)
- Line width fitting: Voigt profile curve_fit (scales with N_absorbers)
- Column density: Pixel-by-pixel conversion

### Optimization Tips

1. **Use appropriate sample sizes:**
   - Testing: 100-1000 sightlines
   - Production: 5000-10000 sightlines
   - Publication: 10000+ sightlines

2. **Batch processing:**
   ```bash
   # Process multiple snapshots in parallel
   python batch_process.py generate data/IllustrisTNG/LH/LH_0/ -n 10000 --workers 4
   ```

3. **HPC job arrays:**
   ```bash
   # Process 100 snapshots simultaneously
   #SBATCH --array=0-99
   ```

4. **Skip unnecessary analysis:**
   - Analyze only files with temperature/density data for Tier 2
   - Metal lines only relevant at z > 1.5

---

## Extending the Project

### Adding New Analysis Functions

1. **Add function to `utils.py`:**
   ```python
   def compute_my_new_statistic(spectra_data):
       """
       Compute my custom statistic.
       
       Args:
           spectra_data: Input data array
           
       Returns:
           result: Computed statistic
       """
       # Your analysis code
       result = np.mean(spectra_data)
       return result
   ```

2. **Add plotting function to `utils.py`:**
   ```python
   def plot_my_statistic(result, save_path):
       """Plot the statistic."""
       fig, ax = plt.subplots(figsize=(10, 6))
       ax.plot(result)
       ax.set_xlabel('X')
       ax.set_ylabel('My Statistic')
       plt.tight_layout()
       plt.savefig(save_path, dpi=150, bbox_inches='tight')
       plt.close()
   ```

3. **Integrate into `analyze_spectra.py`:**
   ```python
   def cmd_analyze(args):
       # ... existing code ...
       
       # Add your analysis
       print("Computing my new statistic...")
       my_result = utils.compute_my_new_statistic(flux)
       
       # Add your plot
       print("Generating my plot...")
       save_path = config.PLOTS_DIR / f"my_statistic_{snap_id}.png"
       utils.plot_my_statistic(my_result, save_path)
   ```

### Adding New Subcommands

1. **Define command function in `analyze_spectra.py`:**
   ```python
   def cmd_mynewcommand(args):
       """
       Handler for my new command.
       """
       print(f"Running my command on {args.input_file}")
       
       # Your command logic here
       with h5py.File(args.input_file, 'r') as f:
           # Process data
           pass
       
       print("Done!")
       return 0  # Success
   ```

2. **Add subparser in `main()` function:**
   ```python
   def main():
       # ... existing parsers ...
       
       # Add your subcommand
       parser_new = subparsers.add_parser(
           'mynewcommand',
           help='Description of my new command'
       )
       parser_new.add_argument('input_file', help='Input HDF5 file')
       parser_new.add_argument('-o', '--output', help='Output file')
       parser_new.set_defaults(func=cmd_mynewcommand)
   ```

3. **Use your command:**
   ```bash
   python analyze_spectra.py mynewcommand data.hdf5 -o output.png
   ```

### Adding New Spectral Lines

Edit `config.py`:

```python
SPECTRAL_LINES = {
    # ... existing lines ...
    
    # Add your new line
    'mynewline': {
        'wavelength': 1234.56,  # Rest wavelength in Angstroms
        'element': 'X',          # Element symbol
        'ion': 3,                # Ionization state
        'oscillator_strength': 0.123,  # f-value (optional)
        'gamma': 1e8,            # Damping constant (optional)
    },
}
```

Then use it:
```bash
python analyze_spectra.py generate data/snap.hdf5 --line lya,mynewline
```

### Customizing Plot Styles

Edit `config.py` or individual plotting functions:

```python
# In config.py
PLOT_STYLE = {
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'font.family': 'serif',
}

# Apply globally
import matplotlib.pyplot as plt
plt.rcParams.update(PLOT_STYLE)
```

Or in individual plot functions:
```python
def plot_my_figure(data, save_path):
    # Publication-ready style
    plt.style.use('seaborn-v0_8-paper')
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Custom colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # High DPI for publications
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
```

---

## Scientific Background

### The Lyman-Alpha Forest

The **Lyman-alpha forest** is a series of absorption features in quasar spectra caused by neutral hydrogen in the **intergalactic medium (IGM)** at various redshifts. Each absorption line corresponds to an HI cloud along the line of sight.

**Key physics:**
- Lyα transition: 1s → 2p at 1215.67 Å rest wavelength
- Absorption strength traces HI column density N_HI
- Line positions encode cosmological distances via Hubble flow
- Line widths reveal gas temperatures and dynamics

**Why study it:**
- Traces large-scale structure of the universe
- Probes thermal history of the IGM
- Constrains cosmological parameters (Ω_m, σ₈, H₀)
- Tests models of reionization and galaxy feedback

### CAMEL Simulations

**CAMEL** (Cosmology and Astrophysics with MachinE Learning) is a suite of hydrodynamical simulations based on IllustrisTNG physics.

**Latin Hypercube (LH) Suite:**
- Systematic variation of cosmological and astrophysical parameters
- 1000+ simulation variations
- Parameters varied: Ω_m, Ω_b, h, n_s, σ₈, feedback strengths
- **Purpose:** Build emulators for parameter inference and ML training

**Snapshot Format:**
- IllustrisTNG snapshots at multiple redshifts (z=0 to z=6)
- Contains gas particles with positions, densities, temperatures, metallicities
- Box size: ~25 Mpc/h (comoving)

### What This Pipeline Does

1. **Generates synthetic spectra** using `fake_spectra` library:
   - Shoots random sightlines through simulation box
   - Computes optical depth τ via Voigt profile convolution
   - Produces realistic Lyα absorption spectra

2. **Analyzes IGM physics:**
   - **Flux statistics:** Basic transmission properties
   - **Power spectrum P_F(k):** Scale-dependent fluctuations
   - **Column densities:** Distribution of absorbers
   - **Line widths:** Temperature inference
   - **T-ρ relation:** IGM equation of state
   - **Metal lines:** Enrichment and multiphase gas

3. **Enables comparative studies:**
   - Parameter sensitivity analysis (LH variations)
   - Redshift evolution tracking
   - Comparison with observations

**Scientific output:** Observables that can be directly compared to real quasar spectra from Keck, VLT, Magellan, etc.

---

---

## Citation & References

### Citing This Work

If you use this pipeline in your research, please cite:

1. **CAMEL Simulations:**
   - Villaescusa-Navarro et al. (2021), ApJ, 915, 71
   - "The CAMEL Project: Cosmology and Astrophysics with MachinE Learning"
   - https://ui.adsabs.harvard.edu/abs/2021ApJ...915...71V

2. **fake_spectra Library:**
   - Bird et al. (2015), MNRAS, 447, 1834
   - "Constraints on Neutrino Masses from Lyman-alpha Forest Power Spectrum"
   - https://github.com/sbird/fake_spectra

3. **IllustrisTNG Physics:**
   - Pillepich et al. (2018), MNRAS, 473, 4077
   - "Simulating Galaxy Formation with the IllustrisTNG Model"
   - https://ui.adsabs.harvard.edu/abs/2018MNRAS.473.4077P

### Useful References

**Lyman-alpha forest reviews:**
- Meiksin (2009), Reviews of Modern Physics, 81, 1405
- McQuinn (2016), ARA&A, 54, 313

**Voigt profile fitting and line widths:**
- Tepper-García (2006), MNRAS, 369, 2025

**Column density distributions:**
- Kim et al. (2013), MNRAS, 382, 1657

**Temperature-density relation:**
- Hui & Gnedin (1997), MNRAS, 292, 27
- Becker et al. (2011), MNRAS, 410, 1096

**Flux power spectrum:**
- Palanque-Delabrouille et al. (2013), A&A, 559, A85
- Iršič et al. (2017), Phys. Rev. D, 96, 023522

---

## Project History

This is a **refactored and feature-enhanced** version of scattered analysis scripts developed over multiple sessions.

**Major Improvements:**
- Consolidated 7+ scattered scripts → 4 well-organized modules
- Added comprehensive CLI with 7 subcommands
- Centralized configuration in `config.py`
- Automatic Python 3.13 compatibility patches
- Complete Tier 1-3 analysis framework (1730+ lines)
- Simulation comparison and redshift evolution tracking
- HPC-ready batch processing with SLURM templates
- Publication-quality plotting throughout
- Comprehensive documentation

**Development Timeline:**
- **Initial scripts:** Basic spectra generation and flux statistics
- **Refactor (Session 1):** CLI interface and modular structure
- **Tier 1 (Session 2):** Power spectrum, column densities, batch processing
- **Tier 2 (Session 3):** Line widths b(N_HI), temperature-density T(ρ)
- **Tier 3 (Session 3):** Metal lines, comparisons, evolution tracking
- **Documentation (Session 3):** Comprehensive README with all features

**Current Status:** Feature-complete and production-ready for PhD research.

**All original scripts** safely archived in `archive/` for reference.

---

## Related Documentation

- **`AGENTS.md`** - Code style guidelines (for AI coding assistants)
- **`ANALYSIS_GUIDE.md`** - Detailed analysis procedures and physics explanations
- **`Reading/Antareep_Proposal.txt`** - PhD research proposal and scientific goals

---

## License & Contact

This project is for **academic research purposes**.

For issues, questions, or contributions, please contact the project maintainer.

---

## Quick Command Cheat Sheet

```bash
# Always activate venv first
source venv/bin/activate

# List available data
python analyze_spectra.py list

# Explore HDF5 file structure
python analyze_spectra.py explore <file.hdf5>

# Generate spectra (10K sightlines, includes T/ρ data)
python analyze_spectra.py generate <snapshot.hdf5> -n 10000

# Generate multiple spectral lines
python analyze_spectra.py generate <snapshot.hdf5> -n 10000 --line lya,civ,ovi

# Full Tier 1-3 analysis
python analyze_spectra.py analyze <spectra.hdf5>

# Compare multiple simulations
python analyze_spectra.py compare file1.hdf5 file2.hdf5 file3.hdf5 \
  -l "Sim1,Sim2,Sim3" -o plots/comparison.png

# Track redshift evolution
python analyze_spectra.py evolve z1.hdf5 z2.hdf5 z3.hdf5 \
  -l "z=2.0,z=2.5,z=3.0" -o plots/evolution.png

# Full pipeline (generate + analyze)
python analyze_spectra.py pipeline <snapshot.hdf5> -n 10000

# Batch processing
python batch_process.py generate data/directory/ -n 10000 --workers 4
```

---

**End of README** • Last updated: December 2025
