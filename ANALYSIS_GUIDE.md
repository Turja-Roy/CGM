# How to Generate Analysis Plots from CAMEL Data

## The 5-Step Pipeline

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐    ┌────────────┐
│   CAMEL     │ →  │  Generate    │ →  │ fake_spectra │ →  │ Extract  │ →  │  Create    │
│  Snapshot   │    │  Sightlines  │    │  Compute τ   │    │   Flux   │    │   Plots    │
└─────────────┘    └──────────────┘    └──────────────┘    └──────────┘    └────────────┘
  snap_014.hdf5     cofm, axis          tau array           flux=exp(-τ)    Statistics, P(k)
```

## Quick Start

```bash
# Activate environment
source venv/bin/activate

# Run complete analysis (generates spectra + plots)
python3 analyze_camel_spectra.py

# Expected runtime: 5-10 minutes for 100 sightlines
```

## What You Get

### 4 Analysis Plots:

1. **camel_sample_spectra.png**
   - Shows 5 random Lyman-α spectra
   - Flux vs velocity
   - See individual absorption features

2. **camel_flux_statistics.png**
   - Mean flux profile (with ±1σ)
   - Flux distribution (PDF)
   - Mean optical depth profile
   - τ_eff distribution

3. **camel_power_spectrum.png**
   - 1D flux power spectrum P(k)
   - Shows clustering on different scales
   - Compare with observations

4. **camel_transmission_stats.png**
   - Transmission distribution
   - Optical depth histogram
   - Flux CDF
   - Summary statistics table

### Spectra Data File:

- **camel_lya_spectra_z6.hdf5**
- Contains: τ array (100 × N_pixels)
- Can load and reanalyze without rerunning fake_spectra

## Key Parameters to Modify

```python
# In analyze_camel_spectra.py:

# Number of sightlines (more = better statistics, slower)
num_sightlines = 100  # Try: 50, 100, 500, 1000

# Velocity resolution (smaller = higher quality, slower)
res=0.1  # km/s per pixel
         # Try: 0.5 (fast), 0.1 (good), 0.05 (high quality)

# Or use fixed number of bins instead:
# nbins=512, res=None

# Random seed (for reproducibility)
np.random.seed(42)  # Change for different sightline samples
```

## Common Analysis Patterns

### Pattern 1: Just Load Existing Spectra (No Recompute)

```python
import h5py
import numpy as np

# Load pre-computed spectra
with h5py.File('camel_lya_spectra_z6.hdf5', 'r') as f:
    tau = f['tau_H_1_1215'][:]  # Optical depth
    dvbin = f.attrs['dvbin']     # km/s per pixel
    
flux = np.exp(-tau)
mean_flux = np.mean(flux)
print(f"Mean flux: {mean_flux:.4f}")
```

### Pattern 2: Compute New Statistics

```python
# After loading flux:

# Median flux per sightline
median_flux_per_line = np.median(flux, axis=1)

# Flux at specific velocity range
v_start, v_end = 1000, 2000  # km/s
flux_slice = flux[:, v_start:v_end]

# Saturated pixels (τ > 3)
saturated = (tau > 3.0).sum()
print(f"Saturated: {saturated} pixels")

# Line width distribution (measure absorption width)
# ... (more complex analysis)
```

### Pattern 3: Compare Different Redshifts

```python
# Load spectra from multiple snapshots
files = {
    'z=6': 'camel_lya_spectra_z6.hdf5',
    'z=4': 'camel_lya_spectra_z4.hdf5',
    'z=2': 'camel_lya_spectra_z2.hdf5'
}

mean_fluxes = {}
for label, file in files.items():
    with h5py.File(file, 'r') as f:
        tau = f['tau_H_1_1215'][:]
        flux = np.exp(-tau)
        mean_fluxes[label] = np.mean(flux)

# Plot evolution
plt.plot(list(mean_fluxes.keys()), list(mean_fluxes.values()))
```

## Understanding the Plots

### Mean Flux Profile
- **Flat profile** = homogeneous IGM
- **Variations** = large-scale structure
- **Edge effects** = periodic boundary conditions

### Flux Distribution
- **Peak near F=1** = mostly transparent regions
- **Long tail to F=0** = strong absorbers
- **Log scale** shows rare deep absorbers

### Power Spectrum P(k)
- **Large k (right)** = small-scale fluctuations
- **Small k (left)** = large-scale clustering
- **Slope** = nature of density fluctuations
- Compare with Becker et al. (2011) observations

### Effective Optical Depth
- At z~6: τ_eff ~ 3-5 (saturated forest)
- At z~3: τ_eff ~ 0.5-1.0
- At z~2: τ_eff ~ 0.1-0.3

## Customizing Plots

### Change Number of Sample Spectra

```python
# In PLOT 1 section:
fig1, axes = plt.subplots(10, 1, figsize=(14, 20))  # Show 10 instead of 5
indices = np.random.choice(num_sightlines, 10, replace=False)
```

### Add Your Own Plot

```python
# After PLOT 4, before Summary:

fig5, ax = plt.subplots(figsize=(10, 6))

# Example: Flux variance vs velocity
flux_variance = np.var(flux, axis=0)
ax.plot(velocity, flux_variance, 'b-', lw=2)
ax.set_xlabel('Velocity [km/s]')
ax.set_ylabel('Flux Variance')
ax.set_title('Flux Variance Profile')
ax.grid(alpha=0.2)

plt.savefig('plots/camel_flux_variance.png', dpi=150)
```

### Change Velocity Resolution

```python
# Higher resolution (slower, more accurate)
spec = spectra.Spectra(..., res=0.05, ...)  # 0.05 km/s

# Lower resolution (faster, less detail)
spec = spectra.Spectra(..., res=1.0, ...)   # 1.0 km/s

# Fixed bins (easier to compare)
spec = spectra.Spectra(..., nbins=1024, res=None, ...)
```

## Performance Tips

| Sightlines | Resolution | Runtime | Use Case |
|------------|------------|---------|----------|
| 10 | 1.0 km/s | ~30 sec | Quick test |
| 50 | 0.5 km/s | ~2 min | Fast preview |
| 100 | 0.1 km/s | ~5 min | Standard analysis |
| 500 | 0.1 km/s | ~25 min | High statistics |
| 1000 | 0.05 km/s | ~2 hours | Publication quality |

## Troubleshooting

**Issue: "fake_spectra takes too long"**
```python
# Solution: Reduce sightlines or resolution
num_sightlines = 50
res = 0.5  # km/s
```

**Issue: "Memory error"**
```python
# Solution: Process in batches
for batch in range(10):
    cofm_batch = cofm[batch*10:(batch+1)*10]
    spec = spectra.Spectra(..., cofm=cofm_batch, ...)
    # Save batch results
```

**Issue: "Plots look noisy"**
```python
# Solution: Increase number of sightlines
num_sightlines = 500  # More statistics
```

## Next Steps

1. **Run the script**: `python3 analyze_camel_spectra.py`
2. **Examine plots**: Look for interesting features
3. **Modify parameters**: Try different resolutions
4. **Compare redshifts**: Analyze multiple snapshots
5. **Add custom analysis**: Implement your own metrics

## Key Differences from Note1.py

| Note1.py | analyze_camel_spectra.py |
|----------|--------------------------|
| Explores snapshot structure | Generates spectra |
| Shows gas particles | Shows Lyman-α absorption |
| 2D spatial projections | 1D velocity-space spectra |
| ~30 seconds | ~5-10 minutes |
| No fake_spectra | Uses fake_spectra |

**Note1.py** = Understand the simulation
**analyze_camel_spectra.py** = Science from the simulation
