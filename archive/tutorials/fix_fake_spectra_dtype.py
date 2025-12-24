#!/usr/bin/env python3
"""
Fixes for fake_spectra bugs
============================

This script contains monkey-patches for multiple bugs in the fake_spectra package.

BUG 1: dtype issue (float64 vs float32)
----------------------------------------
The fake_spectra package's C extension expects all scalar parameters to be
float32, but the Python code passes float64 scalars. This causes a TypeError:
"One of the data arrays does not have 32-bit float type"

ISSUE DETAILS:
The following parameters are passed as float64 but need to be float32:
- box (simulation box size)
- velfac (velocity factor)
- atime (scale factor)
- tautail (optical depth tail threshold)
- line.lambda_X * 1e-8 (wavelength in cm)
- gamma_X (damping parameter)  
- line.fosc_X (oscillator strength)
- amumass (atomic mass)

SOLUTION:
Convert all scalar parameters to np.float32 before passing to C function.
Note: array parameters (pos, vel, elem_den, temp, hh, cofm) are already float32.

BUG 2: Integer overflow in get_npart()
---------------------------------------
When calculating total particle count with large simulations (>4 billion particles),
the calculation: NumPart_Total + 2**32 * NumPart_Total_HighWord
causes an OverflowError because Python tries to cast 2**32 = 4294967296 to uint32.

ISSUE DETAILS:
Error: "Python integer 4294967296 out of bounds for uint32"
This occurs in abstractsnapshot.py line 228 when loading real CAMELS snapshots.

SOLUTION:
Use int64 for the arithmetic to properly handle 64-bit particle counts.
"""

import numpy as np
import fake_spectra.spectra as fsp
from fake_spectra._spectra_priv import _Particle_Interpolate
import fake_spectra.abstractsnapshot as fas

# Save the original methods
original_do_interpolation_work = fsp.Spectra._do_interpolation_work
original_get_npart_abstract = fas.AbstractSnapshot.get_npart
original_get_npart_hdf5 = fas.HDF5Snapshot.get_npart

def fixed_do_interpolation_work(self, pos, vel, elem_den, temp, hh, amumass, line, get_tau):
    """
    Fixed version of _do_interpolation_work that ensures all scalar parameters
    are float32 before calling _Particle_Interpolate, except cofm which must be float64.
    """
    # Determine gamma_X
    if self.turn_off_selfshield:
        gamma_X = 0
    else:
        gamma_X = line.gamma_X
    
    # Convert all scalar parameters to float32
    box_f32 = np.float32(self.box)
    velfac_f32 = np.float32(self.velfac)
    atime_f32 = np.float32(self.atime)
    lambda_cm_f32 = np.float32(line.lambda_X * 1e-8)  # Angstrom to cm
    gamma_f32 = np.float32(gamma_X)
    fosc_f32 = np.float32(line.fosc_X)
    amumass_f32 = np.float32(amumass)
    tautail_f32 = np.float32(self.tautail)
    
    # Keep cofm as float64 (required by C function for sightline positions)
    cofm_f64 = self.cofm  # Must be float64
    
    # Ensure array parameters are float32 (elem_den can become float64 due to multiplications in _read_particle_data)
    pos_f32 = pos.astype(np.float32) if pos.dtype != np.float32 else pos
    vel_f32 = vel.astype(np.float32) if vel.dtype != np.float32 else vel
    elem_den_f32 = elem_den.astype(np.float32) if elem_den.dtype != np.float32 else elem_den
    temp_f32 = temp.astype(np.float32) if temp.dtype != np.float32 else temp
    hh_f32 = hh.astype(np.float32) if hh.dtype != np.float32 else hh
    
    # Call the C function with float32 scalars and arrays, except cofm which must be float64
    return _Particle_Interpolate(
        get_tau * 1,      # Convert bool to int
        self.nbins,       # int
        self.kernel_int,  # int  
        box_f32,          # float32
        velfac_f32,       # float32
        atime_f32,        # float32
        lambda_cm_f32,    # float32
        gamma_f32,        # float32
        fosc_f32,         # float32
        amumass_f32,      # float32
        tautail_f32,      # float32
        pos_f32,          # float32 array
        vel_f32,          # float32 array
        elem_den_f32,     # float32 array (converted if needed)
        temp_f32,         # float32 array
        hh_f32,           # float32 array
        self.axis,        # int32 array
        cofm_f64          # float64 array (required for sightline positions)
    )

def fixed_get_npart(self):
    """
    Fixed version of get_npart that properly handles 64-bit particle counts.
    
    The original implementation causes an integer overflow when calculating:
    NumPart_Total + 2**32 * NumPart_Total_HighWord
    
    This happens because Python tries to cast the intermediate result to uint32,
    but 2**32 = 4294967296 exceeds uint32 range.
    
    Solution: Use int64 for the calculation to avoid overflow.
    """
    num_part_low = self.get_header_attr("NumPart_Total")
    num_part_high = self.get_header_attr("NumPart_Total_HighWord")
    
    # Convert to int64 to avoid overflow
    num_part_low = np.int64(num_part_low)
    num_part_high = np.int64(num_part_high)
    
    # Calculate total particles with 64-bit arithmetic
    total_particles = num_part_low + np.int64(2**32) * num_part_high
    
    return total_particles

# Apply the monkey patches
fsp.Spectra._do_interpolation_work = fixed_do_interpolation_work
fas.AbstractSnapshot.get_npart = fixed_get_npart
fas.HDF5Snapshot.get_npart = fixed_get_npart  # HDF5Snapshot overrides get_npart, so patch it too
