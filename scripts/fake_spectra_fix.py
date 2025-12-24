import numpy as np


def apply_fake_spectra_bugfixes():
    """
    Apply bugfixes for fake_spectra to work with Python 3.13.
    
    Fixes:
    1. uint32 overflow in get_npart calculation
    2. float32/float64 type mismatches in C extension
    
    Returns:
        bool: True if bugfixes applied successfully, False if fake_spectra not installed
    """
    try:
        from fake_spectra import abstractsnapshot
        from fake_spectra import spectra
        from fake_spectra._spectra_priv import _Particle_Interpolate as _PI_original
        
        # FIX 1: uint32 overflow
        def get_npart_fixed(self):
            """Get the total number of particles (fixed for uint32 overflow)."""
            npart_total = self.get_header_attr("NumPart_Total").astype(np.int64)
            npart_high = self.get_header_attr("NumPart_Total_HighWord").astype(np.int64)
            return npart_total + (2**32) * npart_high
        
        abstractsnapshot.AbstractSnapshotFactory.get_npart = get_npart_fixed
        abstractsnapshot.HDF5Snapshot.get_npart = get_npart_fixed
        abstractsnapshot.BigFileSnapshot.get_npart = get_npart_fixed
        
        # FIX 2: float32/float64 type casting
        def _do_interpolation_work_fixed(self, pos, vel, elem_den, temp, hh, amumass, line, get_tau):
            """Run the interpolation with proper float32 casting (fixed for Python 3.13)"""
            if self.turn_off_selfshield:
                gamma_X = 0
            else:
                gamma_X = line.gamma_X
            
            # Ensure all scalar parameters are float32
            box = np.float32(self.box)
            velfac = np.float32(self.velfac)
            atime = np.float32(self.atime)
            lambda_X = np.float32(line.lambda_X * 1e-8)
            gamma_X_f32 = np.float32(gamma_X)
            fosc_X = np.float32(line.fosc_X)
            amumass_f32 = np.float32(amumass)
            tautail = np.float32(self.tautail)
            
            # Ensure all array parameters are float32 (except cofm which needs float64)
            pos = np.asarray(pos, dtype=np.float32)
            vel = np.asarray(vel, dtype=np.float32)
            elem_den = np.asarray(elem_den, dtype=np.float32)
            temp = np.asarray(temp, dtype=np.float32)
            hh = np.asarray(hh, dtype=np.float32)
            axis = np.asarray(self.axis, dtype=np.int32)
            cofm = np.asarray(self.cofm, dtype=np.float64)  # cofm must be float64!
            
            return _PI_original(get_tau*1, self.nbins, self.kernel_int, box, velfac, atime, 
                                lambda_X, gamma_X_f32, fosc_X, amumass_f32, tautail, 
                                pos, vel, elem_den, temp, hh, axis, cofm)
        
        spectra.Spectra._do_interpolation_work = _do_interpolation_work_fixed
        
        print("✓ Applied fake_spectra bugfixes for Python 3.13 compatibility")
        return True
        
    except ImportError:
        print("Warning: fake_spectra not installed - skipping bugfixes")
        return False
