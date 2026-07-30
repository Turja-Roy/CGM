#ifndef CGM_ANALYSIS_CONSTANTS_H
#define CGM_ANALYSIS_CONSTANTS_H

#include <cmath>

namespace cgm {
namespace constants {

constexpr double TAU_TO_COLDEN_CONSTANT = 8.51e11;
constexpr double LYMAN_ALPHA_WAVELENGTH = 1215.67e-10;
constexpr double DAMPING_PARAMETER = 4.7e-4;
constexpr double B_TO_T_FACTOR = 60.57;  // K / (km/s)^2: T = 60.57 * b^2
constexpr double DEFAULT_HUBBLE = 0.6774;
constexpr double DEFAULT_OMEGA_M = 0.3089;
constexpr double DEFAULT_OMEGA_LAMBDA = 0.6911;
constexpr double DEFAULT_TAU_THRESHOLD = 0.5;
constexpr double COLUMN_DENSITY_MIN = 1e12;

// Absorption distance X(z) = int (1+z)^2 H0/H(z) dz.  For a single box,
// X ~ (H0/c) * L_comoving * (1+z)^2.  H100_INV_S is 100 km/s/Mpc in 1/s, so
// the h in a length given in comoving kpc/h cancels and X is dimensionless.
// Reference: fake_spectra unitsystem.py:31-42 (which uses c = 2.99e10, 0.1%
// low; we use the exact value, so our X is 0.265% larger than theirs).
constexpr double H100_INV_S = 3.2407789e-18;   // 100 km/s/Mpc in 1/s
constexpr double LIGHT_CM_S = 2.99792458e10;   // cm/s
constexpr double KPC_IN_CM = 3.085678e21;      // cm per kpc/h (fake_spectra UnitLength_in_cm)

}
}

#endif
