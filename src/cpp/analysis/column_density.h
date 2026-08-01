#ifndef CGM_ANALYSIS_COLUMN_DENSITY_H
#define CGM_ANALYSIS_COLUMN_DENSITY_H

#include <Eigen/Dense>
#include <cmath>
#include <string>

#include "constants.h"

namespace cgm {
namespace analysis {

// How an "absorber" is defined along a sightline.
enum AbsorberMode {
    ABSORBER_THRESHOLD = 0,  // contiguous run of pixels with tau > threshold (ours)
    ABSORBER_SIGHTLINE = 1,  // the whole sightline (fake_spectra column_density_function line=True)
    ABSORBER_CELLS = 2       // fixed cell_dv km/s cells (fake_spectra line=False, close=cell_dv)
};

// How N_HI of one absorber is reduced from the per-pixel colden array.
enum ColdenMode {
    COLDEN_MAX = 0,  // peak pixel (current production behaviour)
    COLDEN_SUM = 1   // sum over pixels (fake_spectra; colden is per-pixel by definition)
};

// What path length the CDDF is normalised by.
enum DxMode {
    DX_COMOVING_MPC = 0,        // box_size_ckpc_h / hubble / 1000 (current; comoving Mpc, z-independent)
    DX_ABSORPTION_DISTANCE = 1  // dimensionless X(z), includes (1+z)^2
};

// What column-density interval the CDDF is normalised by.
enum NormMode {
    NORM_PER_DEX = 0,      // divide by delta_log_N (current; gives N f(N) ln10, so beta_fit = beta_true - 1)
    NORM_PER_LINEAR_N = 1  // divide by linear bin width dN (true f(N) = dn/dN dX)
};

// Every field defaults to the current production behaviour, so an unqualified
// call reproduces the pre-existing numbers bit for bit.
struct CddfOptions {
    int absorber_mode = ABSORBER_THRESHOLD;
    double cell_dv = 50.0;  // km/s, ABSORBER_CELLS only
    int colden_mode = COLDEN_MAX;
    int dx_mode = DX_COMOVING_MPC;
    int norm_mode = NORM_PER_DEX;
    double log_N_min = 12.0;  // histogram range and bin count
    double log_N_max = 22.0;
    int n_bins = 50;
    double fit_log_N_min = 12.0;  // power-law fit range; 14.47712... = log10(3e14)
    double fit_log_N_max = 14.4771212547196624;
    double min_N_gate = constants::COLUMN_DENSITY_MIN;  // drop absorbers at or below this N_HI
};

struct ColumnDensityResult {
    Eigen::VectorXd N_HI;
    Eigen::VectorXi counts;
    Eigen::VectorXd bins;
    Eigen::VectorXd bin_centers;
    Eigen::VectorXd f_N;
    double beta_fit;

    // beta_fit stays the unweighted OLS that matches fake_spectra bin-for-bin. The
    // weighted variant is a systematic: a large gap between the two means the slope
    // is set by the sparsely populated high-N bins.
    double beta_fit_err;           // residual standard error of the unweighted slope
    double beta_fit_weighted;      // Poisson-weighted slope
    double beta_fit_weighted_err;  // its formal error
    int beta_fit_n_bins;           // bins that entered the fit

    int n_absorbers;
    int n_sightlines;
    double dX;
    double redshift;

    // Per-absorber diagnostics, index-aligned with N_HI.
    Eigen::VectorXd N_HI_alt;        // the other colden reduction (sum if colden_mode=max, and vice versa)
    Eigen::VectorXd peak_tau;        // max tau within the absorber
    Eigen::VectorXi feature_pixels;  // width of the absorber in pixels

    // Both path lengths, always computed when redshift/box are finite, so the
    // caller can see what the other normalisation would have given.
    double dX_comoving_mpc;
    double X_absorption;

    int n_features_total;  // absorbers found before the min_N_gate cut
    bool used_colden;      // false => N_HI came from the tau fallback
    CddfOptions options;   // echo, so every output row is self-documenting
};

// New version with raw pointer for proper layout handling
ColumnDensityResult compute_column_density_distribution(
    const Eigen::Ref<const Eigen::ArrayXXf>& tau,
    double velocity_spacing,
    float threshold = 0.5f,
    const float* colden_data = nullptr,
    int colden_rows = 0,
    int colden_cols = 0,
    double redshift = std::nan(""),
    double box_size_ckpc_h = std::nan(""),
    double hubble = 0.6774,
    double omega_m = 0.3089,
    const CddfOptions& options = CddfOptions());

// Old version - deprecated
ColumnDensityResult compute_column_density_distribution(
    const Eigen::Ref<const Eigen::ArrayXXf>& tau,
    double velocity_spacing,
    float threshold,
    const Eigen::Ref<const Eigen::ArrayXXf>* colden,
    double redshift,
    double box_size_ckpc_h,
    double hubble,
    double omega_m);

}
}

#endif
