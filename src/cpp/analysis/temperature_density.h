#ifndef CGM_ANALYSIS_TEMPERATURE_DENSITY_H
#define CGM_ANALYSIS_TEMPERATURE_DENSITY_H

#include <Eigen/Dense>

namespace cgm {
namespace analysis {

struct TemperatureDensityResult {
    Eigen::VectorXd temperature;
    Eigen::VectorXd density;
    Eigen::VectorXd log_T;
    Eigen::VectorXd log_rho;
    double T0;
    double gamma;
    double gamma_err;
    double rho_mean;
    int n_pixels;
};

struct TemperatureDensityBinnedResult {
    Eigen::VectorXd T_median;
    Eigen::VectorXd rho_centers;
    Eigen::VectorXi counts_per_bin;
    double T0;
    double gamma;
    // Formal errors on the T-rho fit, from the scatter of the binned medians.
    double gamma_err;
    double T0_err;      // in K, propagated from the intercept error
    int n_bins_fit;     // populated bins that entered the fit
    double rho_mean;
    int n_pixels;
    int n_bins;
};

TemperatureDensityResult compute_temperature_density_relation(
    const Eigen::Ref<const Eigen::ArrayXXf>& temperature,
    const Eigen::Ref<const Eigen::ArrayXXf>& density,
    const Eigen::Ref<const Eigen::ArrayXXf>& tau,
    float min_tau = 0.1f);

TemperatureDensityBinnedResult compute_tdens_binned(
    const Eigen::Ref<const Eigen::VectorXd>& temperature,
    const Eigen::Ref<const Eigen::VectorXd>& density,
    int n_bins = 30);

}
}

#endif
