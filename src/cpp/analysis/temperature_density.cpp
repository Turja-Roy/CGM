#include "temperature_density.h"
#include "constants.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <omp.h>

namespace cgm {
namespace analysis {

TemperatureDensityResult compute_temperature_density_relation(
    const Eigen::Ref<const Eigen::ArrayXXf>& temperature,
    const Eigen::Ref<const Eigen::ArrayXXf>& density,
    const Eigen::Ref<const Eigen::ArrayXXf>& tau,
    float min_tau) {
    
    const float* temp_data = temperature.data();
    const float* dens_data = density.data();
    const float* tau_data = tau.data();
    const int total_size = temperature.size();
    
    std::vector<float> temp_filtered;
    std::vector<float> dens_filtered;
    temp_filtered.reserve(total_size);
    dens_filtered.reserve(total_size);
    
    for (int i = 0; i < total_size; ++i) {
        if (tau_data[i] > min_tau && temp_data[i] > 0 && dens_data[i] > 0 &&
            std::isfinite(temp_data[i]) && std::isfinite(dens_data[i])) {
            temp_filtered.push_back(temp_data[i]);
            dens_filtered.push_back(dens_data[i]);
        }
    }
    
    TemperatureDensityResult result;
    result.n_pixels = temp_filtered.size();
    
    if (result.n_pixels < 100) {
        result.temperature = Eigen::VectorXd(0);
        result.density = Eigen::VectorXd(0);
        result.log_T = Eigen::VectorXd(0);
        result.log_rho = Eigen::VectorXd(0);
        result.T0 = std::nan("");
        result.gamma = std::nan("");
        result.gamma_err = std::nan("");
        result.rho_mean = std::nan("");
        return result;
    }
    
    // Convert float to double
    std::vector<double> temp_dbl(temp_filtered.begin(), temp_filtered.end());
    std::vector<double> dens_dbl(dens_filtered.begin(), dens_filtered.end());
    
    result.temperature = Eigen::VectorXd::Map(temp_dbl.data(), temp_dbl.size());
    result.density = Eigen::VectorXd::Map(dens_dbl.data(), dens_dbl.size());
    
    // Compute median manually
    std::vector<double> sorted_density = dens_dbl;
    std::sort(sorted_density.begin(), sorted_density.end());
    double rho_mean = sorted_density[sorted_density.size() / 2];
    result.rho_mean = rho_mean;
    Eigen::VectorXd overdensity = result.density / rho_mean;
    
    Eigen::VectorXd log_T(result.n_pixels);
    Eigen::VectorXd log_rho(result.n_pixels);
    for (int i = 0; i < result.n_pixels; ++i) {
        log_T(i) = std::log10(result.temperature(i));
        log_rho(i) = std::log10(overdensity(i));
    }
    result.log_T = log_T;
    result.log_rho = log_rho;
    
    const int n_bins = 30;
    double log_rho_min, log_rho_max;
    log_rho.minCoeff(&log_rho_min);
    log_rho.maxCoeff(&log_rho_max);
    Eigen::VectorXd rho_bins = Eigen::VectorXd::LinSpaced(n_bins, log_rho_min, log_rho_max);
    
    // Each density bin is independent: compute medians in parallel into fixed
    // per-bin slots (no shared push_back), then gather valid bins in order.
    // Output is identical to the serial version.
    std::vector<double> bin_T_median(n_bins - 1, std::nan(""));
    std::vector<double> bin_rho_center(n_bins - 1, std::nan(""));
    std::vector<char> bin_valid(n_bins - 1, 0);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n_bins - 1; ++i) {
        std::vector<double> T_in_bin;
        for (int j = 0; j < result.n_pixels; ++j) {
            if (log_rho(j) >= rho_bins(i) && log_rho(j) < rho_bins(i + 1)) {
                T_in_bin.push_back(log_T(j));
            }
        }
        if (T_in_bin.size() > 10) {
            std::sort(T_in_bin.begin(), T_in_bin.end());
            bin_T_median[i] = T_in_bin[T_in_bin.size() / 2];
            bin_rho_center[i] = (rho_bins(i) + rho_bins(i + 1)) / 2;
            bin_valid[i] = 1;
        }
    }

    std::vector<double> T_median;
    std::vector<double> rho_centers;
    for (int i = 0; i < n_bins - 1; ++i) {
        if (bin_valid[i]) {
            T_median.push_back(bin_T_median[i]);
            rho_centers.push_back(bin_rho_center[i]);
        }
    }
    
    if (rho_centers.size() > 5) {
        int n = rho_centers.size();
        double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
        for (int i = 0; i < n; ++i) {
            sum_x += rho_centers[i];
            sum_y += T_median[i];
            sum_xy += rho_centers[i] * T_median[i];
            sum_xx += rho_centers[i] * rho_centers[i];
        }
        double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
        double intercept = (sum_y - slope * sum_x) / n;
        
        result.gamma = slope + 1.0;
        result.T0 = std::pow(10.0, intercept);
        result.gamma_err = std::nan("");
    } else {
        result.T0 = std::nan("");
        result.gamma = std::nan("");
        result.gamma_err = std::nan("");
        result.rho_mean = std::nan("");
    }
    
    return result;
}

TemperatureDensityBinnedResult compute_tdens_binned(
    const Eigen::Ref<const Eigen::VectorXd>& temperature,
    const Eigen::Ref<const Eigen::VectorXd>& density,
    int n_bins) {
    
    const int n_pixels = temperature.size();
    
    TemperatureDensityBinnedResult result;
    result.n_pixels = n_pixels;
    result.n_bins = n_bins;
    result.T_median = Eigen::VectorXd(n_bins);
    result.rho_centers = Eigen::VectorXd(n_bins);
    result.counts_per_bin = Eigen::VectorXi(n_bins);
    
    if (n_pixels < 100) {
        result.T0 = std::nan("");
        result.gamma = std::nan("");
        result.gamma_err = std::nan("");
        result.T0_err = std::nan("");
        result.n_bins_fit = 0;
        result.rho_mean = std::nan("");
        result.T_median.setConstant(std::nan(""));
        result.rho_centers.setConstant(std::nan(""));
        result.counts_per_bin.setZero();
        return result;
    }
    
    std::vector<double> sorted_density(density.data(), density.data() + n_pixels);
    std::sort(sorted_density.begin(), sorted_density.end());
    double rho_mean = sorted_density[sorted_density.size() / 2];
    result.rho_mean = rho_mean;
    
    std::vector<double> log_T(n_pixels);
    std::vector<double> log_rho(n_pixels);
    for (int i = 0; i < n_pixels; ++i) {
        double overdensity = density(i) / rho_mean;
        log_T[i] = std::log10(std::max(temperature(i), 0.1));
        log_rho[i] = std::log10(std::max(overdensity, 0.01));
    }
    
    double log_rho_min = *std::min_element(log_rho.begin(), log_rho.end());
    double log_rho_max = *std::max_element(log_rho.begin(), log_rho.end());
    double bin_width = (log_rho_max - log_rho_min) / n_bins;
    
    std::vector<std::vector<double>> T_in_bins(n_bins);
    
    for (int i = 0; i < n_pixels; ++i) {
        int bin_idx = static_cast<int>((log_rho[i] - log_rho_min) / bin_width);
        if (bin_idx < 0) bin_idx = 0;
        if (bin_idx >= n_bins) bin_idx = n_bins - 1;
        T_in_bins[bin_idx].push_back(log_T[i]);
    }
    
    // Per-bin sort/median is independent; each thread writes its own bin slots
    // (distinct indices, no race). Valid bins gathered in order afterwards.
    std::vector<char> bin_valid(n_bins, 0);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n_bins; ++i) {
        double bin_center = log_rho_min + (i + 0.5) * bin_width;
        result.rho_centers(i) = bin_center;

        if (T_in_bins[i].size() > 10) {
            std::sort(T_in_bins[i].begin(), T_in_bins[i].end());
            result.T_median(i) = T_in_bins[i][T_in_bins[i].size() / 2];
            result.counts_per_bin(i) = static_cast<int>(T_in_bins[i].size());
            bin_valid[i] = 1;
        } else {
            result.T_median(i) = std::nan("");
            result.counts_per_bin(i) = 0;
        }
    }

    std::vector<double> rho_centers_valid;
    std::vector<double> T_median_valid;
    for (int i = 0; i < n_bins; ++i) {
        if (bin_valid[i]) {
            rho_centers_valid.push_back(result.rho_centers(i));
            T_median_valid.push_back(result.T_median(i));
        }
    }
    
    result.gamma_err = std::nan("");
    result.T0_err = std::nan("");
    result.n_bins_fit = static_cast<int>(rho_centers_valid.size());

    if (rho_centers_valid.size() > 5) {
        int n = rho_centers_valid.size();
        double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
        for (int i = 0; i < n; ++i) {
            sum_x += rho_centers_valid[i];
            sum_y += T_median_valid[i];
            sum_xy += rho_centers_valid[i] * T_median_valid[i];
            sum_xx += rho_centers_valid[i] * rho_centers_valid[i];
        }
        double denominator = n * sum_xx - sum_x * sum_x;
        if (std::abs(denominator) > 1e-10) {
            double slope = (n * sum_xy - sum_x * sum_y) / denominator;
            double intercept = (sum_y - slope * sum_x) / n;
            result.gamma = slope + 1.0;
            result.T0 = std::pow(10.0, intercept);

            // Residual errors of the fit to the binned medians -- not the
            // pixel-level scatter, which is much larger.
            if (n > 2) {
                double ssr = 0.0;
                for (int i = 0; i < n; ++i) {
                    double resid = T_median_valid[i]
                                 - (slope * rho_centers_valid[i] + intercept);
                    ssr += resid * resid;
                }
                double x_mean = sum_x / n;
                double sxx = 0.0;
                for (int i = 0; i < n; ++i) {
                    double dx = rho_centers_valid[i] - x_mean;
                    sxx += dx * dx;
                }
                if (sxx > 0.0) {
                    double s2 = ssr / (n - 2);
                    // gamma = slope + 1, so their errors are identical.
                    result.gamma_err = std::sqrt(s2 / sxx);
                    double intercept_err =
                        std::sqrt(s2 * (1.0 / n + x_mean * x_mean / sxx));
                    // T0 = 10^intercept  =>  dT0 = T0 * ln(10) * d(intercept)
                    result.T0_err = result.T0 * std::log(10.0) * intercept_err;
                }
            }
        } else {
            result.gamma = std::nan("");
            result.T0 = std::nan("");
        }
    } else {
        result.gamma = std::nan("");
        result.T0 = std::nan("");
    }

    return result;
}

}
}
