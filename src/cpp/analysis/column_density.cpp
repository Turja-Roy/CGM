#include "column_density.h"
#include "constants.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <omp.h>

namespace cgm {
namespace analysis {

namespace {

// One absorber, however it was defined.
struct Absorber {
    double N_primary;  // N_HI under the requested colden_mode
    double N_alt;      // N_HI under the other colden_mode (for side-by-side comparison)
    double peak_tau;
    int n_pixels;
};

// Thread-local accumulator. The four vectors stay index-aligned.
struct AbsorberList {
    std::vector<double> N_primary;
    std::vector<double> N_alt;
    std::vector<double> peak_tau;
    std::vector<int> n_pixels;

    void push(const Absorber& a) {
        N_primary.push_back(a.N_primary);
        N_alt.push_back(a.N_alt);
        peak_tau.push_back(a.peak_tau);
        n_pixels.push_back(a.n_pixels);
    }
    size_t size() const { return N_primary.size(); }
};

// Reduce pixels [start, end) of one sightline to a single absorber.
// The max accumulation is kept in float, exactly as the original code did, so
// COLDEN_MAX results stay bit-identical to the pre-flag implementation.
template <typename RowXpr>
Absorber reduce_feature(const RowXpr& tau_line,
                        const float* colden_line,
                        int start, int end,
                        double velocity_spacing,
                        int colden_mode) {
    Absorber a;
    a.n_pixels = end - start;

    float peak = 0;
    for (int k = start; k < end; ++k) {
        peak = std::max(peak, tau_line(k));
    }
    a.peak_tau = peak;

    if (colden_line) {
        float max_colden = 0;
        double sum_colden = 0;
        for (int k = start; k < end; ++k) {
            max_colden = std::max(max_colden, colden_line[k]);
            sum_colden += colden_line[k];
        }
        if (colden_mode == COLDEN_SUM) {
            a.N_primary = sum_colden;
            a.N_alt = max_colden;
        } else {
            a.N_primary = max_colden;
            a.N_alt = sum_colden;
        }
    } else {
        // No colden array: fall back to converting the integrated tau.
        double tau_sum = 0;
        for (int k = start; k < end; ++k) {
            tau_sum += tau_line(k);
        }
        a.N_primary = constants::TAU_TO_COLDEN_CONSTANT * tau_sum * velocity_spacing;
        a.N_alt = a.N_primary;
    }
    return a;
}

}  // namespace

// Version with raw pointer - properly handles row-major numpy arrays
ColumnDensityResult compute_column_density_distribution(
    const Eigen::Ref<const Eigen::ArrayXXf>& tau,
    double velocity_spacing,
    float threshold,
    const float* colden_data,
    int colden_rows,
    int colden_cols,
    double redshift,
    double box_size_ckpc_h,
    double hubble,
    double omega_m,
    const CddfOptions& options) {

    const int n_sightlines = tau.rows();
    const int n_pixels = tau.cols();

    // Check if colden is valid and matches expected dimensions
    const bool has_colden = (colden_data != nullptr && colden_rows == n_sightlines && colden_cols == n_pixels);

    if (options.absorber_mode != ABSORBER_THRESHOLD &&
        options.absorber_mode != ABSORBER_SIGHTLINE &&
        options.absorber_mode != ABSORBER_CELLS) {
        throw std::invalid_argument("absorber_mode must be 0 (threshold), 1 (sightline) or 2 (cells)");
    }
    if (options.absorber_mode != ABSORBER_THRESHOLD && !has_colden) {
        // The tau fallback is only meaningful for a genuine tau feature; summing
        // it over a whole sightline or an arbitrary cell is not a column density.
        throw std::invalid_argument(
            "absorber_mode 1 (sightline) and 2 (cells) require a colden array matching tau's shape");
    }
    if (options.n_bins <= 0) {
        throw std::invalid_argument("n_bins must be positive");
    }
    if (!(options.log_N_max > options.log_N_min)) {
        throw std::invalid_argument("log_N_max must exceed log_N_min");
    }

    // Cell width in pixels for ABSORBER_CELLS. Mirrors fake_spectra
    // spectra.py:1116: cbins = max(round(close/dvbin), 1), and the trailing
    // partial cell is dropped by the integer division below.
    int cell_pixels = 1;
    if (options.absorber_mode == ABSORBER_CELLS) {
        if (!(velocity_spacing > 0)) {
            throw std::invalid_argument("absorber_mode 2 (cells) requires a positive velocity_spacing");
        }
        cell_pixels = std::max(static_cast<int>(std::lround(options.cell_dv / velocity_spacing)), 1);
    }

    // Per-sightline feature detection is independent: parallelize across
    // sightlines into thread-local lists, then concatenate. The CDDF histogram
    // and power-law fit are order-independent, so the result is unchanged.
    const int n_threads = omp_get_max_threads();
    std::vector<AbsorberList> tl_abs(n_threads);
    std::vector<long long> tl_features(n_threads, 0);

    #pragma omp parallel
    {
        AbsorberList& local = tl_abs[omp_get_thread_num()];
        long long& local_features = tl_features[omp_get_thread_num()];

        #pragma omp for schedule(dynamic, 64)
        for (int i = 0; i < n_sightlines; ++i) {
        const auto& tau_line = tau.row(i);
        const float* colden_line = nullptr;
        if (has_colden) {
            colden_line = &colden_data[i * n_pixels];  // Row-major access from Python
        }

        if (options.absorber_mode == ABSORBER_SIGHTLINE) {
            Absorber a = reduce_feature(tau_line, colden_line, 0, n_pixels,
                                        velocity_spacing, options.colden_mode);
            ++local_features;
            if (a.N_primary > options.min_N_gate) {
                local.push(a);
            }
            continue;
        }

        if (options.absorber_mode == ABSORBER_CELLS) {
            const int n_cells = n_pixels / cell_pixels;  // trailing partial cell dropped
            for (int c = 0; c < n_cells; ++c) {
                Absorber a = reduce_feature(tau_line, colden_line, c * cell_pixels,
                                            (c + 1) * cell_pixels, velocity_spacing,
                                            options.colden_mode);
                ++local_features;
                if (a.N_primary > options.min_N_gate) {
                    local.push(a);
                }
            }
            continue;
        }

        // ABSORBER_THRESHOLD: contiguous runs of tau > threshold.
        bool in_feature = false;
        int feature_start = 0;

        for (int j = 0; j < n_pixels; ++j) {
            bool absorbing = tau_line(j) > threshold;

            if (absorbing && !in_feature) {
                in_feature = true;
                feature_start = j;
            } else if (!absorbing && in_feature) {
                Absorber a = reduce_feature(tau_line, colden_line, feature_start, j,
                                            velocity_spacing, options.colden_mode);
                ++local_features;
                if (a.N_primary > options.min_N_gate) {
                    local.push(a);
                }
                in_feature = false;
            }
        }

        if (in_feature) {
            Absorber a = reduce_feature(tau_line, colden_line, feature_start, n_pixels,
                                        velocity_spacing, options.colden_mode);
            ++local_features;
            if (a.N_primary > options.min_N_gate) {
                local.push(a);
            }
        }
        }
    }  // end #pragma omp parallel

    AbsorberList absorbers;
    long long n_features_total = 0;
    for (int t = 0; t < n_threads; ++t) {
        const AbsorberList& src = tl_abs[t];
        absorbers.N_primary.insert(absorbers.N_primary.end(), src.N_primary.begin(), src.N_primary.end());
        absorbers.N_alt.insert(absorbers.N_alt.end(), src.N_alt.begin(), src.N_alt.end());
        absorbers.peak_tau.insert(absorbers.peak_tau.end(), src.peak_tau.begin(), src.peak_tau.end());
        absorbers.n_pixels.insert(absorbers.n_pixels.end(), src.n_pixels.begin(), src.n_pixels.end());
        n_features_total += tl_features[t];
    }
    std::vector<double>& column_densities = absorbers.N_primary;

    // Both path lengths, so the caller can see the alternative.
    double dX_comoving_mpc = std::nan("");
    double X_absorption = std::nan("");
    if (!std::isnan(redshift) && !std::isnan(box_size_ckpc_h)) {
        dX_comoving_mpc = box_size_ckpc_h / hubble / 1000.0;
        // X ~ (H0/c) * L_comoving * (1+z)^2. box_size_ckpc_h is used raw: the h
        // cancels against the h in H100. See constants.h and fake_spectra
        // unitsystem.py:31-42.
        X_absorption = constants::H100_INV_S / constants::LIGHT_CM_S *
                       box_size_ckpc_h * constants::KPC_IN_CM *
                       (1.0 + redshift) * (1.0 + redshift);
    }

    double dX;
    if (!std::isnan(redshift) && !std::isnan(box_size_ckpc_h)) {
        dX = (options.dx_mode == DX_ABSORPTION_DISTANCE) ? X_absorption : dX_comoving_mpc;
    } else {
        dX = 1.0;
    }

    const int n_bins = options.n_bins;
    const double log_N_min = options.log_N_min;
    const double log_N_max = options.log_N_max;

    Eigen::VectorXd bins = Eigen::VectorXd::LinSpaced(n_bins + 1, log_N_min, log_N_max);
    for (int i = 0; i <= n_bins; ++i) {
        bins(i) = std::pow(10.0, bins(i));
    }

    ColumnDensityResult result;
    result.n_sightlines = n_sightlines;
    result.dX = dX;
    result.redshift = redshift;
    result.dX_comoving_mpc = dX_comoving_mpc;
    result.X_absorption = X_absorption;
    result.n_features_total = static_cast<int>(n_features_total);
    result.used_colden = has_colden;
    result.options = options;

    if (column_densities.empty()) {
        result.N_HI = Eigen::VectorXd(0);
        result.N_HI_alt = Eigen::VectorXd(0);
        result.peak_tau = Eigen::VectorXd(0);
        result.feature_pixels = Eigen::VectorXi(0);
        result.counts = Eigen::VectorXi::Zero(n_bins);
        result.bins = bins;
        result.bin_centers = (bins.head(n_bins) + bins.tail(n_bins)) / 2.0;
        result.f_N = Eigen::VectorXd::Zero(n_bins);
        result.beta_fit = std::nan("");
        result.beta_fit_err = std::nan("");
        result.beta_fit_weighted = std::nan("");
        result.beta_fit_weighted_err = std::nan("");
        result.beta_fit_n_bins = 0;
        result.n_absorbers = 0;
        return result;
    }

    result.N_HI = Eigen::VectorXd::Map(column_densities.data(), column_densities.size());
    result.N_HI_alt = Eigen::VectorXd::Map(absorbers.N_alt.data(), absorbers.N_alt.size());
    result.peak_tau = Eigen::VectorXd::Map(absorbers.peak_tau.data(), absorbers.peak_tau.size());
    result.feature_pixels = Eigen::VectorXi::Map(absorbers.n_pixels.data(), absorbers.n_pixels.size());
    result.n_absorbers = column_densities.size();

    Eigen::VectorXi counts = Eigen::VectorXi::Zero(n_bins);
    for (double N : column_densities) {
        if (N >= bins(0) && N <= bins(n_bins)) {
            double log_N = std::log10(N);
            int bin_idx = static_cast<int>((log_N - log_N_min) / (log_N_max - log_N_min) * n_bins);
            bin_idx = std::clamp(bin_idx, 0, n_bins - 1);
            counts(bin_idx)++;
        }
    }

    Eigen::VectorXd bin_centers = (bins.head(n_bins) + bins.tail(n_bins)) / 2.0;

    // NORM_PER_DEX divides by the bin width in dex, NORM_PER_LINEAR_N by the
    // linear width dN = N_{i+1} - N_i. Only the latter gives f(N) = dn/dN dX,
    // which is the quantity the literature beta refers to.
    Eigen::VectorXd bin_width(n_bins);
    if (options.norm_mode == NORM_PER_LINEAR_N) {
        bin_width = bins.tail(n_bins) - bins.head(n_bins);
    } else {
        bin_width = Eigen::VectorXd::Constant(n_bins, (log_N_max - log_N_min) / n_bins);
    }

    Eigen::VectorXd f_N(n_bins);
    double norm_factor = n_sightlines * dX;
    for (int i = 0; i < n_bins; ++i) {
        f_N(i) = static_cast<double>(counts(i)) / (norm_factor * bin_width(i));
    }

    result.bins = bins;
    result.bin_centers = bin_centers;
    result.counts = counts;
    result.f_N = f_N;

    // Fit power law: f(N) = A * N^(-beta) over the requested log N range.
    double beta_fit = std::nan("");

    const double fit_N_min = std::pow(10.0, options.fit_log_N_min);
    const double fit_N_max = std::pow(10.0, options.fit_log_N_max);

    std::vector<double> log_N_fit;
    std::vector<double> log_f_fit;
    std::vector<double> weight_fit;

    for (int i = 0; i < n_bins; ++i) {
        if (bin_centers(i) > fit_N_min && bin_centers(i) < fit_N_max && counts(i) > 0) {
            double log_N = std::log10(bin_centers(i));
            double f_val = f_N(i);
            if (f_val > 0 && std::isfinite(log_N)) {
                log_N_fit.push_back(log_N);
                // No epsilon floor: f_val > 0 is already guaranteed, and an
                // absolute floor destroys the fit under NORM_PER_LINEAR_N, where
                // f(N) is of order 1e-12 cm^2.
                log_f_fit.push_back(std::log10(f_val));
                // Poisson error on log10 f is (1/ln10)/sqrt(counts), so the
                // inverse-variance weight is counts * ln(10)^2.
                const double ln10 = std::log(10.0);
                weight_fit.push_back(static_cast<double>(counts(i)) * ln10 * ln10);
            }
        }
    }

    double beta_fit_err = std::nan("");
    double beta_fit_weighted = std::nan("");
    double beta_fit_weighted_err = std::nan("");
    const int n_fit = static_cast<int>(log_N_fit.size());

    if (n_fit > 5) {
        // Unweighted fit: the production beta, matching fake_spectra bin-for-bin.
        double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
        for (int i = 0; i < n_fit; ++i) {
            sum_x += log_N_fit[i];
            sum_y += log_f_fit[i];
            sum_xy += log_N_fit[i] * log_f_fit[i];
            sum_xx += log_N_fit[i] * log_N_fit[i];
        }

        double denominator = n_fit * sum_xx - sum_x * sum_x;
        if (std::abs(denominator) > 1e-10) {
            double slope = (n_fit * sum_xy - sum_x * sum_y) / denominator;
            double intercept = (sum_y - slope * sum_x) / n_fit;
            beta_fit = -slope;

            // Slope standard error from the scatter of the bins about the line,
            // sqrt( SSR/(n-2) / Sxx ).
            if (n_fit > 2) {
                double ssr = 0.0;
                for (int i = 0; i < n_fit; ++i) {
                    double resid = log_f_fit[i] - (slope * log_N_fit[i] + intercept);
                    ssr += resid * resid;
                }
                double x_mean = sum_x / n_fit;
                double sxx = 0.0;
                for (int i = 0; i < n_fit; ++i) {
                    double dx = log_N_fit[i] - x_mean;
                    sxx += dx * dx;
                }
                if (sxx > 0.0) {
                    beta_fit_err = std::sqrt((ssr / (n_fit - 2)) / sxx);
                }
            }
        }

        // Poisson-weighted fit; see column_density.h.
        double w = 0, wx = 0, wy = 0, wxy = 0, wxx = 0;
        for (int i = 0; i < n_fit; ++i) {
            const double wi = weight_fit[i];
            w += wi;
            wx += wi * log_N_fit[i];
            wy += wi * log_f_fit[i];
            wxy += wi * log_N_fit[i] * log_f_fit[i];
            wxx += wi * log_N_fit[i] * log_N_fit[i];
        }
        double wdenom = w * wxx - wx * wx;
        if (std::abs(wdenom) > 1e-10) {
            double wslope = (w * wxy - wx * wy) / wdenom;
            beta_fit_weighted = -wslope;
            beta_fit_weighted_err = std::sqrt(w / wdenom);
        }
    }

    result.beta_fit = beta_fit;
    result.beta_fit_err = beta_fit_err;
    result.beta_fit_weighted = beta_fit_weighted;
    result.beta_fit_weighted_err = beta_fit_weighted_err;
    result.beta_fit_n_bins = n_fit;

    return result;
}

}
}
