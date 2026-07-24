#include "power_spectrum.h"
#include "constants.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <complex>
#include <omp.h>

#ifdef USE_FFTW
#include <fftw3.h>
#endif

namespace cgm {
namespace analysis {

#ifdef USE_FFTW

// Compute power spectrum using FFTW3
static Eigen::VectorXd compute_fft_power_fftw(
    const Eigen::ArrayXf& flux_contrast, 
    int n_pixels,
    fftwf_plan& plan,
    float* fftw_in,
    fftwf_complex* fftw_out) {
    
    // Copy data to FFTW input buffer
    for (int i = 0; i < n_pixels; ++i) {
        fftw_in[i] = flux_contrast(i);
    }
    
    // Execute FFT
    fftwf_execute(plan);
    
    // Compute power spectrum (only first n_pixels/2 + 1 values are unique for real input)
    int n_k = n_pixels / 2 + 1;
    Eigen::VectorXd power(n_k);
    for (int i = 0; i < n_k; ++i) {
        float real = fftw_out[i][0];
        float imag = fftw_out[i][1];
        power(i) = (real * real + imag * imag) / n_pixels;
    }
    
    return power;
}

#endif

PowerSpectrumResult compute_power_spectrum(
    const Eigen::Ref<const Eigen::ArrayXXf>& flux,
    double velocity_spacing,
    int chunk_size) {
    
    const int n_sightlines = flux.rows();
    const int n_pixels = flux.cols();
    
    // Double accumulator: float32 sum saturates near 2^24 for ~1e8 pixels.
    double mean_flux = flux.cast<double>().mean();
    if (mean_flux <= 0) {
        throw std::invalid_argument("Mean flux must be positive");
    }
    
    int n_k = n_pixels / 2 + 1;
    Eigen::VectorXd k(n_k);
    for (int i = 0; i < n_k; ++i) {
        k(i) = 2.0 * M_PI * i / (n_pixels * velocity_spacing);
    }
    
    std::vector<double> power_sum(n_k, 0.0);
    std::vector<double> power_sum_sq(n_k, 0.0);
    
#ifdef USE_FFTW
    // Note: FFTW threading disabled - using OpenMP for parallelism instead
    float* fftw_in = fftwf_alloc_real(n_pixels);
    fftwf_complex* fftw_out = fftwf_alloc_complex(n_k);
    
    fftwf_plan plan = fftwf_plan_dft_r2c_1d(
        n_pixels, fftw_in, fftw_out, FFTW_ESTIMATE);
    
    int n_chunks = (n_sightlines + chunk_size - 1) / chunk_size;
    
    #pragma omp parallel
    {
        std::vector<double> local_power_sum(n_k, 0.0);
        std::vector<double> local_power_sum_sq(n_k, 0.0);
        
        #pragma omp for schedule(dynamic)
        for (int c = 0; c < n_chunks; ++c) {
            int start = c * chunk_size;
            int end = std::min((c + 1) * chunk_size, n_sightlines);
            
            for (int i = start; i < end; ++i) {
                Eigen::ArrayXf delta_F = flux.row(i) / mean_flux - 1.0f;
                
                Eigen::VectorXd power = compute_fft_power_fftw(
                    delta_F, n_pixels, plan, fftw_in, fftw_out);
                
                for (int j = 0; j < n_k; ++j) {
                    double p = power(j);
                    local_power_sum[j] += p;
                    local_power_sum_sq[j] += p * p;
                }
            }
        }
        
        #pragma omp critical
        {
            for (int j = 0; j < n_k; ++j) {
                power_sum[j] += local_power_sum[j];
                power_sum_sq[j] += local_power_sum_sq[j];
            }
        }
    }
    
    // Cleanup FFTW
    fftwf_destroy_plan(plan);
    fftwf_free(fftw_in);
    fftwf_free(fftw_out);
#else
    // Built without FFTW: do NOT fall back to a slow O(n^2) DFT. Raise so the
    // Python wrapper (scripts/analysis_cpp.py) catches it and uses the
    // vectorized scipy power spectrum instead.
    (void)chunk_size;
    (void)mean_flux;
    throw std::runtime_error(
        "C++ power spectrum requires FFTW (built without USE_FFTW); "
        "use the scipy power-spectrum path instead");
#endif
    
    Eigen::VectorXd power_sum_eigen = Eigen::VectorXd::Map(power_sum.data(), n_k);
    Eigen::VectorXd power_sum_sq_eigen = Eigen::VectorXd::Map(power_sum_sq.data(), n_k);
    
    Eigen::VectorXd P_k_mean = (power_sum_eigen / n_sightlines) * velocity_spacing;
    Eigen::VectorXd mean_power = power_sum_eigen / n_sightlines;
    Eigen::VectorXd mean_power_sq = power_sum_sq_eigen / n_sightlines;
    
    Eigen::VectorXd variance(n_k);
    for (int j = 0; j < n_k; ++j) {
        double mp = mean_power(j);
        double mp2 = mean_power_sq(j);
        double v = mp2 - mp * mp;
        variance(j) = (v > 0) ? v : 0;
    }
    
    Eigen::VectorXd P_k_std(n_k);
    for (int j = 0; j < n_k; ++j) {
        P_k_std(j) = std::sqrt(variance(j)) * velocity_spacing;
    }
    
    Eigen::VectorXd P_k_err = P_k_std / std::sqrt(n_sightlines);
    Eigen::VectorXd n_modes = Eigen::VectorXd::Ones(n_k) * n_sightlines;
    
    PowerSpectrumResult result;
    result.k = k;
    result.P_k_mean = P_k_mean;
    result.P_k_std = P_k_std;
    result.P_k_err = P_k_err;
    result.n_modes = n_modes;
    result.mean_flux = mean_flux;
    result.n_sightlines = n_sightlines;
    result.velocity_spacing = velocity_spacing;
    
    return result;
}

}
}
