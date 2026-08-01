#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "analysis/power_spectrum.h"
#include "analysis/column_density.h"
#include "analysis/line_width.h"
#include "analysis/flux_stats.h"
#include "analysis/temperature_density.h"
#include "analysis/voigt.h"

namespace py = pybind11;

PYBIND11_MODULE(_analysis_cpp, m) {
    m.doc() = "C++ implementation of CGM spectroscopy analysis functions";
    
    m.def("compute_flux_statistics", [](const Eigen::ArrayXXf& tau) {
        auto result = cgm::analysis::compute_flux_statistics(tau);
        py::dict d;
        d["mean_flux"] = result.mean_flux;
        d["median_flux"] = result.median_flux;
        d["std_flux"] = result.std_flux;
        d["min_flux"] = result.min_flux;
        d["max_flux"] = result.max_flux;
        d["mean_tau"] = result.mean_tau;
        d["median_tau"] = result.median_tau;
        d["effective_tau"] = result.effective_tau;
        d["deep_absorption_frac"] = result.deep_absorption_frac;
        d["moderate_absorption_frac"] = result.moderate_absorption_frac;
        d["weak_absorption_frac"] = result.weak_absorption_frac;
        return d;
    }, "Compute basic flux statistics from optical depth",
          py::arg("tau"));
    
    m.def("compute_power_spectrum", 
          [](const Eigen::Ref<const Eigen::ArrayXXf>& flux, double velocity_spacing, int chunk_size) {
        auto result = cgm::analysis::compute_power_spectrum(flux, velocity_spacing, chunk_size);
        py::dict d;
        d["k"] = result.k;
        d["P_k_mean"] = result.P_k_mean;
        d["P_k_std"] = result.P_k_std;
        d["P_k_err"] = result.P_k_err;
        d["n_modes"] = result.n_modes;
        d["mean_flux"] = result.mean_flux;
        d["n_sightlines"] = result.n_sightlines;
        d["velocity_spacing"] = result.velocity_spacing;
        return d;
    },
          "Compute power spectrum from flux array",
          py::arg("flux"),
          py::arg("velocity_spacing"),
          py::arg("chunk_size") = 1000);
    
    m.def("compute_column_density_distribution",
          [](const Eigen::Ref<const Eigen::ArrayXXf>& tau, double velocity_spacing, float threshold,
             py::array_t<float, py::array::c_style | py::array::forcecast> colden,
             double redshift, double box_size_ckpc_h, double hubble, double omega_m,
             int absorber_mode, double cell_dv, int colden_mode, int dx_mode, int norm_mode,
             double log_N_min, double log_N_max, int n_bins,
             double fit_log_N_min, double fit_log_N_max, double min_N_gate) {
        // Get buffer info
        py::buffer_info info = colden.request();

        const float* colden_data = nullptr;
        int colden_rows = 0, colden_cols = 0;

        if (info.size > 0) {
            colden_data = static_cast<const float*>(info.ptr);
            colden_rows = info.shape[0];
            colden_cols = info.shape[1];
        }

        cgm::analysis::CddfOptions opts;
        opts.absorber_mode = absorber_mode;
        opts.cell_dv = cell_dv;
        opts.colden_mode = colden_mode;
        opts.dx_mode = dx_mode;
        opts.norm_mode = norm_mode;
        opts.log_N_min = log_N_min;
        opts.log_N_max = log_N_max;
        opts.n_bins = n_bins;
        opts.fit_log_N_min = fit_log_N_min;
        opts.fit_log_N_max = fit_log_N_max;
        opts.min_N_gate = min_N_gate;

        // Call C++ function with raw pointer
        auto result = cgm::analysis::compute_column_density_distribution(
            tau, velocity_spacing, threshold, colden_data, colden_rows, colden_cols,
            redshift, box_size_ckpc_h, hubble, omega_m, opts);
        py::dict d;
        d["N_HI"] = result.N_HI;
        d["counts"] = result.counts;
        d["bins"] = result.bins;
        d["bin_centers"] = result.bin_centers;
        d["f_N"] = result.f_N;
        d["beta_fit"] = result.beta_fit;
        d["beta_fit_err"] = result.beta_fit_err;
        d["beta_fit_weighted"] = result.beta_fit_weighted;
        d["beta_fit_weighted_err"] = result.beta_fit_weighted_err;
        d["beta_fit_n_bins"] = result.beta_fit_n_bins;
        d["n_absorbers"] = result.n_absorbers;
        d["n_sightlines"] = result.n_sightlines;
        d["dX"] = result.dX;
        d["redshift"] = result.redshift;
        d["N_HI_alt"] = result.N_HI_alt;
        d["peak_tau"] = result.peak_tau;
        d["feature_pixels"] = result.feature_pixels;
        d["dX_comoving_mpc"] = result.dX_comoving_mpc;
        d["X_absorption"] = result.X_absorption;
        d["n_features_total"] = result.n_features_total;
        d["used_colden"] = result.used_colden;
        d["absorber_mode"] = result.options.absorber_mode;
        d["cell_dv"] = result.options.cell_dv;
        d["colden_mode"] = result.options.colden_mode;
        d["dx_mode"] = result.options.dx_mode;
        d["norm_mode"] = result.options.norm_mode;
        d["log_N_min"] = result.options.log_N_min;
        d["log_N_max"] = result.options.log_N_max;
        d["n_bins"] = result.options.n_bins;
        d["fit_log_N_min"] = result.options.fit_log_N_min;
        d["fit_log_N_max"] = result.options.fit_log_N_max;
        d["min_N_gate"] = result.options.min_N_gate;
        return d;
    },
          "Compute column density distribution function.\n"
          "absorber_mode: 0 = contiguous tau > threshold run (default), 1 = whole sightline\n"
          "  (fake_spectra line=True), 2 = fixed cell_dv km/s cells (fake_spectra line=False).\n"
          "colden_mode: 0 = max over the feature (default), 1 = sum.\n"
          "dx_mode: 0 = comoving Mpc box length (default), 1 = absorption distance X(z).\n"
          "norm_mode: 0 = per dex (default), 1 = per linear dN.\n"
          "All defaults reproduce the historical production behaviour.",
          py::arg("tau"),
          py::arg("velocity_spacing"),
          py::arg("threshold") = 0.5f,
          py::arg("colden") = Eigen::ArrayXXf(),
          py::arg("redshift") = std::nan(""),
          py::arg("box_size_ckpc_h") = std::nan(""),
          py::arg("hubble") = 0.6774,
          py::arg("omega_m") = 0.3089,
          py::arg("absorber_mode") = 0,
          py::arg("cell_dv") = 50.0,
          py::arg("colden_mode") = 0,
          py::arg("dx_mode") = 0,
          py::arg("norm_mode") = 0,
          py::arg("log_N_min") = 12.0,
          py::arg("log_N_max") = 22.0,
          py::arg("n_bins") = 50,
          py::arg("fit_log_N_min") = 12.0,
          py::arg("fit_log_N_max") = 14.4771212547196624,
          py::arg("min_N_gate") = 1e12);
    
    m.def("compute_line_width_distribution", 
          [](const Eigen::Ref<const Eigen::ArrayXXf>& tau, double velocity_spacing, float threshold,
             const Eigen::ArrayXXf& colden) {
        // Always pass nullptr since colden handling is broken in binding
        auto result = cgm::analysis::compute_line_width_distribution(tau, velocity_spacing, threshold, nullptr);
        py::dict d;
        d["N_HI"] = result.N_HI;
        d["b_params"] = result.b_params;
        d["temperatures"] = result.temperatures;
        d["b_median"] = result.b_median;
        d["b_mean"] = result.b_mean;
        d["b_std"] = result.b_std;
        d["n_absorbers"] = result.n_absorbers;
        return d;
    },
          "Compute line width (b-parameter) distribution",
          py::arg("tau"),
          py::arg("velocity_spacing"),
          py::arg("threshold") = 0.5f,
          py::arg("colden") = Eigen::ArrayXXf());
    
    m.def("compute_temperature_density_relation", 
          [](const Eigen::Ref<const Eigen::ArrayXXf>& temperature,
             const Eigen::Ref<const Eigen::ArrayXXf>& density,
             const Eigen::Ref<const Eigen::ArrayXXf>& tau, float min_tau) {
        auto result = cgm::analysis::compute_temperature_density_relation(temperature, density, tau, min_tau);
        py::dict d;
        d["temperature"] = result.temperature;
        d["density"] = result.density;
        d["log_T"] = result.log_T;
        d["log_rho"] = result.log_rho;
        d["T0"] = result.T0;
        d["gamma"] = result.gamma;
        d["gamma_err"] = result.gamma_err;
        d["n_pixels"] = result.n_pixels;
        return d;
    },
          "Compute temperature-density relation",
          py::arg("temperature"),
          py::arg("density"),
          py::arg("tau"),
          py::arg("min_tau") = 0.1f);
    
    m.def("compute_voigt_profile",
          [](const Eigen::ArrayXXf& v, double tau_0, double b, double v_center, double damping = 4.7e-4) {
        Eigen::ArrayXXf result(v.size(), 1);
        for (int i = 0; i < v.size(); ++i) {
            result(i) = cgm::analysis::compute_voigt_optical_depth(v(i), tau_0, b, v_center, damping);
        }
        return result;
    },
          "Compute Voigt profile optical depth",
          py::arg("v"),
          py::arg("tau_0"),
          py::arg("b"),
          py::arg("v_center"),
          py::arg("damping") = 4.7e-4);
    
    m.def("compute_tdens_binned",
          [](const Eigen::Ref<const Eigen::VectorXd>& temperature,
             const Eigen::Ref<const Eigen::VectorXd>& density,
             int n_bins) {
        auto result = cgm::analysis::compute_tdens_binned(temperature, density, n_bins);
        py::dict d;
        d["T_median"] = result.T_median;
        d["rho_centers"] = result.rho_centers;
        d["counts_per_bin"] = result.counts_per_bin;
        d["T0"] = result.T0;
        d["gamma"] = result.gamma;
        d["gamma_err"] = result.gamma_err;
        d["T0_err"] = result.T0_err;
        d["n_bins_fit"] = result.n_bins_fit;
        d["rho_mean"] = result.rho_mean;
        d["n_pixels"] = result.n_pixels;
        d["n_bins"] = result.n_bins;
        return d;
    },
          "Compute binned temperature-density statistics (memory-efficient)",
          py::arg("temperature"),
          py::arg("density"),
          py::arg("n_bins") = 30);
}
