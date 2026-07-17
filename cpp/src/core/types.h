#pragma once

#include <complex>
#include <cstdint>
#include <vector>

namespace lineartetrahedron {

constexpr double kPi = 3.141592653589793238462643383279502884;

enum class InconclusiveChargeErrorMode {
    Projected,
    Conservative,
};

struct ChargeValue {
    double charge = 0.0;
    double derivative = 0.0;
    double certificate_error = 0.0;
    double visible_cut_error = 0.0;
    double visible_cut_inconclusive_style_error = 0.0;
    std::int64_t visible_cut_count = 0;
    double inconclusive_error = 0.0;
    std::int64_t inconclusive_count = 0;

    ChargeValue &operator+=(const ChargeValue &other) noexcept {
        charge += other.charge;
        derivative += other.derivative;
        certificate_error += other.certificate_error;
        visible_cut_error += other.visible_cut_error;
        visible_cut_inconclusive_style_error += other.visible_cut_inconclusive_style_error;
        visible_cut_count += other.visible_cut_count;
        inconclusive_error += other.inconclusive_error;
        inconclusive_count += other.inconclusive_count;
        return *this;
    }

    ChargeValue &operator-=(const ChargeValue &other) noexcept {
        charge -= other.charge;
        derivative -= other.derivative;
        certificate_error -= other.certificate_error;
        visible_cut_error -= other.visible_cut_error;
        visible_cut_inconclusive_style_error -= other.visible_cut_inconclusive_style_error;
        visible_cut_count -= other.visible_cut_count;
        inconclusive_error -= other.inconclusive_error;
        inconclusive_count -= other.inconclusive_count;
        return *this;
    }
};

struct ChargeIntegrateResult {
    double charge = 0.0;
    double charge_error = 0.0;
    double dcharge_dmu = 0.0;
    double visible_cut_error = 0.0;
    double visible_cut_inconclusive_style_error = 0.0;
    std::int64_t visible_cut_count = 0;
    double inconclusive_error = 0.0;
    std::int64_t inconclusive_count = 0;
    std::int64_t work = 0;
    std::int64_t refinements = 0;
    std::int64_t n_active_simplices = 0;
    std::int64_t n_active_vertices = 0;
    bool converged = false;
};

struct DensityIntegrateResult {
    std::vector<std::complex<double>> estimate;
    std::vector<double> error_vector;
    double error_scalar = 0.0;
    std::int64_t work = 0;
    std::int64_t refinements = 0;
    std::int64_t n_active_simplices = 0;
    std::int64_t n_active_vertices = 0;
    bool converged = false;
};

struct FermiSurfaceResult {
    std::vector<double> points;
    std::vector<std::int64_t> cells;
    std::vector<std::int64_t> state_band_indices;
    std::vector<double> state_eigenvalues;
    std::vector<std::complex<double>> state_eigenvectors;
    size_t ndim = 0;
    size_t ndof = 0;
    bool has_states = false;
    bool converged = false;
    std::int64_t n_cut_simplices = 0;
    std::int64_t n_feature_size_simplices = 0;
    std::int64_t n_unresolved_simplices = 0;
    double min_feature_size = 0.0;
};

}  // namespace lineartetrahedron
