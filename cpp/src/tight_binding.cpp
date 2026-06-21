#include "lineartetrahedron/tight_binding.h"

#include "lineartetrahedron/linalg.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace lineartetrahedron {
namespace {

using Clock = std::chrono::steady_clock;

std::uint64_t nanoseconds_since(Clock::time_point start) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count()
    );
}

double numerical_lanczos_floor(double scale) {
    return 64.0 * std::numeric_limits<double>::epsilon() * std::max(scale, 1.0);
}

double frobenius_norm(std::span<const std::complex<double>> matrix) {
    auto squared = 0.0;
    for (const auto value : matrix) {
        squared += std::norm(value);
    }
    return std::sqrt(squared);
}

}  // namespace

TightBindingModel::TightBindingModel(KeyArray keys, HoppingMatrixArray matrices) {
    ndim_ = keys.shape(1);
    nterms_ = keys.shape(0);
    if (matrices.shape(0) != nterms_) {
        throw std::runtime_error("TightBindingModel: term axis mismatch");
    }
    if (matrices.shape(1) != matrices.shape(2)) {
        throw std::runtime_error("TightBindingModel: matrices must be square");
    }
    ndof_ = matrices.shape(1);
    keys_.assign(keys.data(), keys.data() + nterms_ * ndim_);
    matrices_.resize(nterms_ * ndof_ * ndof_);
    for (size_t term = 0; term < nterms_; ++term) {
        for (size_t row = 0; row < ndof_; ++row) {
            for (size_t col = 0; col < ndof_; ++col) {
                matrices_[(term * ndof_ + col) * ndof_ + row] =
                    matrices.data()[(term * ndof_ + row) * ndof_ + col];
            }
        }
    }

    hopping_norms_.resize(nterms_);
    global_derivative_bounds_.assign(ndim_, 0.0);
    hessian_bounds_.assign(ndim_ * ndim_, 0.0);
    for (size_t term = 0; term < nterms_; ++term) {
        const auto term_matrix = std::span<const std::complex<double>>(
            matrices_.data() + term * ndof_ * ndof_,
            ndof_ * ndof_
        );
        const auto hopping_norm = matrix_spectral_norm_lanczos(
            term_matrix,
            ndof_,
            numerical_lanczos_floor(frobenius_norm(term_matrix))
        );
        hopping_norms_[term] = hopping_norm;
        for (size_t axis = 0; axis < ndim_; ++axis) {
            const auto axis_key = std::abs(static_cast<double>(keys_[term * ndim_ + axis]));
            global_derivative_bounds_[axis] += axis_key * hopping_norm;
            for (size_t coordinate = 0; coordinate < ndim_; ++coordinate) {
                const auto coordinate_key =
                    std::abs(static_cast<double>(keys_[term * ndim_ + coordinate]));
                hessian_bounds_[axis * ndim_ + coordinate] +=
                    axis_key * coordinate_key * hopping_norm;
            }
        }
    }
    for (const auto bound : global_derivative_bounds_) {
        reduced_lipschitz_bound_ += bound * bound;
    }
    reduced_lipschitz_bound_ = std::sqrt(reduced_lipschitz_bound_);
}

nb::ndarray<nb::numpy, std::complex<double>> TightBindingModel::evaluate_point(
    PointArray point
) const {
    if (point.shape(0) != ndim_) {
        throw std::runtime_error("TightBindingModel: point dimension mismatch");
    }
    const auto h = evaluate_point_raw(point.data());
    std::vector<std::complex<double>> out(ndof_ * ndof_);
    for (size_t row = 0; row < ndof_; ++row) {
        for (size_t col = 0; col < ndof_; ++col) {
            out[row * ndof_ + col] = h[col * ndof_ + row];
        }
    }
    return make_array(std::move(out), {ndof_, ndof_});
}

std::vector<std::complex<double>> TightBindingModel::evaluate_point_raw(
    const double *point
) const {
    std::vector<std::complex<double>> h(ndof_ * ndof_, std::complex<double>(0.0, 0.0));
    for (size_t term = 0; term < nterms_; ++term) {
        double phase_arg = 0.0;
        for (size_t axis = 0; axis < ndim_; ++axis) {
            phase_arg += point[axis] * static_cast<double>(keys_[term * ndim_ + axis]);
        }
        const std::complex<double> phase = std::exp(std::complex<double>(0.0, -phase_arg));
        for (size_t row = 0; row < ndof_; ++row) {
            for (size_t col = 0; col < ndof_; ++col) {
                h[col * ndof_ + row] +=
                    phase * matrices_[(term * ndof_ + col) * ndof_ + row];
            }
        }
    }
    return h;
}

std::vector<std::complex<double>> TightBindingModel::evaluate_derivative_raw(
    const double *point,
    size_t axis
) const {
    if (axis >= ndim_) {
        throw std::runtime_error("TightBindingModel: derivative axis out of bounds");
    }
    std::vector<std::complex<double>> derivative(
        ndof_ * ndof_,
        std::complex<double>(0.0, 0.0)
    );
    for (size_t term = 0; term < nterms_; ++term) {
        double phase_arg = 0.0;
        for (size_t coordinate = 0; coordinate < ndim_; ++coordinate) {
            phase_arg +=
                point[coordinate] * static_cast<double>(keys_[term * ndim_ + coordinate]);
        }
        const auto key = static_cast<double>(keys_[term * ndim_ + axis]);
        if (key == 0.0) {
            continue;
        }
        const auto phase =
            std::complex<double>(0.0, -key) *
            std::exp(std::complex<double>(0.0, -phase_arg));
        for (size_t row = 0; row < ndof_; ++row) {
            for (size_t col = 0; col < ndof_; ++col) {
                derivative[col * ndof_ + row] +=
                    phase * matrices_[(term * ndof_ + col) * ndof_ + row];
            }
        }
    }
    return derivative;
}

double TightBindingModel::derivative_spectral_norm(
    const double *point,
    size_t axis,
    double absolute_uncertainty
) const {
    const auto total_start = Clock::now();
    const auto assembly_start = Clock::now();
    const auto derivative = evaluate_derivative_raw(point, axis);
    const auto assembly_nanoseconds = nanoseconds_since(assembly_start);
    const auto norm = hermitian_spectral_norm_lanczos(
        std::span<const std::complex<double>>(derivative.data(), derivative.size()),
        ndof_,
        absolute_uncertainty
    );
    record_derivative_spectral_norm_timing(
        assembly_nanoseconds,
        nanoseconds_since(total_start)
    );
    return norm;
}

double TightBindingModel::derivative_spectral_norm(PointArray point, size_t axis) const {
    if (point.shape(0) != ndim_) {
        throw std::runtime_error("TightBindingModel: point dimension mismatch");
    }
    if (axis >= ndim_) {
        throw std::runtime_error("TightBindingModel: derivative axis out of bounds");
    }
    return derivative_spectral_norm(
        point.data(),
        axis,
        numerical_lanczos_floor(global_derivative_bounds_[axis])
    );
}

}  // namespace lineartetrahedron
