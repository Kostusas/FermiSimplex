#pragma once

#include "lineartetrahedron/types.h"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace lineartetrahedron {

class TightBindingModel {
public:
    TightBindingModel(KeyArray keys, HoppingMatrixArray matrices);

    size_t ndim() const noexcept { return ndim_; }
    size_t ndof() const noexcept { return ndof_; }
    size_t nterms() const noexcept { return nterms_; }
    double reduced_lipschitz_bound() const noexcept { return reduced_lipschitz_bound_; }
    std::span<const double> hopping_spectral_norms() const noexcept { return hopping_norms_; }
    std::span<const double> global_derivative_bounds() const noexcept {
        return global_derivative_bounds_;
    }
    std::span<const double> hessian_bounds() const noexcept { return hessian_bounds_; }
    double hessian_bound(size_t axis, size_t coordinate) const noexcept {
        return hessian_bounds_[axis * ndim_ + coordinate];
    }

    nb::ndarray<nb::numpy, std::complex<double>> evaluate_point(PointArray point) const;

    std::vector<std::complex<double>> evaluate_point_raw(const double *point) const;
    std::vector<std::complex<double>> evaluate_derivative_raw(
        const double *point,
        size_t axis
    ) const;
    double derivative_spectral_norm_raw(const double *point, size_t axis) const;
    double derivative_spectral_norm(PointArray point, size_t axis) const;

private:
    size_t ndim_ = 0;
    size_t ndof_ = 0;
    size_t nterms_ = 0;
    double reduced_lipschitz_bound_ = 0.0;
    std::vector<std::int64_t> keys_;
    std::vector<std::complex<double>> matrices_;
    std::vector<double> hopping_norms_;
    std::vector<double> global_derivative_bounds_;
    std::vector<double> hessian_bounds_;
};

}  // namespace lineartetrahedron
