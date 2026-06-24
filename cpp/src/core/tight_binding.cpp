#include "core/tight_binding.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace lineartetrahedron {

TightBindingModel::TightBindingModel(
    size_t ndim,
    size_t ndof,
    std::vector<std::int64_t> keys,
    std::vector<std::complex<double>> matrices
) : ndim_(ndim),
    ndof_(ndof),
    nterms_(ndim == 0 ? 0 : keys.size() / ndim),
    keys_(std::move(keys)),
    matrices_(std::move(matrices)) {
    if (ndim_ == 0) {
        throw std::runtime_error("TightBindingModel: dimension must be positive");
    }
    if (ndof_ == 0) {
        throw std::runtime_error("TightBindingModel: matrix size must be positive");
    }
    if (keys_.empty() || keys_.size() % ndim_ != 0) {
        throw std::runtime_error("TightBindingModel: key shape mismatch");
    }
    if (matrices_.size() != nterms_ * ndof_ * ndof_) {
        throw std::runtime_error("TightBindingModel: matrix shape mismatch");
    }
}

std::vector<std::complex<double>> TightBindingModel::evaluate_reduced_point_raw(
    const double *point
) const {
    std::vector<std::complex<double>> h(ndof_ * ndof_, std::complex<double>(0.0, 0.0));
    for (size_t term = 0; term < nterms_; ++term) {
        double phase_arg = 0.0;
        for (size_t axis = 0; axis < ndim_; ++axis) {
            phase_arg +=
                2.0 * kPi * point[axis] * static_cast<double>(keys_[term * ndim_ + axis]);
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

std::vector<std::complex<double>> TightBindingModel::evaluate_point_raw(
    const double *point
) const {
    return evaluate_reduced_point_raw(point);
}

}  // namespace lineartetrahedron
