#include "lineartetrahedron/tight_binding.h"

#include <cmath>
#include <stdexcept>

namespace lineartetrahedron {

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
}

nb::ndarray<nb::numpy, std::complex<double>> TightBindingModel::evaluate_point(
    PointArray point
) const {
    if (point.shape(0) != ndim_) {
        throw std::runtime_error("TightBindingModel: point dimension mismatch");
    }
    const auto h = evaluate_reduced_point_raw(point.data());
    std::vector<std::complex<double>> out(ndof_ * ndof_);
    for (size_t row = 0; row < ndof_; ++row) {
        for (size_t col = 0; col < ndof_; ++col) {
            out[row * ndof_ + col] = h[col * ndof_ + row];
        }
    }
    return make_array(std::move(out), {ndof_, ndof_});
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
