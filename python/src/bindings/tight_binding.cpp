#include "arrays.h"
#include "bindings.h"

#include "core/tight_binding.h"

#include <nanobind/stl/shared_ptr.h>

#include <algorithm>
#include <complex>
#include <cmath>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

namespace lineartetrahedron::bindings {

namespace {

std::vector<std::complex<double>> copy_hopping_matrices(HoppingMatrixArray matrices) {
    const size_t nterms = matrices.shape(0);
    const size_t ndof = matrices.shape(1);
    std::vector<std::complex<double>> result(nterms * ndof * ndof);
    for (size_t term = 0; term < nterms; ++term) {
        for (size_t row = 0; row < ndof; ++row) {
            for (size_t col = 0; col < ndof; ++col) {
                result[(term * ndof + col) * ndof + row] =
                    matrices.data()[(term * ndof + row) * ndof + col];
            }
        }
    }
    return result;
}

TightBindingModel make_model(KeyArray keys, HoppingMatrixArray matrices) {
    const size_t ndim = keys.shape(1);
    const size_t nterms = keys.shape(0);
    if (matrices.shape(0) != nterms) {
        throw std::runtime_error("TightBindingModel: term axis mismatch");
    }
    if (matrices.shape(1) != matrices.shape(2)) {
        throw std::runtime_error("TightBindingModel: matrices must be square");
    }
    return TightBindingModel(
        ndim,
        matrices.shape(1),
        copy_keys(keys),
        copy_hopping_matrices(matrices)
    );
}

double largest_symmetric_eigenvalue(std::vector<double> matrix, size_t size) {
    if (size == 0) {
        return 0.0;
    }
    if (size == 1) {
        return matrix[0];
    }

    for (size_t sweep = 0; sweep < 64; ++sweep) {
        auto pivot_row = size_t{0};
        auto pivot_col = size_t{1};
        auto pivot_abs = std::abs(matrix[pivot_row * size + pivot_col]);
        for (size_t row = 0; row < size; ++row) {
            for (size_t col = row + 1; col < size; ++col) {
                const auto value_abs = std::abs(matrix[row * size + col]);
                if (value_abs > pivot_abs) {
                    pivot_abs = value_abs;
                    pivot_row = row;
                    pivot_col = col;
                }
            }
        }
        if (pivot_abs < 1e-14) {
            break;
        }

        const auto app = matrix[pivot_row * size + pivot_row];
        const auto aqq = matrix[pivot_col * size + pivot_col];
        const auto apq = matrix[pivot_row * size + pivot_col];
        const auto angle = 0.5 * std::atan2(2.0 * apq, aqq - app);
        const auto c = std::cos(angle);
        const auto s = std::sin(angle);

        for (size_t index = 0; index < size; ++index) {
            if (index == pivot_row || index == pivot_col) {
                continue;
            }
            const auto aip = matrix[index * size + pivot_row];
            const auto aiq = matrix[index * size + pivot_col];
            const auto new_ip = c * aip - s * aiq;
            const auto new_iq = s * aip + c * aiq;
            matrix[index * size + pivot_row] = new_ip;
            matrix[pivot_row * size + index] = new_ip;
            matrix[index * size + pivot_col] = new_iq;
            matrix[pivot_col * size + index] = new_iq;
        }

        matrix[pivot_row * size + pivot_row] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        matrix[pivot_col * size + pivot_col] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        matrix[pivot_row * size + pivot_col] = 0.0;
        matrix[pivot_col * size + pivot_row] = 0.0;
    }

    auto result = matrix[0];
    for (size_t index = 1; index < size; ++index) {
        result = std::max(result, matrix[index * size + index]);
    }
    return result;
}

class TightBindingHessianBound {
public:
    TightBindingHessianBound(KeyArray keys, HoppingMatrixArray matrices)
        : ndim_(keys.shape(1)),
          ndof_(matrices.shape(1)),
          nterms_(keys.shape(0)),
          keys_(copy_keys(keys)),
          matrices_(copy_hopping_matrices(matrices)) {
        if (ndim_ == 0) {
            throw std::runtime_error("TightBindingHessianBound: dimension must be positive");
        }
        if (nterms_ == 0) {
            throw std::runtime_error("TightBindingHessianBound: at least one term is required");
        }
        if (ndof_ == 0) {
            throw std::runtime_error("TightBindingHessianBound: matrix size must be positive");
        }
        if (matrices.shape(0) != nterms_) {
            throw std::runtime_error("TightBindingHessianBound: term axis mismatch");
        }
        if (matrices.shape(1) != matrices.shape(2)) {
            throw std::runtime_error("TightBindingHessianBound: matrices must be square");
        }
    }

    size_t ndim() const noexcept { return ndim_; }
    size_t ndof() const noexcept { return ndof_; }
    size_t nterms() const noexcept { return nterms_; }

    double evaluate_point(PointArray point) const {
        if (point.shape(0) != ndim_) {
            throw std::runtime_error("TightBindingHessianBound: point dimension mismatch");
        }
        return evaluate_point_raw(point.data());
    }

private:
    double evaluate_point_raw(const double *point) const {
        std::vector<double> coordinate_bounds(ndim_ * ndim_, 0.0);
        std::vector<std::complex<double>> block(ndof_ * ndof_);

        for (size_t left_axis = 0; left_axis < ndim_; ++left_axis) {
            for (size_t right_axis = left_axis; right_axis < ndim_; ++right_axis) {
                std::fill(block.begin(), block.end(), std::complex<double>{0.0, 0.0});
                for (size_t term = 0; term < nterms_; ++term) {
                    double phase_arg = 0.0;
                    for (size_t axis = 0; axis < ndim_; ++axis) {
                        phase_arg +=
                            2.0 * kPi * point[axis] *
                            static_cast<double>(model_key(term, axis));
                    }
                    const auto key_factor =
                        static_cast<double>(model_key(term, left_axis)) *
                        static_cast<double>(model_key(term, right_axis));
                    const auto scale =
                        -4.0 * kPi * kPi * key_factor *
                        std::exp(std::complex<double>(0.0, -phase_arg));
                    for (size_t index = 0; index < block.size(); ++index) {
                        block[index] += scale * model_matrix(term, index);
                    }
                }

                const auto bound = gershgorin_row_sum_bound(block);
                coordinate_bounds[left_axis * ndim_ + right_axis] = bound;
                coordinate_bounds[right_axis * ndim_ + left_axis] = bound;
            }
        }
        return largest_symmetric_eigenvalue(std::move(coordinate_bounds), ndim_);
    }

    std::int64_t model_key(size_t term, size_t axis) const {
        return keys_[term * ndim_ + axis];
    }

    const std::complex<double> &model_matrix(size_t term, size_t index) const {
        return matrices_[term * ndof_ * ndof_ + index];
    }

    double gershgorin_row_sum_bound(const std::vector<std::complex<double>> &matrix) const {
        auto result = 0.0;
        for (size_t row = 0; row < ndof_; ++row) {
            auto row_sum = 0.0;
            for (size_t col = 0; col < ndof_; ++col) {
                row_sum += std::abs(matrix[col * ndof_ + row]);
            }
            result = std::max(result, row_sum);
        }
        return result;
    }

    size_t ndim_ = 0;
    size_t ndof_ = 0;
    size_t nterms_ = 0;
    std::vector<std::int64_t> keys_;
    std::vector<std::complex<double>> matrices_;
};

nb::ndarray<nb::numpy, std::complex<double>> evaluate_point(
    const TightBindingModel &model,
    PointArray point
) {
    if (point.shape(0) != model.ndim()) {
        throw std::runtime_error("TightBindingModel: point dimension mismatch");
    }
    const auto h = model.evaluate_reduced_point_raw(point.data());
    std::vector<std::complex<double>> out(model.ndof() * model.ndof());
    for (size_t row = 0; row < model.ndof(); ++row) {
        for (size_t col = 0; col < model.ndof(); ++col) {
            out[row * model.ndof() + col] = h[col * model.ndof() + row];
        }
    }
    return make_array(std::move(out), {model.ndof(), model.ndof()});
}

}  // namespace

void bind_tight_binding(nb::module_ &m) {
    using namespace nb::literals;

    nb::class_<TightBindingModel>(m, "TightBindingModel")
        .def(
            "__init__",
            [](TightBindingModel *self, KeyArray keys, HoppingMatrixArray matrices) {
                new (self) TightBindingModel(make_model(keys, matrices));
            },
            "keys"_a,
            "matrices"_a
        )
        .def_prop_ro("ndim", &TightBindingModel::ndim)
        .def_prop_ro("ndof", &TightBindingModel::ndof)
        .def_prop_ro("nterms", &TightBindingModel::nterms)
        .def("evaluate_point", &evaluate_point, "point"_a);

    nb::class_<TightBindingHessianBound>(m, "TightBindingHessianBound")
        .def(
            "__init__",
            [](TightBindingHessianBound *self, KeyArray keys, HoppingMatrixArray matrices) {
                new (self) TightBindingHessianBound(keys, matrices);
            },
            "keys"_a,
            "matrices"_a
        )
        .def_prop_ro("ndim", &TightBindingHessianBound::ndim)
        .def_prop_ro("ndof", &TightBindingHessianBound::ndof)
        .def_prop_ro("nterms", &TightBindingHessianBound::nterms)
        .def("evaluate_point", &TightBindingHessianBound::evaluate_point, "point"_a);
}

}  // namespace lineartetrahedron::bindings
