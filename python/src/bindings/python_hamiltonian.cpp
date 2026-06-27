#include "python_hamiltonian.h"

#include "arrays.h"

#include <Python.h>

#include <complex>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lineartetrahedron::bindings {
namespace {

class PythonHamiltonianModel final : public HamiltonianModel {
public:
    PythonHamiltonianModel(nb::object callable, size_t ndim, size_t ndof)
        : callable_(std::move(callable)), ndim_(ndim), ndof_(ndof) {
        if (ndim_ < 1) {
            throw std::runtime_error("callable Hamiltonian dimension must be positive");
        }
        if (ndof_ < 1) {
            throw std::runtime_error("callable Hamiltonian matrix size must be positive");
        }
    }

    size_t ndim() const noexcept override { return ndim_; }
    size_t ndof() const noexcept override { return ndof_; }

    std::vector<std::complex<double>> evaluate_reduced_point_raw(
        const double *point
    ) const override {
        nb::gil_scoped_acquire gil;
        nb::tuple args = nb::steal<nb::tuple>(
            PyTuple_New(static_cast<Py_ssize_t>(ndim_))
        );
        if (!args.ptr()) {
            nb::raise_python_error();
        }
        for (size_t axis = 0; axis < ndim_; ++axis) {
            PyObject *value = PyFloat_FromDouble(point[axis]);
            if (!value) {
                nb::raise_python_error();
            }
            PyTuple_SET_ITEM(args.ptr(), static_cast<Py_ssize_t>(axis), value);
        }

        const auto matrix = nb::cast<CallbackMatrixArray>(callable_(*args));
        if (matrix.shape(0) != ndof_ || matrix.shape(1) != ndof_) {
            throw std::invalid_argument(
                "callable Hamiltonian returned a matrix with inconsistent shape"
            );
        }

        std::vector<std::complex<double>> h(ndof_ * ndof_);
        for (size_t row = 0; row < ndof_; ++row) {
            for (size_t col = 0; col < ndof_; ++col) {
                h[col * ndof_ + row] = matrix.data()[row * ndof_ + col];
            }
        }
        return h;
    }

private:
    nb::object callable_;
    size_t ndim_;
    size_t ndof_;
};

}  // namespace

std::shared_ptr<const HamiltonianModel> make_python_hamiltonian_model(
    nb::object callable,
    size_t ndim,
    size_t ndof
) {
    return std::make_shared<PythonHamiltonianModel>(std::move(callable), ndim, ndof);
}

}  // namespace lineartetrahedron::bindings
