#include "arrays.h"
#include "bindings.h"

#include "core/tight_binding.h"

#include <nanobind/stl/shared_ptr.h>

#include <complex>
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
}

}  // namespace lineartetrahedron::bindings
