#include "arrays.h"
#include "bindings.h"

#include <lineartetrahedron/hamiltonian.h>

#include <nanobind/stl/shared_ptr.h>

#include <complex>
#include <cstddef>
#include <new>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lineartetrahedron::bindings {
namespace {

std::vector<HoppingTerm> copy_hoppings(
    LatticeVectorArray lattice_vectors,
    HoppingMatrixArray matrices
) {
    const auto term_count = matrices.shape(0);
    const auto ndim = lattice_vectors.shape(1);
    const auto ndof = matrices.shape(1);
    std::vector<HoppingTerm> result;
    result.reserve(term_count);
    for (std::size_t term = 0; term < term_count; ++term) {
        auto hopping = HoppingTerm{};
        hopping.lattice_vector.assign(
            lattice_vectors.data() + term * ndim,
            lattice_vectors.data() + (term + 1) * ndim
        );
        hopping.matrix.resize(ndof * ndof);
        for (std::size_t row = 0; row < ndof; ++row) {
            for (std::size_t column = 0; column < ndof; ++column) {
                hopping.matrix[column * ndof + row] =
                    matrices.data()[(term * ndof + row) * ndof + column];
            }
        }
        result.push_back(std::move(hopping));
    }
    return result;
}

TightBindingModel make_model(
    LatticeVectorArray lattice_vectors,
    HoppingMatrixArray matrices
) {
    if (lattice_vectors.shape(0) == 0 || lattice_vectors.shape(1) == 0) {
        throw std::runtime_error("TightBinding: lattice vectors must be non-empty");
    }
    if (
        matrices.shape(0) != lattice_vectors.shape(0) ||
        matrices.shape(1) == 0 ||
        matrices.shape(1) != matrices.shape(2)
    ) {
        throw std::runtime_error("TightBinding: invalid hopping-matrix shape");
    }
    return TightBindingModel(copy_hoppings(lattice_vectors, matrices));
}

auto evaluate(const TightBindingModel &model, PointArray point) {
    const auto matrix = model.evaluate(
        std::span<const double>(point.data(), point.shape(0))
    );
    std::vector<std::complex<double>> row_major(model.ndof() * model.ndof());
    for (std::size_t row = 0; row < model.ndof(); ++row) {
        for (std::size_t column = 0; column < model.ndof(); ++column) {
            row_major[row * model.ndof() + column] = matrix[column * model.ndof() + row];
        }
    }
    return make_array(std::move(row_major), {model.ndof(), model.ndof()});
}

}  // namespace

void bind_tight_binding(nb::module_ &module) {
    using namespace nb::literals;

    nb::class_<TightBindingModel>(module, "TightBindingModel")
        .def(
            "__init__",
            [](TightBindingModel *self,
               LatticeVectorArray lattice_vectors,
               HoppingMatrixArray matrices) {
                new (self) TightBindingModel(make_model(lattice_vectors, matrices));
            },
            "lattice_vectors"_a,
            "matrices"_a
        )
        .def_prop_ro("ndim", &TightBindingModel::ndim)
        .def_prop_ro("ndof", &TightBindingModel::ndof)
        .def("evaluate", &evaluate, "point"_a);
}

}  // namespace lineartetrahedron::bindings
