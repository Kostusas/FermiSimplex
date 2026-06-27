#include "arrays.h"
#include "bindings.h"
#include "python_hamiltonian.h"

#include "fermi_surface/fermi_surface.h"

#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <cstdint>
#include <utility>

namespace lineartetrahedron::bindings {

namespace {

FermiSurfaceResult fermi_surface_callable(
    nb::object callable,
    size_t ndim,
    size_t ndof,
    double mu,
    double min_feature_size,
    std::int64_t max_diagonalizations,
    double margin,
    double tol,
    bool return_states
) {
    return fermi_surface(
        make_python_hamiltonian_model(std::move(callable), ndim, ndof),
        mu,
        min_feature_size,
        max_diagonalizations,
        margin,
        tol,
        return_states
    );
}

FermiSurfaceResult fermi_surface_tight_binding(
    std::shared_ptr<TightBindingModel> model,
    double mu,
    double min_feature_size,
    std::int64_t max_diagonalizations,
    double margin,
    double tol,
    bool return_states
) {
    return fermi_surface(
        std::static_pointer_cast<const HamiltonianModel>(std::move(model)),
        mu,
        min_feature_size,
        max_diagonalizations,
        margin,
        tol,
        return_states
    );
}

}  // namespace

void bind_fermi_surface(nb::module_ &m) {
    using namespace nb::literals;

    nb::class_<FermiSurfaceResult>(m, "FermiSurfaceResult")
        .def(
            "points_array",
            [](const FermiSurfaceResult &self) {
                return make_array(
                    std::vector<double>(self.points),
                    {self.points.size() / self.ndim, self.ndim}
                );
            }
        )
        .def(
            "cells_array",
            [](const FermiSurfaceResult &self) {
                return make_array(
                    std::vector<std::int64_t>(self.cells),
                    {self.cells.size() / self.ndim, self.ndim}
                );
            }
        )
        .def(
            "state_band_indices_array",
            [](const FermiSurfaceResult &self) {
                return make_array(
                    std::vector<std::int64_t>(self.state_band_indices),
                    {self.state_band_indices.size()}
                );
            }
        )
        .def(
            "state_eigenvalues_array",
            [](const FermiSurfaceResult &self) {
                return make_array(
                    std::vector<double>(self.state_eigenvalues),
                    {self.state_eigenvalues.size()}
                );
            }
        )
        .def(
            "state_eigenvectors_array",
            [](const FermiSurfaceResult &self) {
                return make_array(
                    std::vector<std::complex<double>>(self.state_eigenvectors),
                    {self.state_eigenvectors.size() / self.ndof, self.ndof}
                );
            }
        )
        .def_prop_ro("ndim", [](const FermiSurfaceResult &self) { return self.ndim; })
        .def_prop_ro("ndof", [](const FermiSurfaceResult &self) { return self.ndof; })
        .def_prop_ro("has_states", [](const FermiSurfaceResult &self) { return self.has_states; })
        .def_prop_ro("converged", [](const FermiSurfaceResult &self) { return self.converged; })
        .def_prop_ro("n_cut_simplices", [](const FermiSurfaceResult &self) { return self.n_cut_simplices; })
        .def_prop_ro("n_feature_size_simplices", [](const FermiSurfaceResult &self) { return self.n_feature_size_simplices; })
        .def_prop_ro("n_unresolved_simplices", [](const FermiSurfaceResult &self) { return self.n_unresolved_simplices; })
        .def_prop_ro("min_feature_size", [](const FermiSurfaceResult &self) { return self.min_feature_size; });

    m.def(
        "fermi_surface",
        &fermi_surface_tight_binding,
        "model"_a,
        "mu"_a,
        "min_feature_size"_a,
        "max_diagonalizations"_a = -1,
        "margin"_a = 0.0,
        "tol"_a = 1e-14,
        "return_states"_a = false
    );
    m.def(
        "fermi_surface_callable",
        &fermi_surface_callable,
        "callable"_a,
        "ndim"_a,
        "ndof"_a,
        "mu"_a,
        "min_feature_size"_a,
        "max_diagonalizations"_a = -1,
        "margin"_a = 0.0,
        "tol"_a = 1e-14,
        "return_states"_a = false
    );
    m.def(
        "_product_simplex_triangulation_cells",
        &product_simplex_triangulation_cells,
        "negative_count"_a,
        "positive_count"_a
    );
    m.def("_reset_fermi_surface_stats", &reset_fermi_surface_stats);
    m.def(
        "_fermi_surface_stats",
        []() {
            const auto stats = fermi_surface_stats();
            nb::dict result;
            result["evaluated_vertices"] = stats.evaluated_vertices;
            return result;
        }
    );
}

}  // namespace lineartetrahedron::bindings
