#include "arrays.h"
#include "bindings.h"

#include "certificate/internal.h"
#include "certificate/simplex_certificate.h"
#include "fermi_surface/fermi_surface.h"
#include "core/vertex_spectra.h"

#include <adaptivesimplex/core/root_mesh.h>

#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <cstdint>
#include <complex>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lineartetrahedron::bindings {

namespace {

using Complex = std::complex<double>;
namespace core = adaptivesimplex::core;

std::vector<Complex> copy_square_matrix(CallbackMatrixArray matrix) {
    if (matrix.shape(0) != matrix.shape(1)) {
        throw std::runtime_error("matrix must be square");
    }
    const size_t size = matrix.shape(0);
    std::vector<Complex> result(size * size);
    for (size_t row = 0; row < size; ++row) {
        for (size_t col = 0; col < size; ++col) {
            result[col * size + row] = matrix.data()[row * size + col];
        }
    }
    return result;
}

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
    return fermi_surface_from_model(
        std::make_shared<PythonHamiltonianModel>(std::move(callable), ndim, ndof),
        mu,
        min_feature_size,
        max_diagonalizations,
        margin,
        tol,
        return_states
    );
}

std::optional<double> root_mesh_certificate_gap_bound(
    std::shared_ptr<TightBindingModel> model,
    double mu,
    double margin,
    double tol,
    double gap_bound_precision
) {
    if (!model) {
        throw std::runtime_error("_root_mesh_certificate_gap_bound: model must not be null");
    }
    auto geometry = core::root_geometry(model->ndim(), model->ndim() == 1 ? 2U : 1U);
    const auto active = geometry.simplices().active_simplices();
    const auto simplex_ids = std::vector<core::SimplexId>(active.begin(), active.end());

    core::VertexCache<VertexSpectra> cache;
    VertexSpectraEvaluator evaluator(model);
    const auto missing = geometry.missing_vertices(
        std::span<const core::SimplexId>(simplex_ids.data(), simplex_ids.size()),
        cache,
        0
    );
    for (const auto vertex_id : missing) {
        const auto reduced_point = geometry.vertices().dyadic_vertex(vertex_id).to_point();
        cache.insert(
            vertex_id,
            evaluator.evaluate_reduced_point(
                std::span<const double>(reduced_point.data(), reduced_point.size())
            )
        );
    }

    std::optional<double> result;
    for (const auto simplex_id : simplex_ids) {
        const auto certificate = simplex_certificate::certify_simplex_gap(
            mu,
            geometry,
            simplex_id,
            cache,
            margin,
            tol,
            gap_bound_precision
        );
        if (certificate.status != simplex_certificate::SimplexCertificateStatus::CertifiedGapped ||
            !certificate.gap_bound.has_value()) {
            return std::nullopt;
        }
        result = result.has_value()
            ? std::min(*result, *certificate.gap_bound)
            : certificate.gap_bound;
    }
    return result;
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
        &fermi_surface,
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
    m.def(
        "_hermitian_min_eigenvalue_lanczos",
        [](CallbackMatrixArray matrix, double absolute_uncertainty) {
            const auto size = static_cast<size_t>(matrix.shape(0));
            const auto values = copy_square_matrix(matrix);
            return simplex_certificate::detail::hermitian_min_eigenvalue_lanczos(
                std::span<const Complex>(values.data(), values.size()),
                size,
                absolute_uncertainty
            );
        },
        "matrix"_a,
        "absolute_uncertainty"_a
    );
    m.def(
        "_generalized_hermitian_min_eigenvalue_lanczos",
        [](CallbackMatrixArray matrix, CallbackMatrixArray metric, double absolute_uncertainty) {
            if (matrix.shape(0) != metric.shape(0) || matrix.shape(1) != metric.shape(1)) {
                throw std::runtime_error(
                    "_generalized_hermitian_min_eigenvalue_lanczos: shape mismatch"
                );
            }
            const auto size = static_cast<size_t>(matrix.shape(0));
            const auto matrix_values = copy_square_matrix(matrix);
            const auto metric_values = copy_square_matrix(metric);
            return simplex_certificate::detail::generalized_hermitian_min_eigenvalue_lanczos(
                std::span<const Complex>(matrix_values.data(), matrix_values.size()),
                std::span<const Complex>(metric_values.data(), metric_values.size()),
                size,
                absolute_uncertainty
            );
        },
        "matrix"_a,
        "metric"_a,
        "absolute_uncertainty"_a
    );
    m.def(
        "_root_mesh_certificate_gap_bound",
        &root_mesh_certificate_gap_bound,
        "model"_a,
        "mu"_a,
        "margin"_a = 0.0,
        "tol"_a = 1e-14,
        "gap_bound_precision"_a = 1e-12
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
