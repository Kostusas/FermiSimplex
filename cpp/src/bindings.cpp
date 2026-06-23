#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include "lineartetrahedron/fermi_surface.h"
#include "lineartetrahedron/runtime.h"

#include <cstddef>
#include <cstdint>
#include <complex>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace lineartetrahedron {
namespace adaptive = adaptivesimplex::adaptive;

namespace {

using CallbackMatrixArray =
    nb::ndarray<nb::numpy, const std::complex<double>, nb::ndim<2>, nb::c_contig>;

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
    double tol
) {
    return fermi_surface_from_model(
        std::make_shared<PythonHamiltonianModel>(std::move(callable), ndim, ndof),
        mu,
        min_feature_size,
        max_diagonalizations,
        tol
    );
}

}  // namespace

NB_MODULE(_native, m) {
    m.doc() = "Native runtime for lineartetrahedron";

    nb::class_<TightBindingModel>(m, "TightBindingModel")
        .def(nb::init<KeyArray, HoppingMatrixArray>(), "keys"_a, "matrices"_a)
        .def_prop_ro("ndim", &TightBindingModel::ndim)
        .def_prop_ro("ndof", &TightBindingModel::ndof)
        .def_prop_ro("nterms", &TightBindingModel::nterms)
        .def("evaluate_point", &TightBindingModel::evaluate_point, "point"_a);

    nb::class_<ChargeIntegrateResult>(m, "ChargeIntegrateResult")
        .def_prop_ro("charge", [](const ChargeIntegrateResult &self) { return self.charge; })
        .def_prop_ro("charge_error", [](const ChargeIntegrateResult &self) { return self.charge_error; })
        .def_prop_ro("dcharge_dmu", [](const ChargeIntegrateResult &self) { return self.dcharge_dmu; })
        .def_prop_ro("work", [](const ChargeIntegrateResult &self) { return self.work; })
        .def_prop_ro("refinements", [](const ChargeIntegrateResult &self) { return self.refinements; })
        .def_prop_ro("n_active_simplices", [](const ChargeIntegrateResult &self) { return self.n_active_simplices; })
        .def_prop_ro("n_active_vertices", [](const ChargeIntegrateResult &self) { return self.n_active_vertices; })
        .def_prop_ro("converged", [](const ChargeIntegrateResult &self) { return self.converged; });

    nb::class_<DensityIntegrateResult>(m, "DensityIntegrateResult")
        .def(
            "estimate_array",
            [](const DensityIntegrateResult &self) {
                return make_array(
                    std::vector<std::complex<double>>(self.estimate),
                    {self.estimate.size()}
                );
            }
        )
        .def(
            "error_vector_array",
            [](const DensityIntegrateResult &self) {
                return make_array(std::vector<double>(self.error_vector), {self.error_vector.size()});
            }
        )
        .def_prop_ro("error_scalar", [](const DensityIntegrateResult &self) { return self.error_scalar; })
        .def_prop_ro("work", [](const DensityIntegrateResult &self) { return self.work; })
        .def_prop_ro("refinements", [](const DensityIntegrateResult &self) { return self.refinements; })
        .def_prop_ro("n_active_simplices", [](const DensityIntegrateResult &self) { return self.n_active_simplices; })
        .def_prop_ro("n_active_vertices", [](const DensityIntegrateResult &self) { return self.n_active_vertices; })
        .def_prop_ro("converged", [](const DensityIntegrateResult &self) { return self.converged; });

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
        .def_prop_ro("ndim", [](const FermiSurfaceResult &self) { return self.ndim; })
        .def_prop_ro("converged", [](const FermiSurfaceResult &self) { return self.converged; })
        .def_prop_ro("refinements", [](const FermiSurfaceResult &self) { return self.refinements; })
        .def_prop_ro("n_active_simplices", [](const FermiSurfaceResult &self) { return self.n_active_simplices; })
        .def_prop_ro("n_active_vertices", [](const FermiSurfaceResult &self) { return self.n_active_vertices; })
        .def_prop_ro("n_safe_simplices", [](const FermiSurfaceResult &self) { return self.n_safe_simplices; })
        .def_prop_ro("n_cut_simplices", [](const FermiSurfaceResult &self) { return self.n_cut_simplices; })
        .def_prop_ro("n_feature_size_simplices", [](const FermiSurfaceResult &self) { return self.n_feature_size_simplices; })
        .def_prop_ro("n_unresolved_simplices", [](const FermiSurfaceResult &self) { return self.n_unresolved_simplices; })
        .def_prop_ro("min_feature_size", [](const FermiSurfaceResult &self) { return self.min_feature_size; });

    nb::class_<adaptive::Options>(m, "AdaptiveOptions")
        .def(
            "__init__",
            [](
                adaptive::Options *self,
                double target_error,
                std::int64_t max_refinements,
                std::uint32_t preview_depth,
                std::int64_t min_refinement_batch_size,
                std::int64_t max_refinement_batch_size
            ) {
                new (self) adaptive::Options{
                    .target_error = target_error,
                    .max_refinements = static_cast<std::int64_t>(max_refinements),
                    .preview_depth = static_cast<std::uint32_t>(preview_depth),
                    .min_refinement_batch_size =
                        static_cast<std::size_t>(min_refinement_batch_size),
                    .max_refinement_batch_size =
                        static_cast<std::size_t>(max_refinement_batch_size),
                };
            },
            "target_error"_a,
            "max_refinements"_a = -1,
            "preview_depth"_a = 1,
            "min_refinement_batch_size"_a = 1,
            "max_refinement_batch_size"_a = 100
        )
        .def_rw("target_error", &adaptive::Options::target_error)
        .def_rw("max_refinements", &adaptive::Options::max_refinements)
        .def_rw("preview_depth", &adaptive::Options::preview_depth)
        .def_rw("min_refinement_batch_size", &adaptive::Options::min_refinement_batch_size)
        .def_rw("max_refinement_batch_size", &adaptive::Options::max_refinement_batch_size);

    nb::class_<IntegrationRuntime>(m, "IntegrationRuntime")
        .def(
            nb::init<std::shared_ptr<TightBindingModel>, KeyArray, ComponentIndexArray, ComponentIndexArray, ComponentIndexArray, double>(),
            "model"_a,
            "keys"_a,
            "component_rows"_a,
            "component_cols"_a,
            "component_key_indices"_a,
            "tol"_a = 1e-14
        )
        .def_prop_ro("ndim", &IntegrationRuntime::ndim)
        .def_prop_ro("ndof", &IntegrationRuntime::ndof)
        .def_prop_ro("density_component_count", &IntegrationRuntime::density_component_count)
        .def_prop_ro("n_cached_nodes", &IntegrationRuntime::n_cached_nodes)
        .def_prop_ro("n_active_simplices", &IntegrationRuntime::n_active_simplices)
        .def_prop_ro("n_active_vertices", &IntegrationRuntime::n_active_vertices)
        .def(
            "integrate_charge",
            &IntegrationRuntime::integrate_charge,
            "mu"_a,
            "options"_a
        )
        .def(
            "evaluate_charge",
            &IntegrationRuntime::evaluate_charge,
            "mu"_a,
            "options"_a
        )
        .def(
            "integrate_density",
            &IntegrationRuntime::integrate_density,
            "mu"_a,
            "options"_a
        );

    m.def(
        "fermi_surface",
        &fermi_surface,
        "model"_a,
        "mu"_a,
        "min_feature_size"_a,
        "max_diagonalizations"_a = -1,
        "tol"_a = 1e-14
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
        "tol"_a = 1e-14
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
            result["vertex_evaluation_nanoseconds"] = stats.vertex_evaluation_nanoseconds;
            result["marking_nanoseconds"] = stats.marking_nanoseconds;
            result["refinement_nanoseconds"] = stats.refinement_nanoseconds;
            result["extraction_nanoseconds"] = stats.extraction_nanoseconds;
            result["total_nanoseconds"] = stats.total_nanoseconds;
            result["vertex_evaluation_calls"] = stats.vertex_evaluation_calls;
            result["evaluated_vertices"] = stats.evaluated_vertices;
            result["marking_passes"] = stats.marking_passes;
            result["active_simplex_visits"] = stats.active_simplex_visits;
            result["classified_simplices"] = stats.classified_simplices;
            result["marked_simplices"] = stats.marked_simplices;
            result["refinement_calls"] = stats.refinement_calls;
            return result;
        }
    );
}

}  // namespace lineartetrahedron
