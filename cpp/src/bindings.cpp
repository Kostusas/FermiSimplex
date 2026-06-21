#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include "lineartetrahedron/fermi_surface.h"
#include "lineartetrahedron/linalg.h"
#include "lineartetrahedron/runtime.h"

#include <cstddef>
#include <cstdint>
#include <new>
#include <stdexcept>

namespace nb = nanobind;
using namespace nb::literals;

namespace lineartetrahedron {
namespace adaptive = adaptivesimplex::adaptive;

NB_MODULE(_native, m) {
    m.doc() = "Native runtime for lineartetrahedron";

    nb::class_<TightBindingModel>(m, "TightBindingModel")
        .def(nb::init<KeyArray, HoppingMatrixArray>(), "keys"_a, "matrices"_a)
        .def_prop_ro("ndim", &TightBindingModel::ndim)
        .def_prop_ro("ndof", &TightBindingModel::ndof)
        .def_prop_ro("nterms", &TightBindingModel::nterms)
        .def_prop_ro("reduced_lipschitz_bound", &TightBindingModel::reduced_lipschitz_bound)
        .def_prop_ro(
            "hopping_spectral_norms",
            [](const TightBindingModel &self) {
                const auto norms = self.hopping_spectral_norms();
                return std::vector<double>(norms.begin(), norms.end());
            }
        )
        .def_prop_ro(
            "global_derivative_bounds",
            [](const TightBindingModel &self) {
                const auto bounds = self.global_derivative_bounds();
                return std::vector<double>(bounds.begin(), bounds.end());
            }
        )
        .def_prop_ro(
            "hessian_bounds",
            [](const TightBindingModel &self) {
                const auto bounds = self.hessian_bounds();
                return std::vector<double>(bounds.begin(), bounds.end());
            }
        )
        .def("evaluate_point", &TightBindingModel::evaluate_point, "point"_a)
        .def(
            "derivative_spectral_norm",
            static_cast<double (TightBindingModel::*)(PointArray, size_t) const>(
                &TightBindingModel::derivative_spectral_norm
            ),
            "point"_a,
            "axis"_a
        );

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
            "options"_a,
            "use_weyl_bounds"_a = false
        )
        .def(
            "evaluate_charge",
            &IntegrationRuntime::evaluate_charge,
            "mu"_a,
            "options"_a,
            "use_weyl_bounds"_a = false
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
        "max_refinements"_a = -1,
        "use_weyl_bounds"_a = true,
        "tol"_a = 1e-14
    );
    m.def(
        "_product_simplex_triangulation_cells",
        &product_simplex_triangulation_cells,
        "negative_count"_a,
        "positive_count"_a
    );
    m.def(
        "_hermitian_min_eigenvalue_lanczos",
        [](
            nb::ndarray<nb::numpy, const std::complex<double>, nb::ndim<2>, nb::c_contig> matrix,
            double absolute_uncertainty
        ) {
            if (matrix.shape(0) != matrix.shape(1)) {
                throw std::runtime_error("_hermitian_min_eigenvalue_lanczos: matrix must be square");
            }
            const auto size = static_cast<size_t>(matrix.shape(0));
            return hermitian_min_eigenvalue_lanczos(
                std::span<const std::complex<double>>(matrix.data(), size * size),
                size,
                absolute_uncertainty
            );
        },
        "matrix"_a,
        "absolute_uncertainty"_a = 5e-5
    );
    m.def("_reset_lanczos_stats", &reset_lanczos_stats);
    m.def(
        "_lanczos_stats",
        []() {
            const auto stats = lanczos_stats();
            nb::dict result;
            result["calls"] = stats.calls;
            result["iterations"] = stats.iterations;
            result["ritz_checks"] = stats.ritz_checks;
            result["converged_by_uncertainty"] = stats.converged_by_uncertainty;
            result["converged_at_full_dimension"] = stats.converged_at_full_dimension;
            result["converged_by_zero_residual"] = stats.converged_by_zero_residual;
            result["lanczos_nanoseconds"] = stats.lanczos_nanoseconds;
            result["derivative_calls"] = stats.derivative_calls;
            result["derivative_assembly_nanoseconds"] = stats.derivative_assembly_nanoseconds;
            result["derivative_total_nanoseconds"] = stats.derivative_total_nanoseconds;
            return result;
        }
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
            result["cached_decisions"] = stats.cached_decisions;
            result["classified_simplices"] = stats.classified_simplices;
            result["marked_simplices"] = stats.marked_simplices;
            result["terminal_cached_simplices"] = stats.terminal_cached_simplices;
            result["refinement_calls"] = stats.refinement_calls;
            result["first_safe_marking_pass"] = stats.first_safe_marking_pass;
            result["first_safe_total_nanoseconds"] = stats.first_safe_total_nanoseconds;
            result["first_safe_refinements"] = stats.first_safe_refinements;
            result["first_safe_active_simplices"] = stats.first_safe_active_simplices;
            result["first_safe_new_simplices"] = stats.first_safe_new_simplices;
            return result;
        }
    );
}

}  // namespace lineartetrahedron
