#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include "lineartetrahedron/runtime.h"

#include <cstddef>
#include <cstdint>
#include <new>

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
            &TightBindingModel::derivative_spectral_norm,
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
}

}  // namespace lineartetrahedron
