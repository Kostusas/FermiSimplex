#include "arrays.h"
#include "bindings.h"

#include "integration/runtime.h"

#include <adaptivesimplex/adaptive/types.h>

#include <nanobind/stl/shared_ptr.h>

#include <complex>
#include <cstdint>
#include <new>
#include <vector>

namespace lineartetrahedron::bindings {

namespace adaptive = adaptivesimplex::adaptive;

void bind_integration(nb::module_ &m) {
    using namespace nb::literals;

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
            "__init__",
            [](
                IntegrationRuntime *self,
                std::shared_ptr<TightBindingModel> model,
                KeyArray keys,
                ComponentIndexArray component_rows,
                ComponentIndexArray component_cols,
                ComponentIndexArray component_key_indices,
                double tol
            ) {
                new (self) IntegrationRuntime(
                    std::move(model),
                    copy_keys(keys),
                    copy_1d(component_rows),
                    copy_1d(component_cols),
                    copy_1d(component_key_indices),
                    tol
                );
            },
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
            "options"_a,
            "certify"_a = true
        )
        .def(
            "integrate_density",
            &IntegrationRuntime::integrate_density,
            "mu"_a,
            "options"_a
        );
}

}  // namespace lineartetrahedron::bindings
