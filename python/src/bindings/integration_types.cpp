#include "arrays.h"
#include "bindings.h"

#include <fermisimplex/integration.h>

namespace fermisimplex::bindings {

void bind_integration_types(nb::module_ &module) {
    nb::class_<IntegrationStats>(module, "IntegrationStats")
        .def_ro("evaluations", &IntegrationStats::evaluations)
        .def_ro("refinements", &IntegrationStats::refinements)
        .def_ro("cached_vertices", &IntegrationStats::cached_vertices)
        .def_ro("active_simplices", &IntegrationStats::active_simplices)
        .def_ro("active_vertices", &IntegrationStats::active_vertices)
        .def_ro("target_reached", &IntegrationStats::target_reached);

    nb::class_<ChargeResult>(module, "ChargeResult")
        .def_ro("value", &ChargeResult::value)
        .def_ro("stopping_error", &ChargeResult::stopping_error)
        .def_ro("certified_error_bound", &ChargeResult::certified_error_bound)
        .def_ro("dcharge_dmu", &ChargeResult::dcharge_dmu)
        .def_ro(
            "visible_gapless_simplices",
            &ChargeResult::visible_gapless_simplices
        )
        .def_ro("inconclusive_simplices", &ChargeResult::inconclusive_simplices)
        .def_prop_ro("stats", [](const ChargeResult &result) { return result.stats; });

    nb::class_<DensityMatrixResult>(module, "DensityMatrixResult")
        .def_prop_ro(
            "matrices",
            [](const DensityMatrixResult &result) {
                return make_array(
                    std::vector<std::complex<double>>(result.matrices),
                    {result.lattice_vector_count, result.ndof, result.ndof}
                );
            },
            nb::rv_policy::move
        )
        .def_ro("stopping_error", &DensityMatrixResult::stopping_error)
        .def_prop_ro(
            "stats",
            [](const DensityMatrixResult &result) { return result.stats; }
        );

}

}  // namespace fermisimplex::bindings
