#include "arrays.h"
#include "bindings.h"

#include <fermisimplex/integration.h>

namespace fermisimplex::bindings {

void bind_integration_types(nb::module_ &module) {
    nb::class_<IntegrationStats>(module, "IntegrationStats")
        .def_ro("evaluations", &IntegrationStats::evaluations)
        .def_ro("simplex_visits", &IntegrationStats::simplex_visits)
        .def_ro("refinements", &IntegrationStats::refinements)
        .def_ro("cached_vertices", &IntegrationStats::cached_vertices)
        .def_ro("active_simplices", &IntegrationStats::active_simplices)
        .def_ro("active_vertices", &IntegrationStats::active_vertices)
        .def_ro("target_reached", &IntegrationStats::target_reached);

    nb::class_<ChargeErrorStats>(
        module,
        "ChargeErrorStats",
        "Work, reduction, and fallback diagnostics for charge-error sampling."
    )
        .def_ro("root_simplices", &ChargeErrorStats::root_simplices)
        .def_ro(
            "hamiltonian_evaluations",
            &ChargeErrorStats::hamiltonian_evaluations
        )
        .def_ro("full_eigensystems", &ChargeErrorStats::full_eigensystems)
        .def_ro("reduced_eigensystems", &ChargeErrorStats::reduced_eigensystems)
        .def_ro("norm_eigensystems", &ChargeErrorStats::norm_eigensystems)
        .def_ro("safe_block_solves", &ChargeErrorStats::safe_block_solves)
        .def_ro("schur_reductions", &ChargeErrorStats::schur_reductions)
        .def_ro("micro_simplices", &ChargeErrorStats::micro_simplices)
        .def_ro("terminal_simplices", &ChargeErrorStats::terminal_simplices)
        .def_ro(
            "conservative_fallbacks",
            &ChargeErrorStats::conservative_fallbacks,
            "Number of sampled occupation-range fallbacks."
        )
        .def_ro(
            "singular_schur_failures",
            &ChargeErrorStats::singular_schur_failures
        )
        .def_ro(
            "initial_active_dimension_sum",
            &ChargeErrorStats::initial_active_dimension_sum
        )
        .def_ro(
            "terminal_active_dimension_sum",
            &ChargeErrorStats::terminal_active_dimension_sum
        )
        .def_ro(
            "minimum_active_dimension",
            &ChargeErrorStats::minimum_active_dimension,
            "Smallest reduced dimension, or zero if no reduction succeeded."
        );

    nb::class_<ChargeResult>(module, "ChargeResult")
        .def_ro("value", &ChargeResult::value)
        .def_ro("stopping_error", &ChargeResult::stopping_error)
        .def_ro("dcharge_dmu", &ChargeResult::dcharge_dmu)
        .def_ro(
            "visible_gapless_simplices",
            &ChargeResult::visible_gapless_simplices
        )
        .def_ro("inconclusive_simplices", &ChargeResult::inconclusive_simplices)
        .def_prop_ro(
            "error_stats",
            [](const ChargeResult &result) { return result.error_stats; },
            "Work and fallback diagnostics for the sampled error estimator."
        )
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
