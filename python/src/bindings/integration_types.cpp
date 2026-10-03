#include "arrays.h"
#include "bindings.h"

#include <fermisimplex/integration.h>
#include <fermisimplex/occupation.h>

namespace fermisimplex::bindings {

void bind_integration_types(nb::module_ &module) {
    nb::class_<OccupationEnclosure>(module, "OccupationEnclosure")
        .def_ro("charge_lower", &OccupationEnclosure::charge_lower)
        .def_ro("charge_upper", &OccupationEnclosure::charge_upper)
        .def_ro("density_cut_error", &OccupationEnclosure::density_cut_error)
        .def_ro("interpolation_error", &OccupationEnclosure::interpolation_error)
        .def_ro("model_error", &OccupationEnclosure::model_error)
        .def_ro("safe_gap", &OccupationEnclosure::safe_gap)
        .def_ro("active_dimension", &OccupationEnclosure::active_dimension)
        .def_ro("occupation_lower", &OccupationEnclosure::occupation_lower)
        .def_ro("occupation_upper", &OccupationEnclosure::occupation_upper)
        .def_ro("remainder_is_sampled", &OccupationEnclosure::remainder_is_sampled)
        .def_prop_ro("fixed_occupation", &OccupationEnclosure::fixed_occupation);

    nb::class_<IntegrationStats>(module, "IntegrationStats")
        .def_ro("evaluations", &IntegrationStats::evaluations)
        .def_ro("simplex_visits", &IntegrationStats::simplex_visits)
        .def_ro("refinements", &IntegrationStats::refinements)
        .def_ro("cached_vertices", &IntegrationStats::cached_vertices)
        .def_ro("active_simplices", &IntegrationStats::active_simplices)
        .def_ro("active_vertices", &IntegrationStats::active_vertices)
        .def_ro("p_refinements", &IntegrationStats::p_refinements)
        .def_ro("cubature_evaluations", &IntegrationStats::cubature_evaluations)
        .def_ro("max_degree", &IntegrationStats::max_degree)
        .def_ro("target_reached", &IntegrationStats::target_reached);

    nb::class_<ChargeErrorStats>(
        module,
        "ChargeErrorStats",
        "Work and reduction diagnostics for the sampled occupation enclosure."
    )
        .def_ro("root_simplices", &ChargeErrorStats::root_simplices)
        .def_ro(
            "hamiltonian_evaluations",
            &ChargeErrorStats::hamiltonian_evaluations
        )
        .def_ro("reduced_eigensystems", &ChargeErrorStats::reduced_eigensystems)
        .def_ro("norm_eigensystems", &ChargeErrorStats::norm_eigensystems)
        .def_ro("schur_evaluations", &ChargeErrorStats::schur_evaluations)
        .def_ro("schur_reductions", &ChargeErrorStats::schur_reductions)
        .def_ro("certificate_builds", &ChargeErrorStats::certificate_builds)
        .def_ro("certificate_reuses", &ChargeErrorStats::certificate_reuses)
        .def_ro("micro_simplices", &ChargeErrorStats::micro_simplices)
        .def_ro("terminal_simplices", &ChargeErrorStats::terminal_simplices)
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
        .def_ro("density_cut_error", &ChargeResult::density_cut_error)
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

    nb::class_<CurrentMeshChargeResult>(module, "CurrentMeshChargeResult")
        .def_ro("value", &CurrentMeshChargeResult::value)
        .def_ro("dcharge_dmu", &CurrentMeshChargeResult::dcharge_dmu);

    nb::class_<DensityComponentsResult>(module, "DensityComponentsResult")
        .def_prop_ro(
            "values",
            [](const DensityComponentsResult &result) {
                return make_array(
                    std::vector<std::complex<double>>(result.values),
                    {result.values.size()}
                );
            },
            nb::rv_policy::move
        )
        .def_ro("stopping_error", &DensityComponentsResult::stopping_error)
        .def_prop_ro(
            "stats",
            [](const DensityComponentsResult &result) { return result.stats; }
        );

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
