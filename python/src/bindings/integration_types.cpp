#include "arrays.h"
#include "bindings.h"

#include <fermisimplex/integration.h>

#include <adaptivesimplex/adaptive/types.h>

#include <cmath>
#include <cstdint>
#include <new>

namespace fermisimplex::bindings {
namespace adaptive = adaptivesimplex::adaptive;

namespace {

adaptive::Options options(
    double target_error,
    std::int64_t max_refinements,
    std::uint32_t preview_depth,
    std::size_t min_refinement_batch_size,
    std::size_t max_refinement_batch_size
) {
    if (!std::isfinite(target_error) || target_error < 0.0) {
        throw nb::value_error("target_error must be finite and non-negative");
    }
    if (max_refinements < -1) {
        throw nb::value_error("max_refinements must be -1 or non-negative");
    }
    if (preview_depth == 0) {
        throw nb::value_error("preview_depth must be positive");
    }
    if (min_refinement_batch_size == 0) {
        throw nb::value_error("min_refinement_batch_size must be positive");
    }
    if (max_refinement_batch_size < min_refinement_batch_size) {
        throw nb::value_error(
            "max_refinement_batch_size must be at least min_refinement_batch_size"
        );
    }
    return adaptive::Options{
        .target_error = target_error,
        .max_refinements = max_refinements,
        .preview_depth = preview_depth,
        .min_refinement_batch_size = min_refinement_batch_size,
        .max_refinement_batch_size = max_refinement_batch_size,
    };
}

}  // namespace

void bind_integration_types(nb::module_ &module) {
    using namespace nb::literals;

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

    nb::class_<adaptive::Options>(module, "AdaptiveOptions")
        .def(
            "__init__",
            [](adaptive::Options *self,
               double target_error,
               std::int64_t max_refinements,
               std::uint32_t preview_depth,
               std::size_t min_refinement_batch_size,
               std::size_t max_refinement_batch_size) {
                new (self) adaptive::Options(options(
                    target_error,
                    max_refinements,
                    preview_depth,
                    min_refinement_batch_size,
                    max_refinement_batch_size
                ));
            },
            "target_error"_a,
            "max_refinements"_a = -1,
            "preview_depth"_a = 1,
            "min_refinement_batch_size"_a = 1,
            "max_refinement_batch_size"_a = 100
        )
        .def_ro("target_error", &adaptive::Options::target_error)
        .def_ro("max_refinements", &adaptive::Options::max_refinements)
        .def_ro("preview_depth", &adaptive::Options::preview_depth)
        .def_ro(
            "min_refinement_batch_size",
            &adaptive::Options::min_refinement_batch_size
        )
        .def_ro(
            "max_refinement_batch_size",
            &adaptive::Options::max_refinement_batch_size
        );
}

}  // namespace fermisimplex::bindings
