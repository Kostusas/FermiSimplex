#include "lineartetrahedron/runtime.h"

#include "lineartetrahedron/charge_integrand.h"
#include "lineartetrahedron/density_integrand.h"

#include <adaptivesimplex/adaptive/adaptive_loop.h>

#include <cmath>
#include <utility>

namespace lineartetrahedron {
namespace adaptive = adaptivesimplex::adaptive;

namespace {

constexpr std::uint32_t kPreviewDepth = 1;

}  // namespace

IntegrationRuntime::IntegrationRuntime(
    std::shared_ptr<TightBindingModel> model,
    KeyArray keys,
    ComponentIndexArray component_rows,
    ComponentIndexArray component_cols,
    ComponentIndexArray component_key_indices,
    double tol
) : state_(
        std::move(model),
        keys,
        component_rows,
        component_cols,
        component_key_indices,
        tol
    ) {
}

adaptive::Options IntegrationRuntime::options(
    double target,
    std::int64_t max_refinements
) const {
    return adaptive::Options{
        .target_error = target,
        .max_refinements = max_refinements,
        .preview_depth = kPreviewDepth,
        .min_refinement_batch_size = 1,
        .max_refinement_batch_size = 100,
    };
}

ChargeIntegrateResult IntegrationRuntime::integrate_charge(
    double mu,
    double charge_atol,
    std::int64_t max_refinements
) {
    auto integrand = ChargeIntegrand(state_, mu);
    const auto raw = adaptive::run(state_.geometry(), integrand, options(charge_atol, max_refinements));
    return ChargeIntegrateResult{
        .charge = raw.value.charge,
        .charge_error = std::abs(raw.correction.charge),
        .dcharge_dmu = raw.value.derivative,
        .work = raw.work,
        .refinements = raw.refinements,
        .n_active_simplices = n_active_simplices(),
        .n_active_vertices = n_active_vertices(),
        .converged = raw.converged,
    };
}

DensityIntegrateResult IntegrationRuntime::integrate_density(
    double mu,
    double density_atol,
    std::int64_t max_refinements
) {
    auto integrand = DensityIntegrand(state_, mu);
    const auto raw = adaptive::run(state_.geometry(), integrand, options(density_atol, max_refinements));

    DensityIntegrateResult result;
    result.estimate = std::vector<std::complex<double>>(raw.value.values());
    result.error_vector.resize(raw.correction.size());
    for (size_t index = 0; index < raw.correction.size(); ++index) {
        result.error_vector[index] = std::abs(raw.correction[index]);
    }
    result.error_scalar = raw.error_scalar;
    result.work = raw.work;
    result.refinements = raw.refinements;
    result.n_active_simplices = n_active_simplices();
    result.n_active_vertices = n_active_vertices();
    result.converged = raw.converged;
    return result;
}

}  // namespace lineartetrahedron
