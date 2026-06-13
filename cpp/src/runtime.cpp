#include "lineartetrahedron/runtime.h"

#include "lineartetrahedron/charge_integrand.h"
#include "lineartetrahedron/density_integrand.h"

#include <adaptivesimplex/adaptive/adaptive_loop.h>
#include <adaptivesimplex/adaptive/evaluation.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace lineartetrahedron {
namespace adaptive = adaptivesimplex::adaptive;

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

ChargeIntegrateResult IntegrationRuntime::integrate_charge(
    double mu,
    const adaptive::Options &options
) {
    auto integrand = ChargeIntegrand(state_, mu);
    const auto raw = adaptive::run(state_.geometry(), integrand, options);
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

ChargeIntegrateResult IntegrationRuntime::evaluate_charge(
    double mu,
    const adaptive::Options &options
) {
    auto integrand = ChargeIntegrand(state_, mu);
    const auto preview_depth = std::max<std::uint32_t>(options.preview_depth, 1);
    const auto active = state_.geometry().simplices().active_simplices();
    const auto active_simplices = std::vector<adaptivesimplex::core::SimplexId>(
        active.begin(),
        active.end()
    );

    ChargeValue value;
    ChargeValue correction;
    std::int64_t work = 0;
    for (const auto &estimate :
         adaptive::estimate_simplices(
             state_.geometry(),
             integrand,
             active_simplices,
             preview_depth,
             work
         )) {
        value += estimate.value;
        correction += estimate.correction;
    }

    return ChargeIntegrateResult{
        .charge = value.charge,
        .charge_error = std::abs(correction.charge),
        .dcharge_dmu = value.derivative,
        .work = work,
        .refinements = 0,
        .n_active_simplices = n_active_simplices(),
        .n_active_vertices = n_active_vertices(),
        .converged = std::abs(correction.charge) <= options.target_error,
    };
}

DensityIntegrateResult IntegrationRuntime::integrate_density(
    double mu,
    const adaptive::Options &options
) {
    auto integrand = DensityIntegrand(state_, mu);
    auto raw = integrand.estimate_density(options);

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
