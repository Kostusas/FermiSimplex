#include "integration/runtime.h"

#include <adaptivesimplex/adaptive/adaptive_loop.h>
#include <adaptivesimplex/adaptive/simplex_integrand.h>

#include <cmath>
#include <utility>
#include <vector>

namespace lineartetrahedron {
namespace adaptive = adaptivesimplex::adaptive;
namespace core = adaptivesimplex::core;

namespace {

struct DensityStoppingError {
    template <class IntegralValue> using state_type = IntegralValue;

    template <class IntegralValue> state_type<IntegralValue> zero() const {
        return {};
    }
    template <class IntegralValue>
    state_type<IntegralValue> contribution(
        const adaptive::SimplexEstimate<IntegralValue> &estimate
    ) const {
        return estimate.correction;
    }
    template <class IntegralValue>
    void add(state_type<IntegralValue> &state, const state_type<IntegralValue> &value) const {
        state += value;
    }
    template <class IntegralValue>
    void remove(state_type<IntegralValue> &state, const state_type<IntegralValue> &value) const {
        state -= value;
    }
    template <class IntegralValue> double error(const state_type<IntegralValue> &state) const {
        return state.max_abs();
    }
};

struct DensityRefinementScore {
    template <class Context> double operator()(const Context &context) const {
        return context.correction.max_abs();
    }
};

}  // namespace

IntegrationRuntime::IntegrationRuntime(
    std::shared_ptr<TightBindingModel> model,
    std::vector<std::int64_t> keys,
    std::vector<std::int64_t> component_rows,
    std::vector<std::int64_t> component_cols,
    std::vector<std::int64_t> component_key_indices,
    double tol
) : workspace_(
        std::move(model),
        tol
    ),
    density_(
        workspace_.ndim(),
        workspace_.ndof(),
        std::move(keys),
        std::move(component_rows),
        std::move(component_cols),
        std::move(component_key_indices)
    ) {
}

DensityIntegrateResult IntegrationRuntime::integrate_density(
    double mu,
    const adaptive::Options &options
) {
    auto integrand = adaptive::simplex_integrand(
        workspace_.cache(),
        [this](std::span<const double> point) {
            return workspace_.evaluate_vertex(point);
        },
        [this, mu](
            const core::Geometry &geometry,
            core::SimplexId simplex_id,
            const core::VertexCache<VertexSpectra> &cache
        ) {
            return density_.on_simplex(mu, workspace_, geometry, simplex_id, cache);
        },
        adaptive::estimation_policies<DensityStoppingError, DensityRefinementScore>{}
    );
    const auto raw = adaptive::run(workspace_.geometry(), integrand, options);

    DensityIntegrateResult result;
    result.estimate = std::vector<std::complex<double>>(raw.integral.values());
    result.error_vector.resize(raw.correction.size());
    for (size_t index = 0; index < raw.correction.size(); ++index) {
        result.error_vector[index] = std::abs(raw.correction[index]);
    }
    result.error_scalar = raw.stopping_error;
    result.work = raw.evaluations;
    result.refinements = raw.refinements;
    result.n_active_simplices = n_active_simplices();
    result.n_active_vertices = n_active_vertices();
    result.converged = raw.converged;
    return result;
}

}  // namespace lineartetrahedron
