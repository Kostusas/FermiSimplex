#include "lineartetrahedron/runtime.h"

#include "lineartetrahedron/charge.h"

#include <adaptivesimplex/adaptive/adaptive_loop.h>
#include <adaptivesimplex/adaptive/evaluation.h>
#include <adaptivesimplex/adaptive/simplex_integrand.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace lineartetrahedron {
namespace adaptive = adaptivesimplex::adaptive;
namespace core = adaptivesimplex::core;

namespace {

struct ChargeStoppingError {
    template <class IntegralValue> using state_type = IntegralValue;

    template <class IntegralValue> state_type<IntegralValue> zero() const {
        return {};
    }
    template <class IntegralValue>
    state_type<IntegralValue> contribution(
        const adaptive::SimplexEstimate<IntegralValue> &estimate
    ) const {
        auto contribution = estimate.correction;
        contribution.weyl_indicator_error =
            std::max(estimate.coarse.weyl_indicator_error, estimate.preview.weyl_indicator_error);
        return contribution;
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
        return std::abs(state.charge) + std::max(0.0, state.weyl_indicator_error);
    }
};

struct ChargeRefinementScore {
    template <class Context> double operator()(const Context &context) const {
        return std::max(
            std::abs(context.correction.charge),
            std::max(
                context.coarse.weyl_indicator_error,
                context.preview.weyl_indicator_error
            )
        );
    }
};

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
    KeyArray keys,
    ComponentIndexArray component_rows,
    ComponentIndexArray component_cols,
    ComponentIndexArray component_key_indices,
    double tol
) : workspace_(
        std::move(model),
        tol
    ),
    density_(
        workspace_.ndim(),
        workspace_.ndof(),
        keys,
        component_rows,
        component_cols,
        component_key_indices
    ) {
}

ChargeIntegrateResult IntegrationRuntime::integrate_charge(
    double mu,
    const adaptive::Options &options,
    bool use_weyl_bounds
) {
    const auto weyl_indicator_error =
        use_weyl_bounds
            ? std::nextafter(options.target_error, std::numeric_limits<double>::infinity())
            : 0.0;
    auto weyl_marker_cache = ChargeWeylMarkerCache{};
    auto integrand = adaptive::simplex_integrand(
        workspace_.cache(),
        [this](std::span<const double> point) {
            return workspace_.evaluate_vertex(point);
        },
        [this, mu, weyl_indicator_error, &weyl_marker_cache](
            const core::Geometry &geometry,
            core::SimplexId simplex_id,
            const core::VertexCache<VertexSpectra> &cache
        ) {
            return charge_on_simplex(
                mu,
                workspace_,
                geometry,
                simplex_id,
                cache,
                weyl_indicator_error,
                &weyl_marker_cache
            );
        },
        adaptive::estimation_policies<ChargeStoppingError, ChargeRefinementScore>{}
    );
    const auto raw = adaptive::run(workspace_.geometry(), integrand, options);
    return ChargeIntegrateResult{
        .charge = raw.integral.charge,
        .charge_error = raw.stopping_error,
        .dcharge_dmu = raw.integral.derivative,
        .work = raw.evaluations,
        .refinements = raw.refinements,
        .n_active_simplices = n_active_simplices(),
        .n_active_vertices = n_active_vertices(),
        .converged = raw.converged,
    };
}

ChargeIntegrateResult IntegrationRuntime::evaluate_charge(
    double mu,
    const adaptive::Options &options,
    bool use_weyl_bounds
) {
    const auto weyl_indicator_error =
        use_weyl_bounds
            ? std::nextafter(options.target_error, std::numeric_limits<double>::infinity())
            : 0.0;
    auto weyl_marker_cache = ChargeWeylMarkerCache{};
    auto integrand = adaptive::simplex_integrand(
        workspace_.cache(),
        [this](std::span<const double> point) {
            return workspace_.evaluate_vertex(point);
        },
        [this, mu, weyl_indicator_error, &weyl_marker_cache](
            const core::Geometry &geometry,
            core::SimplexId simplex_id,
            const core::VertexCache<VertexSpectra> &cache
        ) {
            return charge_on_simplex(
                mu,
                workspace_,
                geometry,
                simplex_id,
                cache,
                weyl_indicator_error,
                &weyl_marker_cache
            );
        },
        adaptive::estimation_policies<ChargeStoppingError, ChargeRefinementScore>{}
    );

    const auto preview_depth = std::max<std::uint32_t>(options.preview_depth, 1);
    const auto active = workspace_.geometry().simplices().active_simplices();
    const auto active_simplices = std::vector<core::SimplexId>(active.begin(), active.end());

    ChargeValue value;
    auto stopping_global_error = ChargeStoppingError{};
    auto stopping_state = stopping_global_error.template zero<ChargeValue>();
    std::int64_t evaluations = 0;
    for (const auto &estimate :
         adaptive::estimate_simplices(
             workspace_.geometry(),
             integrand,
             active_simplices,
             preview_depth,
             evaluations
         )) {
        value += estimate.preview;
        auto stopping_contribution =
            stopping_global_error.template contribution<ChargeValue>(estimate);
        stopping_global_error.template add<ChargeValue>(stopping_state, stopping_contribution);
    }
    const auto charge_error =
        stopping_global_error.template error<ChargeValue>(stopping_state);

    return ChargeIntegrateResult{
        .charge = value.charge,
        .charge_error = charge_error,
        .dcharge_dmu = value.derivative,
        .work = evaluations,
        .refinements = 0,
        .n_active_simplices = n_active_simplices(),
        .n_active_vertices = n_active_vertices(),
        .converged = charge_error <= options.target_error,
    };
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
