#include "integration/runtime.h"

#include "integration/charge.h"

#include <adaptivesimplex/adaptive/adaptive_loop.h>
#include <adaptivesimplex/adaptive/evaluation.h>
#include <adaptivesimplex/adaptive/simplex_integrand.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
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
        contribution.certificate_error =
            std::max(estimate.coarse.certificate_error, estimate.preview.certificate_error);
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
        return std::abs(state.charge) + std::max(0.0, state.certificate_error);
    }
};

struct ChargeRefinementScore {
    template <class Context> double operator()(const Context &context) const {
        return std::max(
            std::abs(context.correction.charge),
            std::max(
                context.coarse.certificate_error,
                context.preview.certificate_error
            )
        );
    }
};

double certificate_error_for(const adaptive::Options &options) {
    return std::nextafter(options.target_error, std::numeric_limits<double>::infinity());
}

auto make_charge_integrand(
    IntegrationWorkspace &workspace,
    double mu,
    double certificate_error
) {
    return adaptive::simplex_integrand(
        workspace.cache(),
        [&workspace](std::span<const double> point) {
            return workspace.evaluate_vertex(point);
        },
        [&workspace, mu, certificate_error](
            const core::Geometry &geometry,
            core::SimplexId simplex_id,
            const core::VertexCache<VertexSpectra> &cache
        ) {
            return charge_on_simplex(
                mu,
                workspace,
                geometry,
                simplex_id,
                cache,
                certificate_error
            );
        },
        adaptive::estimation_policies<ChargeStoppingError, ChargeRefinementScore>{}
    );
}

}  // namespace

ChargeIntegrateResult IntegrationRuntime::integrate_charge(
    double mu,
    const adaptive::Options &options
) {
    const auto certificate_error = certificate_error_for(options);
    auto integrand = make_charge_integrand(
        workspace_,
        mu,
        certificate_error
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
    const adaptive::Options &options
) {
    const auto certificate_error = certificate_error_for(options);
    auto integrand = make_charge_integrand(
        workspace_,
        mu,
        certificate_error
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

}  // namespace lineartetrahedron
