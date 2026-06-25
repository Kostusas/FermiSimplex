#include "integration/runtime.h"

#include "integration/charge.h"

#include <adaptivesimplex/adaptive/adaptive_loop.h>
#include <adaptivesimplex/adaptive/evaluation.h>
#include <adaptivesimplex/adaptive/types.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
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
        auto correction_contribution = estimate.correction;
        correction_contribution.certificate_error = estimate.preview.certificate_error;
        return correction_contribution;
    }
    template <class IntegralValue>
    void add(state_type<IntegralValue> &state, const state_type<IntegralValue> &value) const {
        state += value;
    }
    template <class IntegralValue>
    void remove(state_type<IntegralValue> &state, const state_type<IntegralValue> &value) const {
        state -= value;
    }
    template <class IntegralValue>
    double error(const state_type<IntegralValue> &correction_state) const {
        // This state accumulates charge corrections, not the physical charge.
        return std::abs(correction_state.charge) + correction_state.certificate_error;
    }
};

struct ChargeRefinementScore {
    template <class Context> double operator()(const Context &context) const {
        return std::abs(context.correction.charge) + context.preview.certificate_error;
    }
};

struct ChargeEstimateContext {
    const ChargeValue &preview;
    const ChargeValue &correction;
};

class ChargeIntegrand {
public:
    using vertex_value_type = VertexSpectra;
    using integral_value_type = ChargeValue;
    using cache_type = core::VertexCache<VertexSpectra>;

    ChargeIntegrand(
        IntegrationWorkspace &workspace,
        double mu,
        bool certify_preview
    ) : workspace_(workspace),
        cache_(workspace.cache()),
        mu_(mu),
        certify_preview_(certify_preview) {}

    cache_type &cache() {
        return cache_;
    }

    auto stopping_global_error() const {
        return ChargeStoppingError{};
    }

    vertex_value_type evaluate_vertex(core::Geometry &geometry, core::VertexId vertex_id) const {
        const auto point = geometry.vertices().dyadic_vertex(vertex_id).to_point();
        return workspace_.evaluate_vertex(std::span<const double>(point.data(), point.size()));
    }

    std::vector<adaptive::SimplexEstimate<integral_value_type>> estimate_simplices(
        core::Geometry &geometry,
        std::span<const core::SimplexId> simplex_ids,
        std::uint32_t preview_depth
    ) const {
        std::vector<adaptive::SimplexEstimate<integral_value_type>> estimates;
        estimates.reserve(simplex_ids.size());
        for (const auto simplex_id : simplex_ids) {
            auto coarse = charge_on_simplex(
                mu_,
                workspace_,
                geometry,
                simplex_id,
                cache_,
                false
            );

            auto preview = integral_value_type{};
            const auto preview_ids = geometry.preview_active(simplex_id, preview_depth);
            for (const auto preview_id : preview_ids) {
                preview += charge_on_simplex(
                    mu_,
                    workspace_,
                    geometry,
                    preview_id,
                    cache_,
                    certify_preview_
                );
            }

            auto correction = preview;
            correction -= coarse;
            const auto context = ChargeEstimateContext{
                .preview = preview,
                .correction = correction,
            };
            const auto refinement_score =
                static_cast<double>(ChargeRefinementScore{}(context));

            estimates.push_back(adaptive::SimplexEstimate<integral_value_type>{
                .coarse = std::move(coarse),
                .preview = std::move(preview),
                .correction = std::move(correction),
                .refinement_score = refinement_score,
            });
        }
        return estimates;
    }

private:
    IntegrationWorkspace &workspace_;
    cache_type &cache_;
    double mu_ = 0.0;
    bool certify_preview_ = false;
};

ChargeIntegrand make_charge_integrand(
    IntegrationWorkspace &workspace,
    double mu,
    bool certify_preview
) {
    return ChargeIntegrand(workspace, mu, certify_preview);
}

}  // namespace

ChargeIntegrateResult IntegrationRuntime::integrate_charge(
    double mu,
    const adaptive::Options &options
) {
    auto integrand = make_charge_integrand(
        workspace_,
        mu,
        true
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
    bool certify
) {
    auto integrand = make_charge_integrand(
        workspace_,
        mu,
        certify
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
