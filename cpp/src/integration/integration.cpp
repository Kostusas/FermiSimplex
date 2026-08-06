#include <fermisimplex/integration.h>

#include "integration/charge.h"
#include "integration/density.h"

#include <adaptivesimplex/adaptive/adaptive_loop.h>
#include <adaptivesimplex/adaptive/evaluation.h>
#include <adaptivesimplex/adaptive/simplex_integrand.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fermisimplex {
namespace adaptive = adaptivesimplex::adaptive;
namespace core = adaptivesimplex::core;
using integration_detail::ChargeContribution;
using integration_detail::DensityMatrixRule;

namespace {

void validate_target_error(double target_error) {
    if (!std::isfinite(target_error) || target_error < 0.0) {
        throw std::runtime_error("integration target_error must be finite and non-negative");
    }
}

void validate_options(const adaptive::Options &options) {
    validate_target_error(options.target_error);
    if (options.max_refinements < -1) {
        throw std::runtime_error("max_refinements must be -1 or non-negative");
    }
    if (options.min_refinement_batch_size == 0) {
        throw std::runtime_error("min_refinement_batch_size must be positive");
    }
    if (options.max_refinement_batch_size < options.min_refinement_batch_size) {
        throw std::runtime_error(
            "max_refinement_batch_size must be at least min_refinement_batch_size"
        );
    }
}

void validate_mu(double mu) {
    if (!std::isfinite(mu)) {
        throw std::runtime_error("chemical potential mu must be finite");
    }
}

IntegrationStats stats(
    const SpectralMesh &mesh,
    std::int64_t evaluations,
    std::int64_t simplex_visits,
    std::int64_t refinements,
    bool target_reached
) {
    return IntegrationStats{
        .evaluations = evaluations,
        .simplex_visits = simplex_visits,
        .refinements = refinements,
        .cached_vertices = mesh.cached_vertices(),
        .active_simplices = mesh.active_simplices(),
        .active_vertices = mesh.active_vertices(),
        .target_reached = target_reached,
    };
}

template <class SimplexError>
struct SumSimplexErrors {
    SimplexError simplex_error;
    template <class Value> using state_type = double;

    template <class Value> state_type<Value> zero() const {
        return 0.0;
    }

    template <class Value>
    state_type<Value> contribution(const adaptive::SimplexEstimate<Value> &estimate) const {
        return simplex_error(estimate);
    }

    template <class Value>
    void add(state_type<Value> &state, const state_type<Value> &contribution) const {
        state += contribution;
    }

    template <class Value>
    void remove(state_type<Value> &state, const state_type<Value> &contribution) const {
        state -= contribution;
    }

    template <class Value> double error(const state_type<Value> &state) const {
        return std::max(0.0, state);
    }
};

struct ChargeSimplexError {
    template <class Estimate> double operator()(const Estimate &estimate) const {
        return estimate.preview.estimated_error;
    }
};

struct DensitySimplexError {
    template <class Estimate> double operator()(const Estimate &estimate) const {
        return estimate.correction.max_abs();
    }
};

auto charge_integrand(
    SpectralMesh &mesh,
    double mu,
    std::uint32_t error_depth,
    std::int64_t &simplex_visits,
    ChargeErrorStats &error_stats
) {
    auto error_estimator = std::make_shared<
        integration_detail::ChargeErrorEstimator
    >(mesh, mu, error_depth, error_stats);
    return adaptive::simplex_integrand(
        mesh.eigensystems(),
        [&mesh](std::span<const double> point) {
            return mesh.spectrum(point);
        },
        [&mesh, mu, &simplex_visits, error_estimator](
            const core::Geometry &geometry,
            core::SimplexId simplex_id,
            EigensystemCache &
        ) {
            ++simplex_visits;
            return integration_detail::charge_on_simplex(
                mu,
                mesh,
                geometry,
                simplex_id,
                *error_estimator
            );
        },
        adaptive::estimation_policies<
            SumSimplexErrors<ChargeSimplexError>,
            ChargeSimplexError
        >{}
    );
}

auto density_integrand(
    SpectralMesh &mesh,
    double mu,
    DensityMatrixRule &rule,
    std::int64_t &simplex_visits
) {
    return adaptive::simplex_integrand(
        mesh.eigensystems(),
        [&mesh](std::span<const double> point) {
            return mesh.spectrum(point);
        },
        [&mesh, mu, &rule, &simplex_visits](
            const core::Geometry &geometry,
            core::SimplexId simplex_id,
            EigensystemCache &
        ) {
            ++simplex_visits;
            return rule.on_simplex(mu, mesh, geometry, simplex_id);
        },
        adaptive::estimation_policies<
            SumSimplexErrors<DensitySimplexError>,
            DensitySimplexError
        >{}
    );
}

template <class Integrand>
adaptive::IntegrationResult<ChargeContribution> estimate_current_mesh_charge(
    SpectralMesh &mesh,
    Integrand &integrand,
    double target_error
) {
    auto &geometry = mesh.geometry();
    const auto active = geometry.simplices().active_simplices();
    const auto simplex_ids = std::vector<core::SimplexId>(active.begin(), active.end());
    auto evaluations = std::int64_t{0};
    auto value = ChargeContribution{};
    auto correction = ChargeContribution{};
    auto error_policy = integrand.stopping_global_error();
    auto error_state = error_policy.template zero<ChargeContribution>();

    for (const auto &estimate : adaptive::estimate_simplices(
             geometry,
             integrand,
             simplex_ids,
             0,
             evaluations
         )) {
        value += estimate.preview;
        correction += estimate.correction;
        const auto contribution =
            error_policy.template contribution<ChargeContribution>(estimate);
        error_policy.template add<ChargeContribution>(error_state, contribution);
    }

    const auto error = error_policy.template error<ChargeContribution>(error_state);
    return adaptive::IntegrationResult<ChargeContribution>{
        .integral = std::move(value),
        .correction = std::move(correction),
        .stopping_error = error,
        .evaluations = evaluations,
        .refinements = 0,
        .converged = error <= target_error,
    };
}

ChargeResult charge_result(
    const SpectralMesh &mesh,
    const adaptive::IntegrationResult<ChargeContribution> &raw,
    std::int64_t simplex_visits,
    const ChargeErrorStats &error_stats
) {
    const auto &value = raw.integral;
    return ChargeResult{
        .value = value.value,
        .stopping_error = raw.stopping_error,
        .dcharge_dmu = value.dcharge_dmu,
        .visible_gapless_simplices = value.visible_gapless_simplices,
        .inconclusive_simplices = value.inconclusive_simplices,
        .error_stats = error_stats,
        .stats = stats(
            mesh,
            raw.evaluations,
            simplex_visits,
            raw.refinements,
            raw.converged
        ),
    };
}

DensityMatrixResult density_result(
    const SpectralMesh &mesh,
    const adaptive::IntegrationResult<DensityMatrixRule::Value> &raw,
    const DensityMatrixRule &rule,
    std::int64_t simplex_visits
) {
    return DensityMatrixResult{
        .matrices = raw.integral.values(),
        .stopping_error = raw.stopping_error,
        .lattice_vector_count = rule.lattice_vector_count(),
        .ndof = rule.ndof(),
        .stats = stats(
            mesh,
            raw.evaluations,
            simplex_visits,
            raw.refinements,
            raw.converged
        ),
    };
}

}  // namespace

ChargeResult integrate_charge(
    SpectralMesh &mesh,
    double mu,
    const adaptive::Options &options,
    std::uint32_t error_depth
) {
    validate_mu(mu);
    validate_options(options);
    auto simplex_visits = std::int64_t{0};
    auto error_stats = ChargeErrorStats{};
    auto integrand = charge_integrand(
        mesh,
        mu,
        error_depth,
        simplex_visits,
        error_stats
    );
    auto charge_options = options;
    charge_options.preview_depth = 0;
    const auto raw = adaptive::run(
        mesh.geometry(),
        integrand,
        charge_options
    );
    return charge_result(
        mesh,
        raw,
        simplex_visits,
        error_stats
    );
}

ChargeResult estimate_charge_on_current_mesh(
    SpectralMesh &mesh,
    double mu,
    double target_error,
    std::uint32_t error_depth
) {
    validate_mu(mu);
    validate_target_error(target_error);
    auto simplex_visits = std::int64_t{0};
    auto error_stats = ChargeErrorStats{};
    auto integrand = charge_integrand(
        mesh,
        mu,
        error_depth,
        simplex_visits,
        error_stats
    );
    const auto raw = estimate_current_mesh_charge(
        mesh,
        integrand,
        target_error
    );
    return charge_result(
        mesh,
        raw,
        simplex_visits,
        error_stats
    );
}

DensityMatrixResult integrate_density_matrix(
    SpectralMesh &mesh,
    double mu,
    std::vector<LatticeVector> lattice_vectors,
    const adaptive::Options &options
) {
    validate_mu(mu);
    validate_options(options);
    if (options.preview_depth == 0) {
        throw std::runtime_error(
            "density-matrix preview_depth must be positive"
        );
    }
    auto rule = DensityMatrixRule(
        mesh.ndim(),
        mesh.ndof(),
        std::move(lattice_vectors)
    );
    auto simplex_visits = std::int64_t{0};
    auto integrand = density_integrand(mesh, mu, rule, simplex_visits);
    const auto raw = adaptive::run(mesh.geometry(), integrand, options);
    return density_result(mesh, raw, rule, simplex_visits);
}

}  // namespace fermisimplex
