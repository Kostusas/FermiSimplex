#include <fermisimplex/integration.h>

#include "integration/charge.h"
#include "integration/charge_profile.h"
#include "integration/density.h"

#include <adaptivesimplex/adaptive/adaptive_loop.h>
#include <adaptivesimplex/adaptive/evaluation.h>
#include <adaptivesimplex/adaptive/simplex_integrand.h>

#include <algorithm>
#include <chrono>
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
using integration_detail::DensityRule;

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
    void add_local_estimate(
        state_type<Value> &state,
        const Value &local_estimate
    ) const {
        state += simplex_error(local_estimate);
    }

    template <class Value>
    void remove_local_estimate(
        state_type<Value> &state,
        const Value &local_estimate
    ) const {
        state -= simplex_error(local_estimate);
    }

    template <class Value>
    double error(const state_type<Value> &state, const Value &) const {
        return std::max(0.0, state);
    }
};

struct ChargeSimplexError {
    double operator()(const ChargeContribution &local_estimate) const {
        return local_estimate.estimated_error;
    }

    template <class Value, class Cache>
    double operator()(
        const adaptive::SimplexEstimateContext<Value, Cache> &estimate
    ) const {
        return estimate.coarse.estimated_error +
               estimate.correction.estimated_error;
    }
};

struct DensitySimplexError {
    bool has_preview = true;

    double operator()(const DensityRule::Value &local_estimate) const {
        return has_preview ? local_estimate.max_abs() : 0.0;
    }

    template <class Value, class Cache>
    double operator()(
        const adaptive::SimplexEstimateContext<Value, Cache> &estimate
    ) const {
        return estimate.correction.max_abs();
    }
};

auto charge_integrand(
    SpectralMesh &mesh,
    double mu,
    std::uint32_t error_depth,
    std::int64_t &simplex_visits,
    ChargeErrorStats &error_stats,
    integration_detail::ChargeProfile *profile
) {
    auto error_estimator = std::make_shared<
        integration_detail::ChargeErrorEstimator
    >(
        mesh,
        mu,
        error_depth,
        error_stats,
        profile
    );
    return adaptive::simplex_integrand(
        mesh.eigensystems(),
        [&mesh, profile](std::span<const double> point) {
            if (profile == nullptr) {
                return mesh.spectrum(point);
            }
            const auto started = std::chrono::steady_clock::now();
            auto spectrum = mesh.spectrum(point);
            profile->vertex_cache_seconds +=
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - started
                ).count();
            return spectrum;
        },
        [&mesh, mu, &simplex_visits, error_estimator, profile](
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
                *error_estimator,
                profile
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
    DensityRule &rule,
    std::int64_t &simplex_visits,
    std::uint32_t preview_depth
) {
    const auto simplex_error = DensitySimplexError{
        .has_preview = preview_depth > 0,
    };
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
        adaptive::estimation_policies{
            SumSimplexErrors<DensitySimplexError>{simplex_error},
            simplex_error,
        }
    );
}

CurrentMeshChargeResult current_mesh_charge(
    SpectralMesh &mesh,
    double mu
) {
    auto &geometry = mesh.geometry();
    const auto active = geometry.simplices().active_simplices();
    const auto simplex_ids = std::vector<core::SimplexId>(active.begin(), active.end());
    auto evaluations = std::int64_t{0};
    auto value = ChargeContribution{};
    auto integrand = adaptive::simplex_integrand(
        mesh.eigensystems(),
        [&mesh](std::span<const double> point) {
            return mesh.spectrum(point);
        },
        [&mesh, mu](
            const core::Geometry &current_geometry,
            core::SimplexId simplex_id,
            EigensystemCache &
        ) {
            return integration_detail::band_charge_on_simplex(
                mu,
                mesh,
                current_geometry,
                simplex_id
            );
        }
    );

    adaptive::evaluate_vertices(
        geometry,
        integrand,
        simplex_ids,
        0,
        evaluations
    );
    for (const auto simplex_id : simplex_ids) {
        value += integrand.simplex_contribution(geometry, simplex_id);
    }
    return CurrentMeshChargeResult{
        .value = value.value,
        .dcharge_dmu = value.dcharge_dmu,
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

adaptive::IntegrationResult<DensityRule::Value> integrate_density_rule(
    SpectralMesh &mesh,
    double mu,
    DensityRule &rule,
    const adaptive::Options &options,
    std::int64_t &simplex_visits
) {
    auto integrand = density_integrand(
        mesh,
        mu,
        rule,
        simplex_visits,
        options.preview_depth
    );
    return adaptive::run(mesh.geometry(), integrand, options);
}

DensityComponentsResult density_components_result(
    const SpectralMesh &mesh,
    const adaptive::IntegrationResult<DensityRule::Value> &raw,
    std::int64_t simplex_visits
) {
    return DensityComponentsResult{
        .values = raw.integral.values(),
        .stopping_error = raw.stopping_error,
        .stats = stats(
            mesh,
            raw.evaluations,
            simplex_visits,
            raw.refinements,
            raw.converged
        ),
    };
}

DensityMatrixResult density_matrix_result(
    const SpectralMesh &mesh,
    const adaptive::IntegrationResult<DensityRule::Value> &raw,
    const DensityRule &rule,
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

ChargeResult integrate_charge_impl(
    SpectralMesh &mesh,
    double mu,
    const adaptive::Options &options,
    std::uint32_t error_depth,
    integration_detail::ChargeProfile *profile
) {
    validate_mu(mu);
    validate_options(options);
    auto total_started = std::chrono::steady_clock::time_point{};
    if (profile != nullptr) {
        *profile = integration_detail::ChargeProfile{};
        total_started = std::chrono::steady_clock::now();
    }

    auto simplex_visits = std::int64_t{0};
    auto error_stats = ChargeErrorStats{};
    auto integrand = charge_integrand(
        mesh,
        mu,
        error_depth,
        simplex_visits,
        error_stats,
        profile
    );
    auto charge_options = options;
    charge_options.preview_depth = 0;
    const auto raw = adaptive::run(
        mesh.geometry(),
        integrand,
        charge_options
    );
    auto result = charge_result(
        mesh,
        raw,
        simplex_visits,
        error_stats
    );
    if (profile != nullptr) {
        profile->total_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - total_started
        ).count();
    }
    return result;
}

}  // namespace

ChargeResult integrate_charge(
    SpectralMesh &mesh,
    double mu,
    const adaptive::Options &options,
    std::uint32_t error_depth
) {
    return integrate_charge_impl(
        mesh,
        mu,
        options,
        error_depth,
        nullptr
    );
}

namespace integration_detail {

ChargeResult integrate_charge_profiled(
    SpectralMesh &mesh,
    double mu,
    const adaptive::Options &options,
    std::uint32_t error_depth,
    ChargeProfile &profile
) {
    return integrate_charge_impl(mesh, mu, options, error_depth, &profile);
}

}  // namespace integration_detail

CurrentMeshChargeResult estimate_charge_on_current_mesh(
    SpectralMesh &mesh,
    double mu
) {
    validate_mu(mu);
    return current_mesh_charge(mesh, mu);
}

DensityComponentsResult integrate_density_components(
    SpectralMesh &mesh,
    double mu,
    std::vector<LatticeVector> lattice_vectors,
    std::vector<DensityComponent> components,
    const adaptive::Options &options
) {
    validate_mu(mu);
    validate_options(options);
    auto rule = DensityRule(
        mesh.ndim(),
        mesh.ndof(),
        std::move(lattice_vectors),
        std::move(components)
    );
    auto simplex_visits = std::int64_t{0};
    const auto raw = integrate_density_rule(
        mesh,
        mu,
        rule,
        options,
        simplex_visits
    );
    return density_components_result(mesh, raw, simplex_visits);
}

DensityMatrixResult integrate_density_matrix(
    SpectralMesh &mesh,
    double mu,
    std::vector<LatticeVector> lattice_vectors,
    const adaptive::Options &options
) {
    validate_mu(mu);
    validate_options(options);
    auto rule = DensityRule(
        mesh.ndim(),
        mesh.ndof(),
        std::move(lattice_vectors)
    );
    auto simplex_visits = std::int64_t{0};
    const auto raw = integrate_density_rule(
        mesh,
        mu,
        rule,
        options,
        simplex_visits
    );
    return density_matrix_result(mesh, raw, rule, simplex_visits);
}

}  // namespace fermisimplex
