#include <fermisimplex/integration.h>
#include <fermisimplex/occupation.h>

#include "integration/charge.h"
#include "integration/density.h"
#include "occupation/enclosure.h"
#include "core/tight_binding_access.h"

#include <adaptivesimplex/adaptive/adaptive_loop.h>
#include <adaptivesimplex/adaptive/evaluation.h>
#include <adaptivesimplex/adaptive/simplex_integrand.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
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

auto charge_integrand(
    SpectralMesh &mesh, double mu, std::uint32_t error_depth,
    std::int64_t &simplex_visits, ChargeErrorStats &error_stats
) {
    return adaptive::simplex_integrand(
        mesh.eigensystems(),
        [&mesh](std::span<const double> point) { return mesh.spectrum(point); },
        [&mesh, mu, &simplex_visits, &error_stats, error_depth](
            const core::Geometry &geometry, core::SimplexId simplex_id,
            EigensystemCache &
        ) {
            ++simplex_visits;
            auto result = integration_detail::band_charge_on_simplex(
                mu, mesh, geometry, simplex_id);
            const auto enclosure = enclose_occupation(
                mesh, simplex_id, mu, error_depth, error_stats);
            result.estimated_error = std::max(
                std::abs(result.value - enclosure.charge_lower),
                std::abs(enclosure.charge_upper - result.value));
            result.density_cut_error = enclosure.density_cut_error;
            if (!enclosure.fixed_occupation()) {
                if (occupation_detail::vertex_occupation(mesh, simplex_id, mu) !=
                    occupation_detail::VertexOccupation::uniform)
                    result.visible_gapless_simplices = 1;
                else
                    result.inconclusive_simplices = 1;
            }
            return result;
        },
        adaptive::estimation_policies<
            SumSimplexErrors<ChargeSimplexError>, ChargeSimplexError>{}
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
        .density_cut_error = integration_detail::validated_density_cut_error(
            value.density_cut_error, value.value
        ),
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

}  // namespace

ChargeResult integrate_charge(
    SpectralMesh &mesh, double mu, const adaptive::Options &options,
    std::uint32_t error_depth
) {
    validate_mu(mu);
    validate_options(options);
    auto simplex_visits = std::int64_t{0};
    auto error_stats = ChargeErrorStats{};
    auto integrand = charge_integrand(mesh, mu, error_depth, simplex_visits, error_stats);
    auto charge_options = options;
    charge_options.preview_depth = 0;
    // Every cell of a constant matrix has the same error per unit volume.
    // Refinement cannot reduce roundoff uncertainty or level-rounding error.
    if (core_detail::TightBindingModelAccess::constant_spectrum_roundoff(mesh.model()))
        charge_options.max_refinements = 0;
    const auto raw = adaptive::run(mesh.geometry(), integrand, charge_options);
    return charge_result(mesh, raw, simplex_visits, error_stats);
}

CurrentMeshChargeResult estimate_charge_on_current_mesh(
    SpectralMesh &mesh,
    double mu
) {
    validate_mu(mu);
    return current_mesh_charge(mesh, mu);
}

DensityComponentsResult estimate_density_on_current_mesh(
    SpectralMesh &mesh, double mu, std::vector<LatticeVector> lattice_vectors,
    std::vector<DensityComponent> components
) {
    validate_mu(mu);
    const DensityRule rule(mesh.ndim(), mesh.ndof(), std::move(lattice_vectors),
                           std::move(components));
    std::int64_t evaluations = 0;
    auto &cache = mesh.eigensystems();
    for (const auto vertex : mesh.active_vertex_ids()) {
        if (!cache.contains(vertex)) {
            const auto point = mesh.geometry().vertices().dyadic_vertex(vertex).to_point();
            cache.insert(vertex, mesh.spectrum(point));
            ++evaluations;
        }
    }
    DensityRule::Value total(rule.output_size());
    for (const auto id : mesh.geometry().simplices().active_simplices())
        total += rule.on_simplex(mu, mesh, mesh.geometry(), id);
    return {total.values(), 0., stats(mesh, evaluations, mesh.active_simplices(), 0, true)};
}

DensityComponentsResult integrate_density_components(
    SpectralMesh &mesh, double mu, std::vector<LatticeVector> lattice_vectors,
    std::vector<DensityComponent> components, double target_error,
    std::int64_t max_refinements, std::uint32_t max_degree,
    std::int64_t max_h_refinements
) {
    validate_mu(mu);
    const DensityRule rule(mesh.ndim(), mesh.ndof(), std::move(lattice_vectors),
                           std::move(components));
    return integration_detail::integrate_density_cubature(
        mesh, mu, rule, target_error, max_refinements, max_degree, max_h_refinements);
}

DensityMatrixResult integrate_density_matrix(
    SpectralMesh &mesh, double mu, std::vector<LatticeVector> lattice_vectors,
    double target_error, std::int64_t max_refinements, std::uint32_t max_degree,
    std::int64_t max_h_refinements
) {
    validate_mu(mu);
    const DensityRule rule(mesh.ndim(), mesh.ndof(), std::move(lattice_vectors));
    auto result = integration_detail::integrate_density_cubature(
        mesh, mu, rule, target_error, max_refinements, max_degree, max_h_refinements);
    return {std::move(result.values), result.stopping_error,
            rule.lattice_vector_count(), rule.ndof(), result.stats};
}

}  // namespace fermisimplex
