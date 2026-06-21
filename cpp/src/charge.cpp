#include "lineartetrahedron/charge.h"

#include "lineartetrahedron/weyl.h"

#include <adaptivesimplex/cut/simplex_moments.h>

#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

namespace lineartetrahedron {
namespace core = adaptivesimplex::core;
namespace cut = adaptivesimplex::cut;

namespace {

double occupied_fraction_derivative(
    std::span<const double> energies,
    double mu
) {
    const size_t dimension = energies.size() - 1;
    double result = 0.0;
    for (size_t i = 0; i < energies.size(); ++i) {
        const double distance = mu - energies[i];
        if (distance <= 0.0) {
            continue;
        }
        double denominator = 1.0;
        for (size_t j = 0; j < energies.size(); ++j) {
            if (j != i) {
                denominator *= energies[j] - energies[i];
            }
        }
        result += std::pow(distance, static_cast<double>(dimension - 1)) / denominator;
    }
    return static_cast<double>(dimension) * result;
}

bool strictly_ordered(std::span<const double> energies, double tol) {
    for (size_t index = 1; index < energies.size(); ++index) {
        if (energies[index] - energies[index - 1] <= tol) {
            return false;
        }
    }
    return true;
}

std::vector<double> sorted_band_energies(
    const core::Geometry &geometry,
    core::SimplexId simplex_id,
    const core::VertexCache<VertexSpectra> &cache,
    size_t band
) {
    const auto &simplex = geometry.simplices().simplex(simplex_id);
    std::vector<double> energies;
    energies.reserve(simplex.vertex_ids.size());
    for (const auto vertex_id : simplex.vertex_ids) {
        energies.push_back(cache.get(vertex_id).eigenvalues[band]);
    }
    std::stable_sort(energies.begin(), energies.end());
    return energies;
}

}  // namespace

ChargeValue charge_on_simplex(
    double mu,
    const IntegrationWorkspace &workspace,
    const core::Geometry &geometry,
    core::SimplexId simplex_id,
    const core::VertexCache<VertexSpectra> &cache,
    double weyl_indicator_error,
    ChargeWeylMarkerCache *weyl_marker_cache
) {
    const auto &simplex = geometry.simplices().simplex(simplex_id);
    ChargeValue result;
    const auto use_weyl_marker =
        weyl_indicator_error > 0.0 && workspace.reduced_lipschitz_bound() > 0.0;
    if (use_weyl_marker) {
        if (weyl_marker_cache) {
            if (const auto cached = weyl_marker_cache->find(simplex_id);
                cached != weyl_marker_cache->end()) {
                result.weyl_indicator_error = cached->second;
            } else {
                result.weyl_indicator_error =
                    weyl::simplex_has_weyl_marker(
                        mu,
                        workspace.model(),
                        geometry,
                        simplex_id,
                        workspace.tol(),
                        [&](core::VertexId vertex_id, size_t band_index) {
                            return cache.get(vertex_id).eigenvalues[band_index];
                        }
                    )
                        ? weyl_indicator_error
                        : 0.0;
                weyl_marker_cache->emplace(simplex_id, result.weyl_indicator_error);
            }
        } else if (weyl::simplex_has_weyl_marker(
                       mu,
                       workspace.model(),
                       geometry,
                       simplex_id,
                       workspace.tol(),
                       [&](core::VertexId vertex_id, size_t band_index) {
                           return cache.get(vertex_id).eigenvalues[band_index];
                       }
                   )) {
            result.weyl_indicator_error = weyl_indicator_error;
        }
    }

    for (size_t band = 0; band < workspace.ndof(); ++band) {
        const auto moments = cut::simplex_moments(
            geometry,
            simplex_id,
            [&](core::VertexId vertex_id) {
                return cache.get(vertex_id).eigenvalues[band];
            },
            cut::LevelOptions{.level = mu, .level_tolerance = workspace.tol()}
        );

        if (moments.kind == cut::SimplexCutKind::on_level) {
            result.charge += 0.5 * simplex.volume;
            continue;
        }

        result.charge += moments.volume;
        if (moments.kind != cut::SimplexCutKind::partial) {
            continue;
        }

        const auto energies = sorted_band_energies(geometry, simplex_id, cache, band);
        const auto energy_span = std::span<const double>(energies.data(), energies.size());
        if (strictly_ordered(energy_span, workspace.tol())) {
            result.derivative +=
                simplex.volume * occupied_fraction_derivative(energy_span, mu);
        }
    }
    return result;
}

}  // namespace lineartetrahedron
