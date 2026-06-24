#include "integration/charge.h"

#include "certificate/simplex_certificate.h"

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
    bool certify
) {
    const auto &simplex = geometry.simplices().simplex(simplex_id);
    ChargeValue result;
    if (certify) {
        const auto certificate = simplex_certificate::certify_simplex_gap(
            mu,
            geometry,
            simplex_id,
            cache,
            0.0,
            workspace.tol()
        );
        if (certificate.status == simplex_certificate::SimplexCertificateStatus::Inconclusive) {
            const auto occupation = std::min(certificate.vertex_occupation, workspace.ndof());
            const auto unresolved_occupation_bound = std::max(
                occupation,
                workspace.ndof() - occupation
            );
            result.certificate_error =
                static_cast<double>(unresolved_occupation_bound) * simplex.volume;
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
