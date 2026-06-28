#include "integration/charge.h"

#include "certificate/simplex_certificate.h"

#include <adaptivesimplex/cut/simplex_moments.h>

#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
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

double simplex_diameter(
    const core::Geometry &geometry,
    const core::Simplex &simplex
) {
    auto max_squared = 0.0;
    for (size_t left = 0; left < simplex.vertex_ids.size(); ++left) {
        const auto left_point =
            geometry.vertices().dyadic_vertex(simplex.vertex_ids[left]).to_point();
        for (size_t right = left + 1; right < simplex.vertex_ids.size(); ++right) {
            const auto right_point =
                geometry.vertices().dyadic_vertex(simplex.vertex_ids[right]).to_point();
            auto squared = 0.0;
            for (size_t axis = 0; axis < geometry.ndim(); ++axis) {
                const auto delta = left_point[axis] - right_point[axis];
                squared += delta * delta;
            }
            max_squared = std::max(max_squared, squared);
        }
    }
    return std::sqrt(max_squared);
}

double simplex_energy_bound(
    const core::Geometry &geometry,
    const core::Simplex &simplex,
    double hessian_bound,
    double anharmonicity_bound
) {
    if (hessian_bound < 0.0 || !std::isfinite(hessian_bound)) {
        throw std::runtime_error("charge_on_simplex: hessian_bound must be finite and non-negative");
    }
    if (anharmonicity_bound < 0.0 || !std::isfinite(anharmonicity_bound)) {
        throw std::runtime_error(
            "charge_on_simplex: anharmonicity_bound must be finite and non-negative"
        );
    }
    const auto diameter = simplex_diameter(geometry, simplex);
    return 0.5 * hessian_bound * diameter * diameter +
           0.5 * anharmonicity_bound * diameter * diameter * diameter;
}

double occupied_band_volume(
    const core::Geometry &geometry,
    core::SimplexId simplex_id,
    const core::VertexCache<VertexSpectra> &cache,
    size_t band,
    double mu,
    double tol
) {
    const auto &simplex = geometry.simplices().simplex(simplex_id);
    const auto moments = cut::simplex_moments(
        geometry,
        simplex_id,
        [&](core::VertexId vertex_id) {
            return cache.get(vertex_id).eigenvalues[band];
        },
        cut::LevelOptions{.level = mu, .level_tolerance = tol}
    );
    if (moments.kind == cut::SimplexCutKind::on_level) {
        return 0.5 * simplex.volume;
    }
    return moments.volume;
}

double visible_gapless_certificate_error(
    double mu,
    const core::Geometry &geometry,
    core::SimplexId simplex_id,
    const core::VertexCache<VertexSpectra> &cache,
    const simplex_certificate::SimplexCertificate &certificate,
    double tol
) {
    auto result = 0.0;
    const auto lower_mu = mu - certificate.energy_bound;
    const auto upper_mu = mu + certificate.energy_bound;
    for (size_t band = certificate.occupation_bounds.lower;
         band < certificate.occupation_bounds.upper;
         ++band) {
        const auto lower_volume =
            occupied_band_volume(geometry, simplex_id, cache, band, lower_mu, tol);
        const auto upper_volume =
            occupied_band_volume(geometry, simplex_id, cache, band, upper_mu, tol);
        result += std::max(0.0, upper_volume - lower_volume);
    }
    return result;
}

}  // namespace

ChargeValue charge_on_simplex(
    double mu,
    const IntegrationWorkspace &workspace,
    const core::Geometry &geometry,
    core::SimplexId simplex_id,
    const core::VertexCache<VertexSpectra> &cache,
    bool certify,
    ChargeCertificateCache *certificate_cache,
    double hessian_bound,
    double anharmonicity_bound
) {
    const auto &simplex = geometry.simplices().simplex(simplex_id);
    const auto energy_bound =
        simplex_energy_bound(geometry, simplex, hessian_bound, anharmonicity_bound);
    return charge_on_simplex_with_energy_bound(
        mu,
        workspace,
        geometry,
        simplex_id,
        cache,
        certify,
        certificate_cache,
        energy_bound
    );
}

ChargeValue charge_on_simplex_with_energy_bound(
    double mu,
    const IntegrationWorkspace &workspace,
    const core::Geometry &geometry,
    core::SimplexId simplex_id,
    const core::VertexCache<VertexSpectra> &cache,
    bool certify,
    ChargeCertificateCache *certificate_cache,
    double energy_bound
) {
    if (energy_bound < 0.0 || !std::isfinite(energy_bound)) {
        throw std::runtime_error(
            "charge_on_simplex_with_energy_bound: energy_bound must be finite and non-negative"
        );
    }
    const auto &simplex = geometry.simplices().simplex(simplex_id);
    ChargeValue result;
    if (certify) {
        auto certificate = simplex_certificate::SimplexCertificate{};
        const auto *cached_certificate =
            certificate_cache == nullptr
                ? nullptr
                : certificate_cache->find(simplex_id, mu, energy_bound);
        if (cached_certificate != nullptr) {
            certificate = *cached_certificate;
        } else {
            certificate = simplex_certificate::certify_mesh_simplex_with_energy_bound(
                geometry,
                simplex_id,
                cache,
                mu,
                energy_bound,
                workspace.tol(),
                true
            );
            if (certificate_cache != nullptr) {
                certificate_cache->insert(simplex_id, certificate);
            }
        }
        if (certificate.status == simplex_certificate::SimplexCertificateStatus::Inconclusive) {
            result.certificate_error =
                static_cast<double>(simplex_certificate::occupation_width(certificate)) *
                simplex.volume;
        } else if (
            certificate.status == simplex_certificate::SimplexCertificateStatus::VisibleGapless
        ) {
            result.certificate_error = visible_gapless_certificate_error(
                mu,
                geometry,
                simplex_id,
                cache,
                certificate,
                workspace.tol()
            );
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
