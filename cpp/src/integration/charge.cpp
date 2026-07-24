#include "integration/charge.h"

#include "certification/mesh_certificate.h"
#include "integration/projected_error.h"

#include <adaptivesimplex/cut/simplex_moments.h>

#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

namespace fermisimplex::integration_detail {
namespace cert = certification;
namespace core = adaptivesimplex::core;
namespace cut = adaptivesimplex::cut;

namespace {

double truncated_power_derivative(
    double knot,
    double mu,
    std::size_t power,
    std::size_t derivative
) {
    if (mu <= knot || derivative > power) {
        return 0.0;
    }

    auto coefficient = 1.0;
    for (std::size_t index = 0; index < derivative; ++index) {
        coefficient *= -static_cast<double>(power - index) /
                       static_cast<double>(index + 1);
    }
    return coefficient *
           std::pow(mu - knot, static_cast<double>(power - derivative));
}

double occupied_fraction_derivative(
    std::vector<double> energies,
    double mu,
    double tolerance
) {
    const auto dimension = energies.size() - 1;
    const auto energy_scale = std::max(
        {1.0, std::abs(energies.front()), std::abs(energies.back())}
    );
    for (std::size_t index = 1; index < energies.size(); ++index) {
        if (energies[index] - energies[index - 1] <= tolerance * energy_scale) {
            energies[index] = energies[index - 1];
        }
    }

    std::vector<double> divided_differences(energies.size());
    for (std::size_t index = 0; index < energies.size(); ++index) {
        divided_differences[index] = truncated_power_derivative(
            energies[index],
            mu,
            dimension - 1,
            0
        );
    }
    for (std::size_t order = 1; order <= dimension; ++order) {
        for (std::size_t left = 0; left + order < energies.size(); ++left) {
            const auto right = left + order;
            if (energies[right] == energies[left]) {
                divided_differences[left] = truncated_power_derivative(
                    energies[left],
                    mu,
                    dimension - 1,
                    order
                );
            } else {
                divided_differences[left] =
                    (divided_differences[left + 1] - divided_differences[left]) /
                    (energies[right] - energies[left]);
            }
        }
    }
    const auto orientation = dimension % 2 == 0 ? 1.0 : -1.0;
    return std::max(
        0.0,
        orientation * static_cast<double>(dimension) * divided_differences.front()
    );
}

std::vector<double> band_energies(
    const core::Geometry &geometry,
    core::SimplexId simplex_id,
    const EigensystemCache &cache,
    std::size_t band
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

double occupied_band_volume(
    const core::Geometry &geometry,
    core::SimplexId simplex_id,
    const EigensystemCache &cache,
    std::size_t band,
    double mu,
    double tolerance
) {
    const auto &simplex = geometry.simplices().simplex(simplex_id);
    const auto moments = cut::simplex_moments(
        geometry,
        simplex_id,
        [&](core::VertexId vertex_id) {
            return cache.get(vertex_id).eigenvalues[band];
        },
        cut::LevelOptions{.level = mu, .level_tolerance = tolerance}
    );
    return moments.kind == cut::SimplexCutKind::on_level
        ? 0.5 * simplex.volume
        : moments.volume;
}

double projected_charge_error(
    double mu,
    const SpectralMesh &mesh,
    const core::Geometry &geometry,
    core::SimplexId simplex_id,
    cert::OccupationBounds occupation_bounds
) {
    const auto &cache = mesh.eigensystems();
    const auto residual = estimate_projected_residual(
        mesh,
        simplex_id,
        occupation_bounds.lower,
        occupation_bounds.upper
    );

    // A positive projected spectral difference raises actual energies relative
    // to the interpolated bands and reduces occupation; a negative difference
    // lowers them and increases occupation. The asymmetric pairing is:
    const auto guaranteed_occupation_mu = mu - residual.positive_estimate;
    const auto possible_occupation_mu = mu + residual.negative_estimate;

    auto error = 0.0;
    for (auto band = occupation_bounds.lower;
         band < occupation_bounds.upper;
         ++band) {
        const auto guaranteed_volume = occupied_band_volume(
            geometry,
            simplex_id,
            cache,
            band,
            guaranteed_occupation_mu,
            mesh.tolerance()
        );
        const auto possible_volume = occupied_band_volume(
            geometry,
            simplex_id,
            cache,
            band,
            possible_occupation_mu,
            mesh.tolerance()
        );
        error += std::max(0.0, possible_volume - guaranteed_volume);
    }
    return error;
}

}  // namespace

ChargeContribution &ChargeContribution::operator+=(
    const ChargeContribution &other
) noexcept {
    value += other.value;
    dcharge_dmu += other.dcharge_dmu;
    projected_error += other.projected_error;
    certified_error_bound += other.certified_error_bound;
    visible_gapless_simplices += other.visible_gapless_simplices;
    inconclusive_simplices += other.inconclusive_simplices;
    return *this;
}

ChargeContribution &ChargeContribution::operator-=(
    const ChargeContribution &other
) noexcept {
    value -= other.value;
    dcharge_dmu -= other.dcharge_dmu;
    projected_error -= other.projected_error;
    certified_error_bound -= other.certified_error_bound;
    visible_gapless_simplices -= other.visible_gapless_simplices;
    inconclusive_simplices -= other.inconclusive_simplices;
    return *this;
}

ChargeContribution charge_on_simplex(
    double mu,
    const SpectralMesh &mesh,
    const core::Geometry &geometry,
    core::SimplexId simplex_id,
    double curvature_bound
) {
    const auto &simplex = geometry.simplices().simplex(simplex_id);
    const auto &cache = mesh.eigensystems();
    auto result = ChargeContribution{};

    const auto certificate = cert::certify_mesh_simplex(
        mesh,
        simplex_id,
        mu,
        mesh.linearization_error_bound(simplex_id, curvature_bound),
        mesh.tolerance()
    );
    const auto occupation_bounds = certificate.occupation_bounds;
    result.certified_error_bound =
        static_cast<double>(cert::occupation_width(certificate)) *
        simplex.volume;
    if (certificate.status == cert::SimplexCertificateStatus::VisibleGapless) {
        result.visible_gapless_simplices = 1;
    } else if (certificate.status == cert::SimplexCertificateStatus::Inconclusive) {
        result.inconclusive_simplices = 1;
    }

    if (occupation_bounds.lower < occupation_bounds.upper) {
        result.projected_error = projected_charge_error(
            mu,
            mesh,
            geometry,
            simplex_id,
            occupation_bounds
        );
    }

    for (std::size_t band = 0; band < mesh.ndof(); ++band) {
        const auto moments = cut::simplex_moments(
            geometry,
            simplex_id,
            [&](core::VertexId vertex_id) {
                return cache.get(vertex_id).eigenvalues[band];
            },
            cut::LevelOptions{.level = mu, .level_tolerance = mesh.tolerance()}
        );

        if (moments.kind == cut::SimplexCutKind::on_level) {
            result.value += 0.5 * simplex.volume;
            continue;
        }

        result.value += moments.volume;
        const auto energies = band_energies(geometry, simplex_id, cache, band);
        result.dcharge_dmu += simplex.volume * occupied_fraction_derivative(
            energies,
            mu,
            mesh.tolerance()
        );
    }
    return result;
}

}  // namespace fermisimplex::integration_detail
