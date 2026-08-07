#include "integration/charge.h"

#include "certification/mesh_certificate.h"

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

    auto divided_differences = std::vector<double>(energies.size());
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
        orientation * static_cast<double>(dimension) *
            divided_differences.front()
    );
}

std::vector<double> band_energies(
    const core::Geometry &geometry,
    core::SimplexId simplex_id,
    const EigensystemCache &cache,
    std::size_t band
) {
    const auto &simplex = geometry.simplices().simplex(simplex_id);
    auto energies = std::vector<double>{};
    energies.reserve(simplex.vertex_ids.size());
    for (const auto vertex_id : simplex.vertex_ids) {
        energies.push_back(cache.get(vertex_id).eigenvalues[band]);
    }
    std::stable_sort(energies.begin(), energies.end());
    return energies;
}

}  // namespace

ChargeContribution &ChargeContribution::operator+=(
    const ChargeContribution &other
) noexcept {
    value += other.value;
    dcharge_dmu += other.dcharge_dmu;
    estimated_error += other.estimated_error;
    visible_gapless_simplices += other.visible_gapless_simplices;
    inconclusive_simplices += other.inconclusive_simplices;
    return *this;
}

ChargeContribution &ChargeContribution::operator-=(
    const ChargeContribution &other
) noexcept {
    value -= other.value;
    dcharge_dmu -= other.dcharge_dmu;
    estimated_error -= other.estimated_error;
    visible_gapless_simplices -= other.visible_gapless_simplices;
    inconclusive_simplices -= other.inconclusive_simplices;
    return *this;
}

ChargeContribution band_charge_on_simplex(
    double mu,
    const SpectralMesh &mesh,
    const core::Geometry &geometry,
    core::SimplexId simplex_id
) {
    const auto &simplex = geometry.simplices().simplex(simplex_id);
    const auto &cache = mesh.eigensystems();
    auto result = ChargeContribution{};

    for (std::size_t band = 0; band < mesh.ndof(); ++band) {
        const auto moments = cut::simplex_moments(
            geometry,
            simplex_id,
            [&](core::VertexId vertex_id) {
                return cache.get(vertex_id).eigenvalues[band];
            },
            cut::LevelOptions{
                .level = mu,
                .level_tolerance = mesh.tolerance(),
            }
        );

        if (moments.kind == cut::SimplexCutKind::on_level) {
            result.value += 0.5 * simplex.volume;
            continue;
        }

        result.value += moments.volume;
        const auto energies =
            band_energies(geometry, simplex_id, cache, band);
        result.dcharge_dmu +=
            simplex.volume * occupied_fraction_derivative(
                energies,
                mu,
                mesh.tolerance()
            );
    }
    return result;
}

ChargeContribution charge_on_simplex(
    double mu,
    SpectralMesh &mesh,
    const core::Geometry &geometry,
    core::SimplexId simplex_id,
    ChargeErrorEstimator &error_estimator
) {
    const auto certificate = cert::certify_mesh_simplex(
        mesh,
        simplex_id,
        mu,
        0.0,
        mesh.tolerance()
    );

    auto result =
        band_charge_on_simplex(mu, mesh, geometry, simplex_id);
    if (certificate.status ==
        cert::SimplexCertificateStatus::VisibleGapless) {
        result.visible_gapless_simplices = 1;
    } else if (
        certificate.status ==
        cert::SimplexCertificateStatus::Inconclusive
    ) {
        result.inconclusive_simplices = 1;
    }
    result.estimated_error = error_estimator.estimate(
        geometry,
        simplex_id,
        result.value,
        certificate
    );
    return result;
}

}  // namespace fermisimplex::integration_detail
