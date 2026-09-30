#include "integration/charge.h"

#include "occupation/affine_cut.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fermisimplex::integration_detail {
namespace core = adaptivesimplex::core;

double validated_density_cut_error(double estimate, double charge) {
    const auto roundoff = 32 * std::numeric_limits<double>::epsilon() *
        std::max(1.0, std::abs(charge));
    if (!std::isfinite(estimate) || estimate < -roundoff) {
        throw std::runtime_error("invalid density cut error estimate");
    }
    return std::max(0.0, estimate);
}

ChargeContribution &ChargeContribution::operator+=(
    const ChargeContribution &other
) noexcept {
    value += other.value;
    dcharge_dmu += other.dcharge_dmu;
    estimated_error += other.estimated_error;
    density_cut_error += other.density_cut_error;
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
    density_cut_error -= other.density_cut_error;
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
        std::vector<double> energies;
        energies.reserve(simplex.vertex_ids.size());
        for (const auto vertex_id : simplex.vertex_ids)
            energies.push_back(cache.get(vertex_id).eigenvalues[band] - mu);
        const occupation_detail::AffineCut cut(std::move(energies), mesh.tolerance());
        const auto integral = cut.fraction_and_derivative();
        result.value += simplex.volume * integral.fraction;
        result.dcharge_dmu += simplex.volume * integral.derivative;
    }
    return result;
}

}  // namespace fermisimplex::integration_detail
