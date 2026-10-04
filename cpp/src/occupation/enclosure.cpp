#include <fermisimplex/occupation.h>

#include "occupation/model.h"
#include "occupation/enclosure.h"
#include "occupation/cut_disagreement.h"
#include "core/tight_binding_access.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace fermisimplex {
namespace {
using namespace occupation_detail;
namespace cut_error = occupation_detail;
constexpr double cut_tolerance = 64 * std::numeric_limits<double>::epsilon();

std::optional<OccupationEnclosure> constant_enclosure(
    const SpectralMesh &mesh, adaptivesimplex::core::SimplexId id, double mu
) {
    const auto roundoff = core_detail::TightBindingModelAccess::constant_spectrum_roundoff(
        mesh.model());
    if (!roundoff) return std::nullopt;

    // Constancy removes interpolation error, not eigensolver uncertainty.
    // Keep near-level eigenvalues uncertain unless the spectrum is exact.
    const auto &simplex = mesh.geometry().simplices().simplex(id);
    OccupationEnclosure result;
    result.remainder_is_sampled = false;
    result.model_error = *roundoff;
    for (const auto energy : mesh.eigensystems().get(simplex.vertex_ids.front()).eigenvalues) {
        const std::array relative{energy - mu};
        const auto kind = classify_cut(relative, mesh.tolerance()).kind;
        const auto negative = relative[0] < -*roundoff;
        const auto positive = relative[0] > *roundoff;
        const auto exact_level = *roundoff == 0 && relative[0] == 0;
        const auto lower = negative ? 1. : exact_level ? .5 : 0.;
        const auto upper = positive ? 0. : exact_level ? .5 : 1.;
        const auto reported = occupied_volume(1., relative, mesh.tolerance(), kind);
        result.charge_lower += simplex.volume * lower;
        result.charge_upper += simplex.volume * upper;
        result.density_cut_error += simplex.volume * std::max(
            std::abs(reported - lower), std::abs(upper - reported));
        result.occupation_lower += negative;
        result.occupation_upper += !positive;
        result.active_dimension += !negative && !positive;
    }
    return result;
}

struct Interval {
    double lower = 0, upper = 0, cut_error = 0;
    std::size_t occupation_lower = 0, occupation_upper = 0;
};

struct AffineBand {
    Weights energies;
    double lower_radius, upper_radius;
    bool negative;
    bool positive;
};

struct AffineBounds {
    std::vector<AffineBand> bands;
    std::size_t occupation_lower = 0, occupation_upper = 0;
};

double volume_below(double volume, const Weights &values, bool upper) {
    const auto kind = classify_cut(values, cut_tolerance).kind;
    if (kind == CutKind::on_level) return upper ? volume : 0.;
    return occupied_volume(volume, values, cut_tolerance, kind);
}

AffineBounds affine_bounds(const Polynomial &polynomial, double epsilon) {
    const auto q = polynomial.size, v = polynomial.vertices;
    AffineBounds result;
    result.bands.reserve(q);
    for (std::size_t band = 0; band < q; ++band) {
        double curvature_radius = 0, off_diagonal_radius = 0;
        double minimum = std::numeric_limits<double>::infinity();
        double maximum = -minimum;
        for (std::size_t i = 0; i < v; ++i)
            for (std::size_t j = i; j < v; ++j) {
                const auto &control = polynomial.at(i, j);
                const auto diagonal = control[band + band * q].real();
                auto curvature = std::abs(diagonal -
                    .5 * (polynomial.at(i, i)[band + band * q].real() +
                          polynomial.at(j, j)[band + band * q].real()));
                double row_radius = 0;
                for (std::size_t other = 0; other < q; ++other)
                    if (other != band) row_radius += std::abs(control[band + other * q]);
                // The edge Bernstein weights sum to at most (v-1)/v.
                // Bound off-diagonal entries separately before applying this
                // factor to the curvature-only polynomial (zero at vertices).
                curvature_radius = std::max(curvature_radius, curvature);
                off_diagonal_radius = std::max(off_diagonal_radius, row_radius);
                minimum = std::min(minimum, diagonal - row_radius);
                maximum = std::max(maximum, diagonal + row_radius);
            }
        auto radius = epsilon + off_diagonal_radius +
            (static_cast<double>(v - 1) / v) * curvature_radius;
        double scale = std::max({1., std::abs(minimum), std::abs(maximum), radius});
        // Outward slack absorbs the cut routine's level classification and
        // prevents nearly endpoint crossings from rounding their fraction to 1.
        radius += 8 * cut_tolerance * scale;
        Weights affine(v);
        bool below_negative = true, above_nonpositive = false;
        for (std::size_t i = 0; i < v; ++i) {
            affine[i] = polynomial.at(i, i)[band + band * q].real();
            below_negative &= affine[i] + radius < 0;
            above_nonpositive |= affine[i] - radius <= 0;
        }
        const auto negative = maximum + epsilon < 0;
        const auto positive = minimum - epsilon > 0;
        if (negative || below_negative)
            ++result.occupation_lower;
        if (!positive && above_nonpositive)
            ++result.occupation_upper;
        result.bands.push_back({std::move(affine), radius, radius, negative, positive});
    }
    return result;
}

Interval charge_interval(const AffineBounds &bounds,
                         const std::vector<Weights> &root_cuts, double volume) {
    Interval result;
    result.occupation_lower = bounds.occupation_lower;
    result.occupation_upper = bounds.occupation_upper;
    Weights below(root_cuts.size()), above(root_cuts.size()), root(root_cuts.size());
    for (std::size_t band = 0; band < bounds.bands.size(); ++band) {
        const auto &bound = bounds.bands[band];
        for (std::size_t i = 0; i < root_cuts.size(); ++i) {
            below[i] = bound.energies[i] + bound.upper_radius;
            above[i] = bound.energies[i] - bound.lower_radius;
            root[i] = root_cuts[i][band];
        }
        const auto lower = bound.negative ? volume : volume_below(volume, below, false);
        const auto upper = bound.positive ? 0. : volume_below(volume, above, true);
        result.lower += lower;
        result.upper += upper;
        result.cut_error += cut_error::cut_disagreement(
            volume, root, bound.energies, 0.) + std::max(0., upper - lower);
    }
    return result;
}

AffineBounds residual_bounds(const Polynomial &polynomial,
    const ResidualMatrices &residual, const Matrix *basis) {
    const auto q = polynomial.size, v = polynomial.vertices;
    const auto infinity = std::numeric_limits<double>::infinity();
    std::vector<double> lower(q, -infinity), upper(q, -infinity);
    std::vector<double> minimum(q, infinity), maximum(q, -infinity);
    Matrix scratch, rotated_low, rotated_high;
    if (basis) {
        rotated_low = rotate_controls(residual.lower, *basis, q, scratch);
        rotated_high = rotate_controls(residual.upper, *basis, q, scratch);
    }
    const auto &low_controls = basis ? rotated_low : residual.lower;
    const auto &high_controls = basis ? rotated_high : residual.upper;
    for (std::size_t c = 0; c < residual.layout->indices.size(); ++c) {
        const auto *low = low_controls.data()+c*q*q;
        const auto *high = high_controls.data()+c*q*q;
        for (std::size_t band = 0; band < q; ++band) {
            double low_row = 0, high_row = 0, affine = 0;
            for (std::size_t other = 0; other < q; ++other)
                if (other != band) {
                    low_row += std::abs(low[band+other*q]);
                    high_row += std::abs(high[band+other*q]);
                }
            for (std::size_t i = 0; i < v; ++i)
                affine += double(residual.layout->indices[c][i])/6 *
                          polynomial.at(i,i)[band+band*q].real();
            const auto lo = low[band+band*q].real() - low_row;
            const auto hi = high[band+band*q].real() + high_row;
            lower[band] = std::max(lower[band], affine-lo);
            upper[band] = std::max(upper[band], hi-affine);
            minimum[band] = std::min(minimum[band], lo);
            maximum[band] = std::max(maximum[band], hi);
        }
    }
    AffineBounds result;
    for (std::size_t band = 0; band < q; ++band) {
        const auto scale = std::max({1., std::abs(minimum[band]), std::abs(maximum[band]),
                                    residual.remainder, lower[band], upper[band]});
        const auto slack = residual.remainder + 8*cut_tolerance*scale;
        const auto lo = lower[band]+slack, hi = upper[band]+slack;
        Weights affine(v);
        bool below = true, above = false;
        for (std::size_t i = 0; i < v; ++i) {
            affine[i] = polynomial.at(i,i)[band+band*q].real();
            below &= affine[i]+hi < 0;
            above |= affine[i]-lo <= 0;
        }
        const auto negative = maximum[band]+slack < 0;
        const auto positive = minimum[band]-slack > 0;
        result.occupation_lower += negative || below;
        result.occupation_upper += !positive && above;
        result.bands.push_back({std::move(affine), lo, hi, negative, positive});
    }
    return result;
}

struct Bounds {
    AffineBounds scalar;
    std::optional<AffineBounds> residual;
    std::size_t lower() const {
        return residual ? std::max(scalar.occupation_lower, residual->occupation_lower)
                        : scalar.occupation_lower;
    }
    std::size_t upper() const {
        return residual ? std::min(scalar.occupation_upper, residual->occupation_upper)
                        : scalar.occupation_upper;
    }
};

Bounds framed_bounds(const Polynomial &polynomial, double epsilon,
    const ResidualMatrices *residual, ChargeErrorStats &stats, Sectors *sectors = nullptr) {
    auto basis = polynomial.center();
    std::optional<Polynomial> rotated;
    if (!diagonal_center(basis, polynomial.size)) {
        std::vector<double> eigenvalues;
        linalg::diagonalize_hermitian_in_place(basis, eigenvalues, polynomial.size,
            true, "occupation polynomial center");
        ++stats.reduced_eigensystems;
        rotated = polynomial.rotated(basis);
    }
    const auto &framed = rotated ? *rotated : polynomial;
    Bounds result{affine_bounds(framed, epsilon), std::nullopt};
    if (sectors) *sectors = sign_sectors(framed, epsilon, stats);
    if (residual && result.lower() != result.upper() &&
        (!sectors || sectors->negative+sectors->positive != polynomial.size))
        result.residual = residual_bounds(framed, *residual, rotated ? &basis : nullptr);
    return result;
}

Interval charge_interval(const Bounds &bounds,
                         const std::vector<Weights> &root_cuts, double volume) {
    auto result = charge_interval(bounds.scalar, root_cuts, volume);
    if (bounds.residual) {
        const auto other = charge_interval(*bounds.residual, root_cuts, volume);
        result.lower = std::max(result.lower, other.lower);
        result.upper = std::min(result.upper, other.upper);
        result.cut_error = std::min(result.cut_error, other.cut_error);
        result.occupation_lower = bounds.lower();
        result.occupation_upper = bounds.upper();
    }
    return result;
}

template <bool IntegrateCharge>
Interval traverse(const Polynomial &polynomial, const Bounds &bounds, double epsilon,
                   const ResidualMatrices *residual,
                   const std::vector<Weights> &points,
                   const std::vector<Weights> &root_cuts, double volume,
                   std::uint32_t remaining, ChargeErrorStats &stats) {
    ++stats.micro_simplices;
    if (remaining == 0 || bounds.lower() == bounds.upper()) {
        ++stats.terminal_simplices;
        stats.terminal_active_dimension_sum += polynomial.size;
        if constexpr (IntegrateCharge) return charge_interval(bounds, root_cuts, volume);
        return {.occupation_lower = bounds.lower(),
                .occupation_upper = bounds.upper()};
    }
    // Longest physical edge bisection; only the polynomial is evaluated below.
    double longest = -1;
    std::size_t left = 0, right = 1;
    for (std::size_t i = 0; i < points.size(); ++i)
        for (std::size_t j = i + 1; j < points.size(); ++j) {
            double squared = 0;
            for (std::size_t axis = 0; axis < points[i].size(); ++axis)
                squared += std::pow(points[i][axis] - points[j][axis], 2);
            if (squared > longest) { longest = squared; left = i; right = j; }
        }
    Interval result;
    result.occupation_lower = polynomial.size;
    for (const auto replaced : {left, right}) {
        auto child_points = points, child_cuts = root_cuts;
        for (std::size_t axis = 0; axis < points[0].size(); ++axis)
            child_points[replaced][axis] = .5 * (points[left][axis] + points[right][axis]);
        if constexpr (IntegrateCharge)
            for (std::size_t band = 0; band < polynomial.size; ++band)
                child_cuts[replaced][band] = .5 * (root_cuts[left][band] + root_cuts[right][band]);
        const auto child_polynomial = polynomial.bisected(left, right, replaced);
        std::optional<ResidualMatrices> child_residual;
        if (residual) child_residual = residual->bisected(left, right, replaced);
        const auto child_pointer = child_residual ? &*child_residual : nullptr;
        const auto child_bounds = framed_bounds(child_polynomial, epsilon, child_pointer, stats);
        const auto child = traverse<IntegrateCharge>(child_polynomial, child_bounds, epsilon,
            child_pointer, child_points, child_cuts, volume / 2, remaining - 1, stats);
        result.lower += child.lower; result.upper += child.upper;
        result.cut_error += child.cut_error;
        result.occupation_lower = std::min(result.occupation_lower, child.occupation_lower);
        result.occupation_upper = std::max(result.occupation_upper, child.occupation_upper);
    }
    return result;
}

}  // namespace

occupation_detail::VertexOccupation occupation_detail::vertex_occupation(
    const SpectralMesh &mesh, adaptivesimplex::core::SimplexId id, double mu
) {
    std::size_t maximum_lower = 0, minimum_upper = mesh.ndof();
    bool touches_level = false;
    for (const auto vertex : mesh.geometry().simplices().simplex(id).vertex_ids) {
        const auto &values = mesh.eigensystems().get(vertex).eigenvalues;
        const auto tolerance = mesh.tolerance() * std::max(
            {1., std::abs(mu), std::abs(values.front()), std::abs(values.back())});
        const auto lower = std::lower_bound(values.begin(), values.end(), mu - tolerance);
        const auto upper = std::upper_bound(lower, values.end(), mu + tolerance);
        touches_level |= lower != upper;
        maximum_lower = std::max(maximum_lower, static_cast<std::size_t>(lower - values.begin()));
        minimum_upper = std::min(minimum_upper, static_cast<std::size_t>(upper - values.begin()));
        if (maximum_lower > minimum_upper) return VertexOccupation::crossing;
    }
    return touches_level ? VertexOccupation::touches_level : VertexOccupation::uniform;
}

namespace {
// Both consumers use the same bounds and subdivision. Compile out integrated
// quantities for the sign query; no algorithm choice reaches the public API.
template <bool IntegrateCharge>
OccupationEnclosure enclosure(const SpectralMesh &mesh,
    adaptivesimplex::core::SimplexId simplex_id, double mu, std::uint32_t depth,
    ChargeErrorStats &stats, std::optional<double> interpolation_error_bound) {
    if (!std::isfinite(mu)) throw std::invalid_argument("mu must be finite");
    if (depth > 20 / mesh.ndim())
        throw std::invalid_argument("quadratic occupation depth * ndim must be <= 20");
    if (interpolation_error_bound &&
        (!std::isfinite(*interpolation_error_bound) || *interpolation_error_bound < 0))
        throw std::invalid_argument("interpolation error bound must be finite and nonnegative");
    ++stats.root_simplices;
    if (const auto constant = constant_enclosure(mesh, simplex_id, mu)) return *constant;
    const auto model = build_model(mesh, simplex_id, mu, stats, interpolation_error_bound);
    const auto &simplex = mesh.geometry().simplices().simplex(simplex_id);
    const auto q = model.polynomial.size;
    OccupationEnclosure result;
    result.interpolation_error = model.eta;
    result.model_error = model.epsilon;
    result.safe_gap = model.delta;
    result.active_dimension = q;
    result.remainder_is_sampled = !interpolation_error_bound;
    result.charge_lower = result.charge_upper = simplex.volume * model.safe_occupation;
    result.occupation_lower = result.occupation_upper = model.safe_occupation;
    std::vector<Weights> cuts;
    if constexpr (IntegrateCharge) {
        cuts.assign(simplex.vertex_ids.size(), Weights(q));
        Weights energies(simplex.vertex_ids.size());
        for (std::size_t band = 0; band < mesh.ndof(); ++band) {
            for (std::size_t i = 0; i < energies.size(); ++i)
                energies[i] = mesh.eigensystems().get(simplex.vertex_ids[i]).eigenvalues[band] - mu;
            const auto kind = snap_cut_to_level(energies, mesh.tolerance());
            if (band >= model.safe_occupation && band < model.safe_occupation + q) {
                for (std::size_t i = 0; i < energies.size(); ++i)
                    cuts[i][band - model.safe_occupation] = energies[i];
            } else {
                // A strict safe sign can still be half occupied by the reported
                // cut when its energies fall within the user's level tolerance.
                const auto occupied = occupied_volume(simplex.volume, energies, 0., kind);
                result.density_cut_error += band < model.safe_occupation
                    ? simplex.volume - occupied : occupied;
            }
        }
    }
    if (q == 0) return result;
    std::vector<Weights> points;
    for (const auto vertex : simplex.vertex_ids)
        points.push_back(mesh.geometry().vertices().dyadic_vertex(vertex).to_point());
    // A fixed polynomial can have a better block sign proof than row bounds.
    Sectors sectors;
    const auto residual = model.residual ? &*model.residual : nullptr;
    const auto bounds = framed_bounds(model.polynomial, model.epsilon, residual, stats, &sectors);
    const auto interval = traverse<IntegrateCharge>(model.polynomial, bounds, model.epsilon, residual, points, cuts,
        simplex.volume, sectors.negative + sectors.positive == q ? 0 : depth * mesh.ndim(), stats);
    result.occupation_lower += std::max(sectors.negative, interval.occupation_lower);
    result.occupation_upper += std::min(q - sectors.positive, interval.occupation_upper);
    if constexpr (IntegrateCharge) {
        result.charge_lower += std::max(simplex.volume * sectors.negative, interval.lower);
        result.charge_upper += std::min(simplex.volume * (q - sectors.positive), interval.upper);
        if (result.fixed_occupation()) {
            // A fixed rank determines every ordered occupation. Compare the
            // reported cuts directly with those constants, including snapped
            // half occupations, instead of retaining loose affine row bounds.
            const auto occupied_count = result.occupation_lower - model.safe_occupation;
            Weights energies(cuts.size());
            for (std::size_t band = 0; band < q; ++band) {
                for (std::size_t i = 0; i < cuts.size(); ++i)
                    energies[i] = cuts[i][band];
                const auto kind = classify_cut(energies, 0.).kind;
                const auto reported = occupied_volume(simplex.volume, energies, 0., kind);
                result.density_cut_error += band < occupied_count
                    ? simplex.volume - reported : reported;
            }
        } else {
            result.density_cut_error += std::min(simplex.volume * q, interval.cut_error);
        }
    }
    return result;
}

}  // namespace

OccupationEnclosure enclose_occupation(const SpectralMesh &mesh,
    adaptivesimplex::core::SimplexId simplex_id, double mu, std::uint32_t depth,
    ChargeErrorStats &stats, std::optional<double> interpolation_error_bound) {
    return enclosure<true>(mesh, simplex_id, mu, depth, stats, interpolation_error_bound);
}

bool occupation_detail::fixed_occupation(const SpectralMesh &mesh,
    adaptivesimplex::core::SimplexId simplex_id, double mu, std::uint32_t depth,
    ChargeErrorStats &stats, std::optional<double> interpolation_error_bound) {
    return enclosure<false>(mesh, simplex_id, mu, depth, stats,
                            interpolation_error_bound).fixed_occupation();
}

}  // namespace fermisimplex
