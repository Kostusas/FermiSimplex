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
    const auto *tb = dynamic_cast<const TightBindingModel *>(&mesh.model());
    if (!tb) return std::nullopt;
    const auto hoppings = core_detail::TightBindingModelAccess::hoppings(*tb);
    for (const auto &term : hoppings)
        for (const auto component : term.lattice_vector)
            if (component != 0) return std::nullopt;

    // Constant H has an exact occupied measure, including half-filled flat
    // bands. Its charge interval can collapse without asserting a strict gap.
    const auto &simplex = mesh.geometry().simplices().simplex(id);
    OccupationEnclosure result;
    result.remainder_is_sampled = false;
    for (const auto energy : mesh.eigensystems().get(simplex.vertex_ids.front()).eigenvalues) {
        const auto kind = classify_cut(std::array{energy - mu}, mesh.tolerance()).kind;
        if (kind == CutKind::full) {
            ++result.occupation_lower;
            ++result.occupation_upper;
            result.charge_lower += simplex.volume;
        } else if (kind == CutKind::on_level) {
            ++result.occupation_upper;
            ++result.active_dimension;
            result.charge_lower += .5 * simplex.volume;
        }
    }
    result.charge_upper = result.charge_lower;
    return result;
}

struct Interval {
    double lower = 0, upper = 0, cut_error = 0;
    std::size_t occupation_lower = 0, occupation_upper = 0;
};

struct AffineBand {
    Weights energies;
    double radius;
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

std::optional<Polynomial> rotated_center_frame(const Polynomial &polynomial, ChargeErrorStats &stats) {
    auto basis = polynomial.center();
    if (diagonal_center(basis, polynomial.size)) return std::nullopt;
    std::vector<double> eigenvalues;
    linalg::diagonalize_hermitian_in_place(basis, eigenvalues, polynomial.size,
        true, "occupation polynomial center");
    ++stats.reduced_eigensystems;
    return polynomial.rotated(basis);
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
        result.bands.push_back({std::move(affine), radius, negative, positive});
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
            below[i] = bound.energies[i] + bound.radius;
            above[i] = bound.energies[i] - bound.radius;
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

template <bool IntegrateCharge>
Interval traverse(const Polynomial &polynomial, const AffineBounds &bounds, double epsilon,
                   const std::vector<Weights> &points,
                   const std::vector<Weights> &root_cuts, double volume,
                   std::uint32_t remaining, ChargeErrorStats &stats) {
    ++stats.micro_simplices;
    if (remaining == 0 || bounds.occupation_lower == bounds.occupation_upper) {
        ++stats.terminal_simplices;
        stats.terminal_active_dimension_sum += polynomial.size;
        if constexpr (IntegrateCharge) return charge_interval(bounds, root_cuts, volume);
        return {.occupation_lower = bounds.occupation_lower,
                .occupation_upper = bounds.occupation_upper};
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
        AffineBounds child_bounds;
        {
            const auto rotated = rotated_center_frame(child_polynomial, stats);
            child_bounds = affine_bounds(rotated ? *rotated : child_polynomial, epsilon);
        }
        const auto child = traverse<IntegrateCharge>(child_polynomial, child_bounds, epsilon,
            child_points, child_cuts, volume / 2, remaining - 1, stats);
        result.lower += child.lower; result.upper += child.upper;
        result.cut_error += child.cut_error;
        result.occupation_lower = std::min(result.occupation_lower, child.occupation_lower);
        result.occupation_upper = std::max(result.occupation_upper, child.occupation_upper);
    }
    return result;
}

}  // namespace

bool visible_occupation_change(const SpectralMesh &mesh, adaptivesimplex::core::SimplexId id, double mu) {
    auto previous_count = mesh.ndof() + 1;
    for (const auto vertex : mesh.geometry().simplices().simplex(id).vertex_ids) {
        const auto &values = mesh.eigensystems().get(vertex).eigenvalues;
        const auto tolerance = mesh.tolerance() * std::max(
            {1., std::abs(mu), std::abs(values.front()), std::abs(values.back())});
        const auto first = std::lower_bound(values.begin(), values.end(), mu - tolerance);
        if (first != values.end() && *first <= mu + tolerance) return true;
        const auto count = static_cast<std::size_t>(first - values.begin());
        if (previous_count <= mesh.ndof() && count != previous_count) return true;
        previous_count = count;
    }
    return false;
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
    AffineBounds bounds;
    {
        const auto rotated = rotated_center_frame(model.polynomial, stats);
        const auto &framed = rotated ? *rotated : model.polynomial;
        sectors = sign_sectors(framed, model.epsilon, stats);
        bounds = affine_bounds(framed, model.epsilon);
    }
    const auto interval = traverse<IntegrateCharge>(model.polynomial, bounds, model.epsilon, points, cuts,
        simplex.volume, sectors.negative + sectors.positive == q ? 0 : depth * mesh.ndim(), stats);
    result.occupation_lower += std::max(sectors.negative, interval.occupation_lower);
    result.occupation_upper += std::min(q - sectors.positive, interval.occupation_upper);
    if constexpr (IntegrateCharge) {
        result.charge_lower += std::max(simplex.volume * sectors.negative, interval.lower);
        result.charge_upper += std::min(simplex.volume * (q - sectors.positive), interval.upper);
        result.density_cut_error += std::min(simplex.volume * q, interval.cut_error);
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
