#include <fermisimplex/occupation.h>

#include "occupation/model.h"
#include "integration/charge_error/cut_disagreement.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace fermisimplex {
namespace {
using namespace occupation_detail;
namespace cut = adaptivesimplex::cut;
namespace cut_error = integration_detail::charge_error_detail;
constexpr double cut_tolerance = 64 * std::numeric_limits<double>::epsilon();

struct Interval {
    double lower = 0, upper = 0, cut_error = 0;
    std::size_t occupation_lower = 0, occupation_upper = 0;
};

double volume_below(double volume, const Weights &values, bool upper) {
    const auto moments = cut::simplex_moments(
        volume, values, {.level = 0., .level_tolerance = cut_tolerance});
    if (moments.kind == cut::SimplexCutKind::on_level) return upper ? volume : 0.;
    return moments.volume;
}

Polynomial center_frame(const Polynomial &polynomial, ChargeErrorStats &stats) {
    auto basis = polynomial.center();
    if (diagonal_center(basis, polynomial.size)) return polynomial;
    std::vector<double> eigenvalues;
    linalg::diagonalize_hermitian_in_place(basis, eigenvalues, polynomial.size,
        true, "occupation polynomial center");
    ++stats.reduced_eigensystems;
    return polynomial.rotated(basis);
}

Interval affine_interval(const Polynomial &polynomial, double epsilon,
                         const std::vector<Weights> &root_cuts, double volume,
                         ChargeErrorStats &stats) {
    const auto q = polynomial.size, v = polynomial.vertices;
    const auto rotated = center_frame(polynomial, stats);
    Interval result;
    for (std::size_t band = 0; band < q; ++band) {
        double curvature_radius = 0, off_diagonal_radius = 0;
        double minimum = std::numeric_limits<double>::infinity();
        double maximum = -minimum;
        for (std::size_t i = 0; i < v; ++i)
            for (std::size_t j = i; j < v; ++j) {
                const auto &control = rotated.at(i, j);
                const auto diagonal = control[band + band * q].real();
                auto curvature = std::abs(diagonal -
                    .5 * (rotated.at(i, i)[band + band * q].real() +
                          rotated.at(j, j)[band + band * q].real()));
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
        Weights affine(v), below(v), above(v), root(v);
        for (std::size_t i = 0; i < v; ++i) {
            affine[i] = rotated.at(i, i)[band + band * q].real();
            below[i] = affine[i] + radius;
            above[i] = affine[i] - radius;
            root[i] = root_cuts[i][band];
        }
        const auto negative = maximum + epsilon < 0;
        const auto positive = minimum - epsilon > 0;
        const auto lower = negative ? volume : volume_below(volume, below, false);
        const auto upper = positive ? 0. : volume_below(volume, above, true);
        result.lower += lower;
        result.upper += upper;
        result.cut_error += cut_error::cut_disagreement(volume, root, affine, cut_tolerance) +
            std::max(0., upper - lower);
        if (negative || *std::max_element(below.begin(), below.end()) < 0)
            ++result.occupation_lower;
        if (!positive && *std::min_element(above.begin(), above.end()) <= 0)
            ++result.occupation_upper;
    }
    return result;
}

Interval integrate(const Polynomial &polynomial, double epsilon,
                   const std::vector<Weights> &points,
                   const std::vector<Weights> &root_cuts, double volume,
                   std::uint32_t remaining, ChargeErrorStats &stats) {
    ++stats.micro_simplices;
    auto interval = affine_interval(polynomial, epsilon, root_cuts, volume, stats);
    if (remaining == 0 || interval.occupation_lower == interval.occupation_upper) {
        ++stats.terminal_simplices;
        stats.terminal_active_dimension_sum += polynomial.size;
        return interval;
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
        auto weights = unit_weights(points.size());
        weights[replaced][replaced] = 0;
        weights[replaced][left] = weights[replaced][right] = .5;
        auto child_points = points, child_cuts = root_cuts;
        for (std::size_t axis = 0; axis < points[0].size(); ++axis)
            child_points[replaced][axis] = .5 * (points[left][axis] + points[right][axis]);
        for (std::size_t band = 0; band < polynomial.size; ++band)
            child_cuts[replaced][band] = .5 * (root_cuts[left][band] + root_cuts[right][band]);
        const auto child = integrate(polynomial.restrict_to(weights), epsilon,
            child_points, child_cuts, volume / 2, remaining - 1, stats);
        result.lower += child.lower; result.upper += child.upper;
        result.cut_error += child.cut_error;
        result.occupation_lower = std::min(result.occupation_lower, child.occupation_lower);
        result.occupation_upper = std::max(result.occupation_upper, child.occupation_upper);
    }
    return result;
}

}  // namespace

OccupationEnclosure enclose_occupation(const SpectralMesh &mesh,
    adaptivesimplex::core::SimplexId simplex_id, double mu, std::uint32_t depth,
    ChargeErrorStats &stats, std::optional<double> interpolation_error_bound) {
    if (!std::isfinite(mu)) throw std::invalid_argument("mu must be finite");
    if (depth > 20 / mesh.ndim())
        throw std::invalid_argument("quadratic occupation depth * ndim must be <= 20");
    if (interpolation_error_bound &&
        (!std::isfinite(*interpolation_error_bound) || *interpolation_error_bound < 0))
        throw std::invalid_argument("interpolation error bound must be finite and nonnegative");
    ++stats.root_simplices;
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
    if (q == 0) return result;
    std::vector<Weights> points, cuts;
    for (const auto vertex : simplex.vertex_ids) {
        points.push_back(mesh.geometry().vertices().dyadic_vertex(vertex).to_point());
        const auto &values = mesh.eigensystems().get(vertex).eigenvalues;
        Weights energies(q);
        for (std::size_t band = 0; band < q; ++band)
            energies[band] = values[band + model.safe_occupation] - mu;
        cuts.push_back(std::move(energies));
    }
    // A fixed polynomial can have a better block sign proof than row bounds.
    const auto sectors = sign_sectors(center_frame(model.polynomial, stats),
        model.epsilon, stats);
    const auto interval = integrate(model.polynomial, model.epsilon, points, cuts,
        simplex.volume, sectors.negative + sectors.positive == q ? 0 : depth * mesh.ndim(), stats);
    result.occupation_lower += std::max(sectors.negative, interval.occupation_lower);
    result.occupation_upper += std::min(q - sectors.positive, interval.occupation_upper);
    result.charge_lower += std::max(simplex.volume * sectors.negative, interval.lower);
    result.charge_upper += std::min(simplex.volume * (q - sectors.positive), interval.upper);
    result.density_cut_error = std::min(simplex.volume * q, interval.cut_error);
    return result;
}

}  // namespace fermisimplex
