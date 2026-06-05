#include "lineartetrahedron/simplex_rules.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace lineartetrahedron::simplex_rules {
namespace {

using BaryPoint = std::array<double, 4>;

void require_supported_dimension(size_t ndim) {
    if (ndim < 1 || ndim > 3) {
        throw std::runtime_error("LinearTetrahedron supports dimensions 1, 2, and 3");
    }
}

std::vector<BaryPoint> basis_points(size_t ndim) {
    std::vector<BaryPoint> points(ndim + 1, BaryPoint{0.0, 0.0, 0.0, 0.0});
    for (size_t vertex = 0; vertex <= ndim; ++vertex) {
        points[vertex][vertex] = 1.0;
    }
    return points;
}

BaryPoint edge_point(
    const std::vector<BaryPoint> &basis,
    const double *energies,
    double mu,
    size_t inside,
    size_t outside,
    double tol
) {
    const double denom = energies[outside] - energies[inside];
    const double alpha = denom <= tol
        ? 0.0
        : std::clamp((mu - energies[inside]) / denom, 0.0, 1.0);
    BaryPoint point{0.0, 0.0, 0.0, 0.0};
    for (size_t index = 0; index < basis.size(); ++index) {
        point[index] = (1.0 - alpha) * basis[inside][index] +
                       alpha * basis[outside][index];
    }
    return point;
}

double relative_volume(
    const std::vector<BaryPoint> &simplex,
    size_t ndim
) {
    if (ndim == 1) {
        return std::abs(simplex[1][1] - simplex[0][1]);
    }
    if (ndim == 2) {
        const double ax = simplex[1][1] - simplex[0][1];
        const double ay = simplex[1][2] - simplex[0][2];
        const double bx = simplex[2][1] - simplex[0][1];
        const double by = simplex[2][2] - simplex[0][2];
        return std::abs(ax * by - ay * bx);
    }

    const double ax = simplex[1][1] - simplex[0][1];
    const double ay = simplex[1][2] - simplex[0][2];
    const double az = simplex[1][3] - simplex[0][3];
    const double bx = simplex[2][1] - simplex[0][1];
    const double by = simplex[2][2] - simplex[0][2];
    const double bz = simplex[2][3] - simplex[0][3];
    const double cx = simplex[3][1] - simplex[0][1];
    const double cy = simplex[3][2] - simplex[0][2];
    const double cz = simplex[3][3] - simplex[0][3];
    return std::abs(
        ax * (by * cz - bz * cy) -
        ay * (bx * cz - bz * cx) +
        az * (bx * cy - by * cx)
    );
}

void add_simplex_moments(
    std::vector<double> &fractions,
    const std::vector<BaryPoint> &simplex,
    size_t ndim
) {
    const double scale = relative_volume(simplex, ndim) / static_cast<double>(ndim + 1);
    for (size_t vertex = 0; vertex <= ndim; ++vertex) {
        double bary_sum = 0.0;
        for (const auto &point : simplex) {
            bary_sum += point[vertex];
        }
        fractions[vertex] += scale * bary_sum;
    }
}

}  // namespace

std::pair<double, double> simplex_fraction_and_derivative(
    const double *energies,
    double mu,
    size_t dimension,
    double tol
) {
    require_supported_dimension(dimension);
    if (dimension == 1) {
        const double e0 = energies[0];
        const double e1 = energies[1];
        const double denom = e1 - e0;
        const double fraction = std::clamp((mu - e0) / denom, 0.0, 1.0);
        const double derivative = (mu > e0 + tol && mu < e1 - tol) ? 1.0 / denom : 0.0;
        return {fraction, derivative};
    }
    if (dimension == 2) {
        const double e0 = energies[0];
        const double e1 = energies[1];
        const double e2 = energies[2];
        if (mu < e1) {
            const double denom = (e1 - e0) * (e2 - e0);
            const double x = mu - e0;
            return {std::clamp((x * x) / denom, 0.0, 1.0), 2.0 * x / denom};
        }
        const double denom = (e2 - e0) * (e2 - e1);
        const double x = e2 - mu;
        return {std::clamp(1.0 - (x * x) / denom, 0.0, 1.0), 2.0 * x / denom};
    }

    const double e0 = energies[0];
    const double e1 = energies[1];
    const double e2 = energies[2];
    const double e3 = energies[3];
    if (mu < e1) {
        const double denom = (e1 - e0) * (e2 - e0) * (e3 - e0);
        const double x = mu - e0;
        return {std::clamp((x * x * x) / denom, 0.0, 1.0), 3.0 * x * x / denom};
    }
    if (mu < e2) {
        const double denom0 = (e1 - e0) * (e2 - e0) * (e3 - e0);
        const double denom1 = (e0 - e1) * (e2 - e1) * (e3 - e1);
        const double x0 = mu - e0;
        const double x1 = mu - e1;
        return {
            std::clamp(
                (x0 * x0 * x0) / denom0 + (x1 * x1 * x1) / denom1,
                0.0,
                1.0
            ),
            3.0 * (x0 * x0 / denom0 + x1 * x1 / denom1),
        };
    }
    const double denom = (e3 - e0) * (e3 - e1) * (e3 - e2);
    const double x = e3 - mu;
    return {
        std::clamp(1.0 - (x * x * x) / denom, 0.0, 1.0),
        3.0 * x * x / denom,
    };
}

std::vector<double> occupied_linear_moment_fractions(
    const double *sorted_energies,
    size_t ndim,
    double mu,
    double tol
) {
    require_supported_dimension(ndim);

    const size_t n_vertices = ndim + 1;
    size_t n_inside = 0;
    while (n_inside < n_vertices && sorted_energies[n_inside] <= mu + tol) {
        ++n_inside;
    }

    std::vector<double> fractions(n_vertices, 0.0);
    if (n_inside == 0) {
        return fractions;
    }
    if (n_inside == n_vertices) {
        std::fill(fractions.begin(), fractions.end(), 1.0 / static_cast<double>(n_vertices));
        return fractions;
    }

    const auto basis = basis_points(ndim);
    if (ndim == 1) {
        const auto p01 = edge_point(basis, sorted_energies, mu, 0, 1, tol);
        add_simplex_moments(fractions, {basis[0], p01}, ndim);
        return fractions;
    }

    if (ndim == 2) {
        if (n_inside == 1) {
            const auto p01 = edge_point(basis, sorted_energies, mu, 0, 1, tol);
            const auto p02 = edge_point(basis, sorted_energies, mu, 0, 2, tol);
            add_simplex_moments(fractions, {basis[0], p01, p02}, ndim);
            return fractions;
        }
        const auto p02 = edge_point(basis, sorted_energies, mu, 0, 2, tol);
        const auto p12 = edge_point(basis, sorted_energies, mu, 1, 2, tol);
        add_simplex_moments(fractions, {basis[0], basis[1], p12}, ndim);
        add_simplex_moments(fractions, {basis[0], p02, p12}, ndim);
        return fractions;
    }

    if (n_inside == 1) {
        const auto p01 = edge_point(basis, sorted_energies, mu, 0, 1, tol);
        const auto p02 = edge_point(basis, sorted_energies, mu, 0, 2, tol);
        const auto p03 = edge_point(basis, sorted_energies, mu, 0, 3, tol);
        add_simplex_moments(fractions, {basis[0], p01, p02, p03}, ndim);
        return fractions;
    }
    if (n_inside == 2) {
        const auto p02 = edge_point(basis, sorted_energies, mu, 0, 2, tol);
        const auto p03 = edge_point(basis, sorted_energies, mu, 0, 3, tol);
        const auto p12 = edge_point(basis, sorted_energies, mu, 1, 2, tol);
        const auto p13 = edge_point(basis, sorted_energies, mu, 1, 3, tol);
        add_simplex_moments(fractions, {basis[0], basis[1], p12, p13}, ndim);
        add_simplex_moments(fractions, {basis[0], p02, p12, p13}, ndim);
        add_simplex_moments(fractions, {basis[0], p02, p03, p13}, ndim);
        return fractions;
    }

    const auto p03 = edge_point(basis, sorted_energies, mu, 0, 3, tol);
    const auto p13 = edge_point(basis, sorted_energies, mu, 1, 3, tol);
    const auto p23 = edge_point(basis, sorted_energies, mu, 2, 3, tol);
    add_simplex_moments(fractions, {basis[0], basis[1], basis[2], p23}, ndim);
    add_simplex_moments(fractions, {basis[0], basis[1], p13, p23}, ndim);
    add_simplex_moments(fractions, {basis[0], p03, p13, p23}, ndim);
    return fractions;
}

}  // namespace lineartetrahedron::simplex_rules
