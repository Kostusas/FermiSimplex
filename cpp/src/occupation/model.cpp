#include "occupation/model.h"

#include <limits>
#include <numeric>

namespace fermisimplex::occupation_detail {
namespace {

std::vector<std::size_t> indices(std::size_t first, std::size_t last) {
    std::vector<std::size_t> result(last - first);
    std::iota(result.begin(), result.end(), first);
    return result;
}

Matrix from_spectrum(const Eigensystem &spectrum, double mu) {
    const auto n = spectrum.eigenvalues.size();
    if (n == 1) return {spectrum.eigenvalues[0] - mu};
    auto weighted = spectrum.eigenvectors;
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = 0; i < n; ++i)
            weighted[i + j * n] *= spectrum.eigenvalues[j] - mu;
    Matrix result(n * n);
    linalg::matrix_multiply('N', 'C', n, n, n, 1., weighted.data(), n,
        spectrum.eigenvectors.data(), n, 0., result.data(), n);
    return result;
}

struct Interpolant {
    Polynomial polynomial;
    double remainder;
};

Interpolant interpolate(const SpectralMesh &mesh,
                        adaptivesimplex::core::SimplexId simplex_id, double mu,
                        ChargeErrorStats &stats, std::optional<double> bound) {
    const auto &simplex = mesh.geometry().simplices().simplex(simplex_id);
    const auto v = simplex.vertex_ids.size(), n = mesh.ndof();
    Polynomial polynomial{v, n, std::vector<Matrix>(v * v)};
    std::vector<Weights> points;
    for (std::size_t i = 0; i < v; ++i) {
        const auto vertex = simplex.vertex_ids[i];
        points.push_back(mesh.geometry().vertices().dyadic_vertex(vertex).to_point());
        polynomial.at(i, i) = from_spectrum(mesh.eigensystems().get(vertex), mu);
    }
    const auto evaluate = [&](const Weights &weights) {
        Weights point(mesh.ndim());
        for (std::size_t i = 0; i < v; ++i)
            for (std::size_t axis = 0; axis < point.size(); ++axis)
                point[axis] += weights[i] * points[i][axis];
        auto matrix = mesh.hamiltonian(point);
        ++stats.hamiltonian_evaluations;
        for (std::size_t i = 0; i < n; ++i) matrix[i + i * n] -= mu;
        return matrix;
    };
    for (std::size_t i = 0; i < v; ++i)
        for (std::size_t j = i + 1; j < v; ++j) {
            Weights weights(v);
            weights[i] = weights[j] = .5;
            auto midpoint = evaluate(weights);
            for (std::size_t k = 0; k < n * n; ++k)
                midpoint[k] = 2. * midpoint[k] -
                    .5 * (polynomial.at(i, i)[k] + polynomial.at(j, j)[k]);
            polynomial.at(j, i) = polynomial.at(i, j) = std::move(midpoint);
        }
    double defect = 0;
    const auto probe = [&](const Weights &weights) {
        defect = std::max(defect, hermitian_norm_bound(difference(
            evaluate(weights), polynomial.blossom(weights, weights)), n));
    };
    if (!bound) {
        for (std::size_t i = 0; i < v; ++i)
            for (std::size_t j = i + 1; j < v; ++j) {
                Weights weights(v);
                weights[i] = .25; weights[j] = .75;
                probe(weights);
                std::swap(weights[i], weights[j]);
                probe(weights);
                for (std::size_t k = j + 1; k < v; ++k) {
                    weights.assign(v, 0);
                    weights[i] = weights[j] = weights[k] = 1. / 3;
                    probe(weights);
                }
            }
        // In 1D the center is an interpolation node; in 2D it was a face probe.
        if (v > 3) probe(Weights(v, 1. / v));
    }
    double scale = std::max(1., std::abs(mu));
    for (const auto &control : polynomial.controls) scale = std::max(scale, norm(control));
    const auto roundoff = 64 * std::numeric_limits<double>::epsilon() * scale;
    return {std::move(polynomial), bound.value_or(2 * defect) + roundoff};
}

}  // namespace

Sectors sign_sectors(const Polynomial &polynomial, double allowance,
                     ChargeErrorStats &stats) {
    const auto n = polynomial.size, v = polynomial.vertices;
    if (n == 0) return {};
    const auto test = [&](std::size_t count, bool negative, bool find_margin) {
        auto margin = std::numeric_limits<double>::infinity();
        if (count == 0) return margin;
        auto sector = negative ? indices(0, count) : indices(n - count, n);
        for (std::size_t i = 0; i < v; ++i)
            for (std::size_t j = i; j < v; ++j) {
                auto matrix = block(polynomial.at(i, j), n, sector, sector);
                if (negative) for (auto &value : matrix) value = -value;
                double row_margin = std::numeric_limits<double>::infinity();
                for (std::size_t row = 0; row < count; ++row) {
                    const auto diagonal = matrix[row + row * count].real() - allowance;
                    if (diagonal <= 0) return -1.;
                    auto lower = diagonal;
                    for (std::size_t col = 0; col < count; ++col)
                        if (row != col) lower -= std::abs(matrix[row + col * count]);
                    row_margin = std::min(row_margin, lower);
                }
                if (row_margin > 0) {
                    margin = std::min(margin, row_margin);
                    continue;
                }
                if (find_margin) {
                    std::vector<double> eigenvalues;
                    linalg::diagonalize_hermitian_in_place(matrix, eigenvalues,
                        count, false, "occupation sector margin");
                    ++stats.norm_eigensystems;
                    margin = std::min(margin, eigenvalues.front() - allowance);
                } else {
                    for (std::size_t k = 0; k < count; ++k)
                        matrix[k + k * count] -= allowance;
                    if (linalg::cholesky_factor_lower(matrix.data(), count) != 0) return -1.;
                }
            }
        return margin;
    };
    const auto largest = [&](bool negative, std::size_t maximum) {
        // Definiteness requires every diagonal entry to have the right sign.
        // This cheap necessary test usually locates the whole safe sector;
        // certify it once before searching smaller projected blocks.
        for (std::size_t i = 0; i < v; ++i)
            for (std::size_t j = i; j < v; ++j) {
                const auto &control = polynomial.at(i, j);
                for (std::size_t rank = 0; rank < maximum; ++rank) {
                    const auto band = negative ? rank : n - 1 - rank;
                    const auto diagonal = control[band + band * n].real();
                    if ((negative ? -diagonal : diagonal) <= allowance) {
                        maximum = rank;
                        break;
                    }
                }
            }
        if (test(maximum, negative, false) > 0) return maximum;
        std::size_t low = 0, high = maximum;
        while (low < high) {
            const auto middle = low + (high - low + 1) / 2;
            if (test(middle, negative, false) > 0) low = middle;
            else high = middle - 1;
        }
        return low;
    };
    const auto negative = largest(true, n);
    const auto positive = largest(false, n - negative);
    const auto gap = std::min(test(negative, true, true), test(positive, false, true));
    return {negative, positive, std::isfinite(gap) ? gap : 0.};
}

Model build_model(const SpectralMesh &mesh,
                  adaptivesimplex::core::SimplexId simplex_id, double mu,
                  ChargeErrorStats &stats, std::optional<double> remainder) {
    auto interpolation = interpolate(mesh, simplex_id, mu, stats, remainder);
    const auto &simplex = mesh.geometry().simplices().simplex(simplex_id);
    const auto &anchor = mesh.eigensystems().get(simplex.vertex_ids.front());
    auto full = interpolation.polynomial.rotated(anchor.eigenvectors);
    const auto eta = interpolation.remainder;
    const auto sectors = sign_sectors(full, eta, stats);
    const auto n = full.size, v = full.vertices;
    const auto q = n - sectors.negative - sectors.positive;
    stats.initial_active_dimension_sum += q;
    if (q == n) return {std::move(full), 0, eta, eta, 0};
    if (q == 0) return {Polynomial{v, 0, {}}, sectors.negative, eta, eta, sectors.gap};

    const auto active = indices(sectors.negative, n - sectors.positive);
    auto safe = indices(0, sectors.negative);
    const auto positive = indices(n - sectors.positive, n);
    safe.insert(safe.end(), positive.begin(), positive.end());
    const auto s = safe.size();
    Matrix d0(s * s);
    for (std::size_t i = 0; i < s; ++i)
        d0[i + i * s] = anchor.eigenvalues[safe[i]] - mu;
    std::vector<Matrix> coupling, solution;
    double x = 0, b = 0, d = 0;
    for (std::size_t i = 0; i < v; ++i) {
        coupling.push_back(block(full.at(i, i), n, safe, active));
        auto solved = coupling.back();
        for (std::size_t col = 0; col < q; ++col)
            for (std::size_t row = 0; row < s; ++row)
                solved[row + col * s] /= d0[row + row * s].real();
        x = std::max(x, norm(solved));
        solution.push_back(std::move(solved));
    }
    Polynomial reduced{v, q, std::vector<Matrix>(v * v)};
    for (std::size_t i = 0; i < v; ++i)
        for (std::size_t j = i; j < v; ++j) {
            auto residual = block(full.at(i, j), n, safe, active);
            for (std::size_t k = 0; k < residual.size(); ++k)
                residual[k] -= .5 * (coupling[i][k] + coupling[j][k]);
            b = std::max(b, norm(residual));
            d = std::max(d, hermitian_norm_bound(
                difference(block(full.at(i, j), n, safe, safe), d0), s));
            auto control = block(full.at(i, j), n, active, active);
            // Polarization of B1* D0^-1 B1 gives its quadratic controls.
            linalg::matrix_multiply('C', 'N', q, q, s, -.5, coupling[i].data(), s,
                solution[j].data(), s, 1., control.data(), q);
            linalg::matrix_multiply('C', 'N', q, q, s, -.5, coupling[j].data(), s,
                solution[i].data(), s, 1., control.data(), q);
            reduced.at(j, i) = reduced.at(i, j) = std::move(control);
        }
    b += eta; d += eta;
    const auto f = b + d * x;
    const auto epsilon = eta + 2 * b * x + d * x * x + f * f / sectors.gap;
    ++stats.schur_reductions;
    stats.schur_evaluations += v * (v + 1) / 2;
    if (stats.minimum_active_dimension == 0 || q < stats.minimum_active_dimension)
        stats.minimum_active_dimension = q;
    return {std::move(reduced), sectors.negative, epsilon, eta, sectors.gap};
}

}  // namespace fermisimplex::occupation_detail
