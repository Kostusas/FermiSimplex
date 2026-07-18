#include "certification/bounds/mu_bounds.h"

#include "certification/linalg/cholesky.h"
#include "linalg/blas_lapack.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <vector>

namespace lineartetrahedron::certification::detail {

Complex hermitian_entry(
    const std::vector<Complex> &matrix,
    size_t size,
    size_t row,
    size_t column
) {
    if (row >= column) {
        return matrix[column_major_index(row, column, size)];
    }
    return std::conj(matrix[column_major_index(column, row, size)]);
}

// Gershgorin lower row bound: g_i(B) = Re B_ii - sum_{j != i} |B_ij|.
double gershgorin_row_lower_bound(
    const std::vector<Complex> &matrix,
    size_t size,
    size_t row,
    size_t active_size
) {
    auto bound = std::real(matrix[column_major_index(row, row, size)]);
    for (size_t column = 0; column < active_size; ++column) {
        if (column != row) {
            bound -= std::abs(hermitian_entry(matrix, size, row, column));
        }
    }
    return bound;
}

// For an oriented rotated frame X, changing mu by delta shifts the occupation
// block as B(mu + delta) = B(mu) - delta G, where G = I + X X*.
// Row bounds for G give a conservative denominator for the reusable
// chemical-potential radius: |delta| <= (g_i(B) - eta) / g_i^row(G).
double occupation_block_mu_radius(
    const std::vector<Complex> &occupation_block,
    std::span<const double> gram_row_bounds,
    size_t size,
    double tol
) {
    if (size == 0) {
        return std::numeric_limits<double>::infinity();
    }

    auto radius = std::numeric_limits<double>::infinity();
    const auto margin = certificate_margin(tol);
    for (size_t row = 0; row < size; ++row) {
        const auto denominator = gram_row_bounds[row];
        if (denominator <= 0.0) {
            return 0.0;
        }
        const auto row_radius =
            (gershgorin_row_lower_bound(occupation_block, size, row, size) - margin) /
            denominator;
        radius = std::min(radius, row_radius);
    }
    return std::max(0.0, radius);
}

std::vector<double> frame_gram_row_bounds(
    std::span<const Complex> rotation,
    size_t size,
    size_t opposite_size
) {
    std::vector<double> result(size, 1.0);
    if (size == 0 || opposite_size == 0) {
        return result;
    }

    std::vector<Complex> gram(size * size, Complex{0.0, 0.0});
    linalg::matrix_multiply(
        'N',
        'C',
        size,
        size,
        opposite_size,
        Complex{1.0, 0.0},
        rotation.data(),
        size,
        rotation.data(),
        size,
        Complex{0.0, 0.0},
        gram.data(),
        size
    );
    for (size_t row = 0; row < size; ++row) {
        result[row] += std::real(gram[column_major_index(row, row, size)]);
        for (size_t column = 0; column < size; ++column) {
            if (column != row) {
                result[row] += std::abs(hermitian_entry(gram, size, row, column));
            }
        }
    }
    return result;
}

}  // namespace lineartetrahedron::certification::detail
