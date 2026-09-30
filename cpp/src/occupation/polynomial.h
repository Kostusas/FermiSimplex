#pragma once

#include "linalg/blas_lapack.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace fermisimplex::occupation_detail {

using Matrix = std::vector<std::complex<double>>;
using Weights = std::vector<double>;

inline double norm(const Matrix &matrix) {
    double squared = 0;
    for (const auto value : matrix) squared += std::norm(value);
    return std::sqrt(squared);
}

inline double hermitian_norm_bound(const Matrix &matrix, std::size_t n) {
    std::vector<double> row_sums(n);
    double squared = 0;
    // Visit column-major storage once, retaining both triangles so roundoff
    // asymmetry does not disappear from either bound.
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
            const auto value = matrix[i + j * n];
            row_sums[i] += std::abs(value);
            squared += std::norm(value);
        }
    }
    const auto maximum = row_sums.empty() ? 0. :
        *std::max_element(row_sums.begin(), row_sums.end());
    return std::min(std::sqrt(squared), maximum);
}

inline Matrix difference(const Matrix &a, const Matrix &b) {
    auto result = a;
    for (std::size_t i = 0; i < a.size(); ++i) result[i] -= b[i];
    return result;
}

inline Matrix rotate(const Matrix &matrix, const Matrix &basis, std::size_t n) {
    if (n == 1) return matrix;
    Matrix temporary(n * n), result(n * n);
    linalg::matrix_multiply('N', 'N', n, n, n, 1., matrix.data(), n,
        basis.data(), n, 0., temporary.data(), n);
    linalg::matrix_multiply('C', 'N', n, n, n, 1., basis.data(), n,
        temporary.data(), n, 0., result.data(), n);
    return result;
}

inline Matrix block(const Matrix &matrix, std::size_t n,
                    std::span<const std::size_t> rows,
                    std::span<const std::size_t> columns) {
    Matrix result(rows.size() * columns.size());
    for (std::size_t j = 0; j < columns.size(); ++j)
        for (std::size_t i = 0; i < rows.size(); ++i)
            result[i + j * rows.size()] = matrix[rows[i] + columns[j] * n];
    return result;
}

// Symmetric control table: P(lambda) = sum_ij lambda_i lambda_j C_ij.
// C_ij=C_ji is a Hermitian matrix, including for i != j.
struct Polynomial {
    std::size_t vertices = 0;
    std::size_t size = 0;
    std::vector<Matrix> controls;

    Polynomial(std::size_t vertex_count, std::size_t matrix_size)
        : vertices(vertex_count), size(matrix_size),
          controls(vertex_count * (vertex_count + 1) / 2) {}

    const Matrix &at(std::size_t i, std::size_t j) const {
        if (i > j) std::swap(i, j);
        return controls[j * (j + 1) / 2 + i];
    }
    Matrix &at(std::size_t i, std::size_t j) {
        return const_cast<Matrix &>(std::as_const(*this).at(i, j));
    }
    Matrix blossom(const Weights &a, const Weights &b) const {
        Matrix result(size * size);
        for (std::size_t i = 0; i < vertices; ++i)
            for (std::size_t j = i; j < vertices; ++j) {
                const double weight = a[i] * b[j] + (i == j ? 0. : a[j] * b[i]);
                if (weight == 0) continue;
                for (std::size_t k = 0; k < result.size(); ++k)
                    result[k] += weight * at(i, j)[k];
            }
        return result;
    }
    Matrix center() const {
        Weights weights(vertices, 1. / vertices);
        return blossom(weights, weights);
    }
    Polynomial restrict_to(const std::vector<Weights> &points) const {
        Polynomial result{vertices, size};
        for (std::size_t i = 0; i < vertices; ++i)
            for (std::size_t j = i; j < vertices; ++j)
                result.at(i, j) = blossom(points[i], points[j]);
        return result;
    }
    Polynomial rotated(const Matrix &basis) const {
        Polynomial result{vertices, size};
        for (std::size_t i = 0; i < vertices; ++i)
            for (std::size_t j = i; j < vertices; ++j)
                result.at(i, j) = rotate(at(i, j), basis, size);
        return result;
    }
};

// Retain the current frame when its center is diagonal to working precision.
// No entries are dropped: their row sums still enter every occupation bound.
inline bool diagonal_center(const Matrix &center, std::size_t n) {
    double off_diagonal = 0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            if (i != j) off_diagonal += std::norm(center[i + j * n]);
    return std::sqrt(off_diagonal) <= 64 * std::numeric_limits<double>::epsilon() *
        std::max(1., norm(center));
}

inline std::vector<Weights> unit_weights(std::size_t vertices) {
    std::vector<Weights> result(vertices, Weights(vertices));
    for (std::size_t i = 0; i < vertices; ++i) result[i][i] = 1;
    return result;
}

}  // namespace fermisimplex::occupation_detail
