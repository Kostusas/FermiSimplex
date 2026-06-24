#include "internal.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace lineartetrahedron::simplex_certificate::detail {
namespace {

Complex hermitian_entry(
    std::span<const Complex> matrix,
    size_t row,
    size_t column,
    size_t size
) {
    if (row >= column) {
        return matrix[column_major_index(row, column, size)];
    }
    return std::conj(matrix[column_major_index(column, row, size)]);
}

std::vector<Complex> full_hermitian_copy(std::span<const Complex> matrix, size_t size) {
    std::vector<Complex> result(size * size, Complex{0.0, 0.0});
    for (size_t column = 0; column < size; ++column) {
        for (size_t row = 0; row < size; ++row) {
            result[column_major_index(row, column, size)] =
                hermitian_entry(matrix, row, column, size);
        }
    }
    return result;
}

void hermitize_square(std::vector<Complex> &matrix, size_t size) {
    for (size_t column = 0; column < size; ++column) {
        matrix[column_major_index(column, column, size)] =
            Complex{std::real(matrix[column_major_index(column, column, size)]), 0.0};
        for (size_t row = column + 1; row < size; ++row) {
            const auto lower = matrix[column_major_index(row, column, size)];
            const auto upper = std::conj(matrix[column_major_index(column, row, size)]);
            const auto value = 0.5 * (lower + upper);
            matrix[column_major_index(row, column, size)] = value;
            matrix[column_major_index(column, row, size)] = std::conj(value);
        }
    }
}

std::vector<Complex> cholesky_transformed_operator(
    std::span<const Complex> matrix,
    std::span<const Complex> metric,
    size_t size
) {
    auto factor = full_hermitian_copy(metric, size);
    if (!cholesky_lower_in_place(factor, size)) {
        throw std::runtime_error("generalized_hermitian_min_eigenvalue_lanczos: metric is not positive definite");
    }

    const auto dense_matrix = full_hermitian_copy(matrix, size);
    std::vector<Complex> left_solved(size * size, Complex{0.0, 0.0});
    for (size_t column = 0; column < size; ++column) {
        for (size_t row = 0; row < size; ++row) {
            auto value = dense_matrix[column_major_index(row, column, size)];
            for (size_t index = 0; index < row; ++index) {
                value -=
                    factor[column_major_index(row, index, size)] *
                    left_solved[column_major_index(index, column, size)];
            }
            left_solved[column_major_index(row, column, size)] =
                value / factor[column_major_index(row, row, size)];
        }
    }

    std::vector<Complex> transformed(size * size, Complex{0.0, 0.0});
    for (size_t row = 0; row < size; ++row) {
        for (size_t column = 0; column < size; ++column) {
            auto value = left_solved[column_major_index(row, column, size)];
            for (size_t index = 0; index < column; ++index) {
                value -=
                    transformed[column_major_index(row, index, size)] *
                    std::conj(factor[column_major_index(column, index, size)]);
            }
            transformed[column_major_index(row, column, size)] =
                value / std::conj(factor[column_major_index(column, column, size)]);
        }
    }

    hermitize_square(transformed, size);
    return transformed;
}

}  // namespace

std::vector<Complex> positive_frame_metric(
    std::span<const Complex> rotation,
    size_t npos,
    size_t nneg
) {
    if (rotation.size() != npos * nneg) {
        throw std::runtime_error("positive_frame_metric: rotation size mismatch");
    }
    std::vector<Complex> metric(npos * npos, Complex{0.0, 0.0});
    for (size_t index = 0; index < npos; ++index) {
        metric[column_major_index(index, index, npos)] = Complex{1.0, 0.0};
    }
    if (npos != 0 && nneg != 0) {
        gemm(
            'N',
            'C',
            npos,
            npos,
            nneg,
            Complex{1.0, 0.0},
            rotation.data(),
            npos,
            rotation.data(),
            npos,
            Complex{1.0, 0.0},
            metric.data(),
            npos
        );
        hermitize_square(metric, npos);
    }
    return metric;
}

std::vector<Complex> negative_frame_metric(
    std::span<const Complex> rotation,
    size_t npos,
    size_t nneg
) {
    if (rotation.size() != npos * nneg) {
        throw std::runtime_error("negative_frame_metric: rotation size mismatch");
    }
    std::vector<Complex> metric(nneg * nneg, Complex{0.0, 0.0});
    for (size_t index = 0; index < nneg; ++index) {
        metric[column_major_index(index, index, nneg)] = Complex{1.0, 0.0};
    }
    if (npos != 0 && nneg != 0) {
        gemm(
            'C',
            'N',
            nneg,
            nneg,
            npos,
            Complex{1.0, 0.0},
            rotation.data(),
            npos,
            rotation.data(),
            npos,
            Complex{1.0, 0.0},
            metric.data(),
            nneg
        );
        hermitize_square(metric, nneg);
    }
    return metric;
}

double generalized_hermitian_min_eigenvalue_lanczos(
    std::span<const Complex> matrix,
    std::span<const Complex> metric,
    size_t size,
    double absolute_uncertainty
) {
    if (matrix.size() != size * size || metric.size() != size * size) {
        throw std::runtime_error("generalized_hermitian_min_eigenvalue_lanczos: matrix size mismatch");
    }
    if (size == 0) {
        return std::numeric_limits<double>::infinity();
    }
    const auto transformed = cholesky_transformed_operator(matrix, metric, size);
    return hermitian_min_eigenvalue_lanczos(
        std::span<const Complex>(transformed.data(), transformed.size()),
        size,
        absolute_uncertainty
    );
}

}  // namespace lineartetrahedron::simplex_certificate::detail
