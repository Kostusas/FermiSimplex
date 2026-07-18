#include "certification/simplex/vertex_blocks.h"

#include "linalg/blas_lapack.h"

#include <algorithm>

namespace lineartetrahedron::certification::detail {
namespace {

std::vector<Complex> full_overlap(
    std::span<const Complex> anchor_vectors,
    size_t ndof,
    std::span<const Complex> target_vectors
) {
    std::vector<Complex> result(ndof * ndof, Complex{0.0, 0.0});
    linalg::matrix_multiply(
        'C',
        'N',
        ndof,
        ndof,
        ndof,
        Complex{1.0, 0.0},
        anchor_vectors.data(),
        ndof,
        target_vectors.data(),
        ndof,
        Complex{0.0, 0.0},
        result.data(),
        ndof
    );
    return result;
}

std::vector<Complex> scale_overlap_columns(
    const Complex *overlap,
    size_t overlap_rows,
    size_t rows,
    size_t columns,
    std::span<const double> target_eigenvalues,
    double mu
) {
    std::vector<Complex> result(rows * columns, Complex{0.0, 0.0});
    for (size_t column = 0; column < columns; ++column) {
        const auto scale = signed_eigenvalue(target_eigenvalues, column, mu);
        for (size_t row = 0; row < rows; ++row) {
            result[column_major_index(row, column, rows)] =
                scale * overlap[column_major_index(row, column, overlap_rows)];
        }
    }
    return result;
}

std::vector<Complex> submatrix_copy(
    std::span<const Complex> matrix,
    size_t rows,
    size_t row_offset,
    size_t column_offset,
    size_t sub_rows,
    size_t sub_columns
) {
    std::vector<Complex> result(sub_rows * sub_columns, Complex{0.0, 0.0});
    for (size_t column = 0; column < sub_columns; ++column) {
        for (size_t row = 0; row < sub_rows; ++row) {
            result[column_major_index(row, column, sub_rows)] =
                matrix[column_major_index(row + row_offset, column + column_offset, rows)];
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

std::vector<Complex> diagonal_sector_block(
    std::span<const double> eigenvalues,
    size_t offset,
    size_t size,
    double mu
) {
    std::vector<Complex> result(size * size, Complex{0.0, 0.0});
    for (size_t index = 0; index < size; ++index) {
        result[column_major_index(index, index, size)] =
            Complex{signed_eigenvalue(eigenvalues, offset + index, mu), 0.0};
    }
    return result;
}

}  // namespace

VertexBlocks build_anchor_vertex_blocks(
    size_t ndof,
    size_t nocc,
    std::span<const double> anchor_eigenvalues,
    double mu
) {
    const auto nunocc = ndof - nocc;
    VertexBlocks blocks;
    blocks.occupied_block = diagonal_sector_block(anchor_eigenvalues, 0, nocc, mu);
    blocks.unoccupied_block = diagonal_sector_block(anchor_eigenvalues, nocc, nunocc, mu);
    blocks.coupling_block.assign(nunocc * nocc, Complex{0.0, 0.0});
    return blocks;
}

VertexBlocks build_vertex_blocks(
    std::span<const Complex> anchor_vectors,
    size_t ndof,
    size_t nocc,
    std::span<const double> target_eigenvalues,
    std::span<const Complex> target_eigenvectors,
    double mu
) {
    const auto nunocc = ndof - nocc;
    const auto overlap = full_overlap(anchor_vectors, ndof, target_eigenvectors);
    const auto scaled_overlap =
        scale_overlap_columns(overlap.data(), ndof, ndof, ndof, target_eigenvalues, mu);

    std::vector<Complex> anchor_matrix(ndof * ndof, Complex{0.0, 0.0});
    linalg::matrix_multiply(
        'N',
        'C',
        ndof,
        ndof,
        ndof,
        Complex{1.0, 0.0},
        scaled_overlap.data(),
        ndof,
        overlap.data(),
        ndof,
        Complex{0.0, 0.0},
        anchor_matrix.data(),
        ndof
    );

    VertexBlocks blocks;
    blocks.unoccupied_block = submatrix_copy(
        std::span<const Complex>(anchor_matrix.data(), anchor_matrix.size()),
        ndof,
        nocc,
        nocc,
        nunocc,
        nunocc
    );
    blocks.occupied_block = submatrix_copy(
        std::span<const Complex>(anchor_matrix.data(), anchor_matrix.size()),
        ndof,
        0,
        0,
        nocc,
        nocc
    );
    blocks.coupling_block = submatrix_copy(
        std::span<const Complex>(anchor_matrix.data(), anchor_matrix.size()),
        ndof,
        nocc,
        0,
        nunocc,
        nocc
    );
    hermitize_square(blocks.unoccupied_block, nunocc);
    hermitize_square(blocks.occupied_block, nocc);
    return blocks;
}

}  // namespace lineartetrahedron::certification::detail
