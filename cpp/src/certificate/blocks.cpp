#include "internal.h"

#include <algorithm>

namespace lineartetrahedron::simplex_certificate::detail {
namespace {

std::vector<Complex> full_overlap(
    std::span<const Complex> anchor_vectors,
    size_t ndof,
    std::span<const Complex> target_vectors
) {
    std::vector<Complex> result(ndof * ndof, Complex{0.0, 0.0});
    gemm(
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
    const VertexSpectra &target,
    double mu
) {
    std::vector<Complex> result(rows * columns, Complex{0.0, 0.0});
    for (size_t column = 0; column < columns; ++column) {
        const auto scale = signed_eigenvalue(target, column, mu);
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

}  // namespace

void negate_in_place(std::vector<Complex> &matrix) {
    for (auto &value : matrix) {
        value = -value;
    }
}

VertexBlocks build_vertex_blocks(
    std::span<const Complex> anchor_vectors,
    size_t ndof,
    size_t npos,
    size_t nneg,
    const VertexSpectra &target,
    double mu
) {
    const auto target_vectors =
        std::span<const Complex>(target.eigenvectors.data(), target.eigenvectors.size());
    const auto overlap = full_overlap(anchor_vectors, ndof, target_vectors);
    const auto scaled_overlap =
        scale_overlap_columns(overlap.data(), ndof, ndof, ndof, target, mu);

    std::vector<Complex> anchor_matrix(ndof * ndof, Complex{0.0, 0.0});
    gemm(
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
    blocks.positive_same_sign = submatrix_copy(
        std::span<const Complex>(anchor_matrix.data(), anchor_matrix.size()),
        ndof,
        nneg,
        nneg,
        npos,
        npos
    );
    blocks.negative_same_sign = submatrix_copy(
        std::span<const Complex>(anchor_matrix.data(), anchor_matrix.size()),
        ndof,
        0,
        0,
        nneg,
        nneg
    );
    blocks.positive_negative_coupling = submatrix_copy(
        std::span<const Complex>(anchor_matrix.data(), anchor_matrix.size()),
        ndof,
        nneg,
        0,
        npos,
        nneg
    );
    hermitize_square(blocks.positive_same_sign, npos);
    hermitize_square(blocks.negative_same_sign, nneg);
    return blocks;
}

}  // namespace lineartetrahedron::simplex_certificate::detail
