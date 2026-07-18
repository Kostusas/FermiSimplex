#include "certification/linalg/rotated_blocks.h"

#include "linalg/blas_lapack.h"

namespace fermisimplex::certification::detail {
namespace {

void add_lower_triangle(
    std::vector<Complex> &target,
    std::span<const Complex> value,
    size_t size
) {
    for (size_t column = 0; column < size; ++column) {
        for (size_t row = column; row < size; ++row) {
            target[column_major_index(row, column, size)] +=
                value[column_major_index(row, column, size)];
        }
    }
}

std::vector<Complex> scaled_square_copy(
    std::span<const Complex> block,
    size_t size,
    double scale
) {
    std::vector<Complex> result(size * size, Complex{0.0, 0.0});
    for (size_t index = 0; index < result.size(); ++index) {
        result[index] = scale * block[index];
    }
    return result;
}

}  // namespace

std::vector<Complex> rotated_block(
    std::span<const Complex> base_block,
    double base_scale,
    std::span<const Complex> opposite_block,
    std::span<const Complex> coupling,
    std::span<const Complex> rotation,
    size_t size,
    size_t opposite_size,
    double cross_scale,
    double quadratic_scale
) {
    auto result = scaled_square_copy(base_block, size, base_scale);
    if (size == 0 || opposite_size == 0) {
        return result;
    }

    if (cross_scale != 0.0) {
        linalg::hermitian_rank_2k_update(
            'N',
            size,
            opposite_size,
            Complex{cross_scale, 0.0},
            coupling.data(),
            size,
            rotation.data(),
            size,
            1.0,
            result.data(),
            size
        );
    }

    if (quadratic_scale == 0.0) {
        return result;
    }
    std::vector<Complex> product(size * opposite_size, Complex{0.0, 0.0});
    linalg::hermitian_matrix_multiply(
        'R',
        size,
        opposite_size,
        Complex{1.0, 0.0},
        opposite_block.data(),
        opposite_size,
        rotation.data(),
        size,
        Complex{0.0, 0.0},
        product.data(),
        size
    );
    std::vector<Complex> quadratic(size * size, Complex{0.0, 0.0});
    linalg::matrix_multiply(
        'N',
        'C',
        size,
        size,
        opposite_size,
        Complex{quadratic_scale, 0.0},
        product.data(),
        size,
        rotation.data(),
        size,
        Complex{0.0, 0.0},
        quadratic.data(),
        size
    );
    add_lower_triangle(result, std::span<const Complex>(quadratic.data(), quadratic.size()), size);
    return result;
}

void subtract_frame_margin(
    std::vector<Complex> &block,
    std::span<const Complex> rotation,
    size_t size,
    size_t opposite_size,
    double margin
) {
    if (margin == 0.0 || size == 0) {
        return;
    }
    if (opposite_size != 0) {
        linalg::matrix_multiply(
            'N',
            'C',
            size,
            size,
            opposite_size,
            Complex{-margin, 0.0},
            rotation.data(),
            size,
            rotation.data(),
            size,
            Complex{1.0, 0.0},
            block.data(),
            size
        );
    }
    for (size_t index = 0; index < size; ++index) {
        block[column_major_index(index, index, size)] -= margin;
    }
}

}  // namespace fermisimplex::certification::detail
