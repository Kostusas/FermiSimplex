#include "internal.h"

namespace lineartetrahedron::simplex_certificate::detail {
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

}  // namespace

void subtract_positive_frame_margin(
    std::vector<Complex> &block,
    std::span<const Complex> rotation,
    size_t npos,
    size_t nneg,
    double margin
) {
    if (margin == 0.0 || npos == 0) {
        return;
    }
    if (nneg != 0) {
        gemm(
            'N',
            'C',
            npos,
            npos,
            nneg,
            Complex{-margin, 0.0},
            rotation.data(),
            npos,
            rotation.data(),
            npos,
            Complex{1.0, 0.0},
            block.data(),
            npos
        );
    }
    for (size_t index = 0; index < npos; ++index) {
        block[column_major_index(index, index, npos)] -= margin;
    }
}

void subtract_negative_frame_margin(
    std::vector<Complex> &block,
    std::span<const Complex> rotation,
    size_t npos,
    size_t nneg,
    double margin
) {
    if (margin == 0.0 || nneg == 0) {
        return;
    }
    if (npos != 0) {
        gemm(
            'C',
            'N',
            nneg,
            nneg,
            npos,
            Complex{-margin, 0.0},
            rotation.data(),
            npos,
            rotation.data(),
            npos,
            Complex{1.0, 0.0},
            block.data(),
            nneg
        );
    }
    for (size_t index = 0; index < nneg; ++index) {
        block[column_major_index(index, index, nneg)] -= margin;
    }
}

std::vector<Complex> rotated_positive_block(
    const VertexBlocks &blocks,
    std::span<const Complex> rotation,
    size_t npos,
    size_t nneg
) {
    auto result = blocks.positive_same_sign;
    if (npos == 0 || nneg == 0) {
        return result;
    }

    her2k(
        'N',
        npos,
        nneg,
        Complex{1.0, 0.0},
        blocks.positive_negative_coupling.data(),
        npos,
        rotation.data(),
        npos,
        1.0,
        result.data(),
        npos
    );

    std::vector<Complex> product(npos * nneg, Complex{0.0, 0.0});
    hemm(
        'R',
        npos,
        nneg,
        Complex{1.0, 0.0},
        blocks.negative_same_sign.data(),
        nneg,
        rotation.data(),
        npos,
        Complex{0.0, 0.0},
        product.data(),
        npos
    );
    std::vector<Complex> quadratic(npos * npos, Complex{0.0, 0.0});
    gemm(
        'N',
        'C',
        npos,
        npos,
        nneg,
        Complex{1.0, 0.0},
        product.data(),
        npos,
        rotation.data(),
        npos,
        Complex{0.0, 0.0},
        quadratic.data(),
        npos
    );
    add_lower_triangle(result, std::span<const Complex>(quadratic.data(), quadratic.size()), npos);
    return result;
}

std::vector<Complex> rotated_negative_block(
    const VertexBlocks &blocks,
    std::span<const Complex> rotation,
    size_t npos,
    size_t nneg
) {
    auto result = blocks.negative_same_sign;
    if (npos == 0 || nneg == 0) {
        return result;
    }

    her2k(
        'C',
        nneg,
        npos,
        Complex{-1.0, 0.0},
        blocks.positive_negative_coupling.data(),
        npos,
        rotation.data(),
        npos,
        1.0,
        result.data(),
        nneg
    );

    std::vector<Complex> product(npos * nneg, Complex{0.0, 0.0});
    hemm(
        'L',
        npos,
        nneg,
        Complex{1.0, 0.0},
        blocks.positive_same_sign.data(),
        npos,
        rotation.data(),
        npos,
        Complex{0.0, 0.0},
        product.data(),
        npos
    );
    std::vector<Complex> quadratic(nneg * nneg, Complex{0.0, 0.0});
    gemm(
        'C',
        'N',
        nneg,
        nneg,
        npos,
        Complex{1.0, 0.0},
        rotation.data(),
        npos,
        product.data(),
        npos,
        Complex{0.0, 0.0},
        quadratic.data(),
        nneg
    );
    add_lower_triangle(result, std::span<const Complex>(quadratic.data(), quadratic.size()), nneg);
    return result;
}

}  // namespace lineartetrahedron::simplex_certificate::detail
