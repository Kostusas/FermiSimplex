#include "internal.h"

#include "core/linear_algebra.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace lineartetrahedron::simplex_certificate::detail {
namespace {

void require_block_shape(
    const std::vector<std::vector<Complex>> &blocks,
    size_t size
) {
    const auto expected = size * size;
    for (const auto &block : blocks) {
        if (block.size() != expected) {
            throw std::runtime_error("simplex certificate: common-rank block shape mismatch");
        }
    }
}

std::vector<Complex> average_block(
    const std::vector<std::vector<Complex>> &blocks,
    size_t size
) {
    std::vector<Complex> result(size * size, Complex{0.0, 0.0});
    const auto scale = 1.0 / static_cast<double>(blocks.size());
    for (const auto &block : blocks) {
        for (size_t index = 0; index < result.size(); ++index) {
            result[index] += scale * block[index];
        }
    }
    return result;
}

std::vector<Complex> descending_eigenvectors(
    std::vector<Complex> block,
    size_t size
) {
    std::vector<double> eigenvalues;
    diagonalize_hermitian_in_place(
        block,
        eigenvalues,
        size,
        true,
        "simplex certificate common-rank estimator"
    );

    std::vector<Complex> basis(size * size, Complex{0.0, 0.0});
    for (size_t column = 0; column < size; ++column) {
        const auto source_column = size - 1 - column;
        for (size_t row = 0; row < size; ++row) {
            basis[column_major_index(row, column, size)] =
                block[column_major_index(row, source_column, size)];
        }
    }
    return basis;
}

std::vector<Complex> transformed_block(
    const std::vector<Complex> &block,
    std::span<const Complex> basis,
    size_t size
) {
    std::vector<Complex> product(size * size, Complex{0.0, 0.0});
    hemm(
        'L',
        size,
        size,
        Complex{1.0, 0.0},
        block.data(),
        size,
        basis.data(),
        size,
        Complex{0.0, 0.0},
        product.data(),
        size
    );

    std::vector<Complex> result(size * size, Complex{0.0, 0.0});
    gemm(
        'C',
        'N',
        size,
        size,
        size,
        Complex{1.0, 0.0},
        basis.data(),
        size,
        product.data(),
        size,
        Complex{0.0, 0.0},
        result.data(),
        size
    );
    return result;
}

}  // namespace

size_t estimate_common_rank(
    const std::vector<std::vector<Complex>> &blocks,
    size_t size,
    double tol
) {
    if (size == 0 || blocks.empty()) {
        return 0;
    }
    require_block_shape(blocks, size);

    const auto average = average_block(blocks, size);
    const auto basis = descending_eigenvectors(average, size);
    const auto basis_span = std::span<const Complex>(basis.data(), basis.size());

    auto rank = size;
    for (const auto &block : blocks) {
        rank = std::min(
            rank,
            positive_definite_prefix(transformed_block(block, basis_span, size), size, tol)
        );
    }
    return rank;
}

}  // namespace lineartetrahedron::simplex_certificate::detail
