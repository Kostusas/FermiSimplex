#include "internal.h"

#include <algorithm>
#include <limits>
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

std::vector<size_t> candidate_order_by_worst_margin(
    const std::vector<std::vector<Complex>> &blocks,
    size_t size
) {
    std::vector<std::pair<double, size_t>> margins;
    margins.reserve(size);
    for (size_t index = 0; index < size; ++index) {
        auto margin = std::numeric_limits<double>::infinity();
        for (const auto &block : blocks) {
            margin = std::min(
                margin,
                std::real(block[column_major_index(index, index, size)])
            );
        }
        margins.emplace_back(margin, index);
    }

    std::sort(
        margins.begin(),
        margins.end(),
        [](const auto &left, const auto &right) {
            if (left.first != right.first) {
                return left.first > right.first;
            }
            return left.second < right.second;
        }
    );

    std::vector<size_t> order;
    order.reserve(size);
    for (const auto &[unused_margin, index] : margins) {
        (void)unused_margin;
        order.push_back(index);
    }
    return order;
}

std::vector<Complex> permuted_block(
    const std::vector<Complex> &block,
    std::span<const size_t> order,
    size_t size
) {
    std::vector<Complex> result(size * size, Complex{0.0, 0.0});
    for (size_t column = 0; column < size; ++column) {
        const auto source_column = order[column];
        for (size_t row = 0; row < size; ++row) {
            result[column_major_index(row, column, size)] =
                block[column_major_index(order[row], source_column, size)];
        }
    }
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

    const auto order = candidate_order_by_worst_margin(blocks, size);
    const auto order_span = std::span<const size_t>(order.data(), order.size());

    auto rank = size;
    for (const auto &block : blocks) {
        rank = std::min(
            rank,
            positive_definite_prefix(permuted_block(block, order_span, size), size, tol)
        );
    }
    return rank;
}

}  // namespace lineartetrahedron::simplex_certificate::detail
