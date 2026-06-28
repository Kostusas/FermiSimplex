#include "certificate/bounds/occupation_bounds.h"

#include "certificate/linalg/cholesky.h"
#include "certificate/bounds/mu_bounds.h"

#include <algorithm>
#include <limits>
#include <span>
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
            throw std::runtime_error("simplex certificate: occupation-bound block shape mismatch");
        }
    }
}

std::vector<size_t> ordered_subset_by_worst_margin(
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

double ordered_subset_mu_radius(
    const std::vector<std::vector<Complex>> &blocks,
    std::span<const size_t> order,
    size_t size,
    size_t rank,
    double tol,
    double margin
) {
    if (rank == 0) {
        return std::numeric_limits<double>::infinity();
    }

    auto radius = std::numeric_limits<double>::infinity();
    const auto total_margin = certificate_margin(tol) + margin;
    for (const auto &block : blocks) {
        const auto permuted = permuted_block(block, order, size);
        for (size_t row = 0; row < rank; ++row) {
            const auto row_bound =
                gershgorin_row_lower_bound(permuted, size, row, rank);
            radius = std::min(radius, row_bound - total_margin);
        }
    }
    return std::max(0.0, radius);
}

}  // namespace

OccupationRankEstimate estimate_ordered_subset_rank_with_mu_radius(
    const std::vector<std::vector<Complex>> &blocks,
    size_t size,
    double tol,
    double margin
) {
    if (size == 0 || blocks.empty()) {
        return OccupationRankEstimate{
            .rank = 0,
            .mu_radius = std::numeric_limits<double>::infinity(),
        };
    }
    require_block_shape(blocks, size);

    const auto order = ordered_subset_by_worst_margin(blocks, size);
    const auto order_span = std::span<const size_t>(order.data(), order.size());

    auto rank = size;
    for (const auto &block : blocks) {
        auto permuted = permuted_block(block, order_span, size);
        rank = std::min(
            rank,
            positive_definite(permuted, size, certificate_margin(tol) + margin).accepted_rank
        );
    }
    return OccupationRankEstimate{
        .rank = rank,
        .mu_radius = ordered_subset_mu_radius(blocks, order_span, size, rank, tol, margin),
    };
}

size_t estimate_ordered_subset_rank(
    const std::vector<std::vector<Complex>> &blocks,
    size_t size,
    double tol,
    double margin
) {
    return estimate_ordered_subset_rank_with_mu_radius(blocks, size, tol, margin).rank;
}

}  // namespace lineartetrahedron::simplex_certificate::detail
