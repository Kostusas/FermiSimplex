#include "certification/simplex/occupation_certificate.h"

#include "certification/linalg/cholesky.h"
#include "certification/bounds/mu_bounds.h"
#include "certification/bounds/occupation_bounds.h"
#include "certification/linalg/rotated_blocks.h"

#include <cmath>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lineartetrahedron::certification::detail {
namespace {

OccupationBounds unconstrained_occupation(size_t ndof) {
    return OccupationBounds{.lower = 0, .upper = ndof};
}

size_t square_matrix_size(std::span<const Complex> matrix) {
    const auto size = static_cast<size_t>(std::sqrt(matrix.size()));
    if (size * size != matrix.size()) {
        throw std::logic_error("occupation certificate: sector block must be square");
    }
    return size;
}

OccupationSectorCheck check_oriented_sector(
    std::span<const Complex> base_block,
    double base_scale,
    std::span<const Complex> opposite_block,
    std::span<const Complex> coupling,
    std::span<const Complex> rotation,
    std::span<const double> gram_row_bounds,
    double quadratic_scale,
    double linearization_error_bound,
    double tolerance
) {
    const auto size = square_matrix_size(base_block);
    const auto opposite_size = square_matrix_size(opposite_block);
    if (rotation.size() != size * opposite_size ||
        coupling.size() != size * opposite_size ||
        gram_row_bounds.size() != size) {
        throw std::logic_error("occupation certificate: inconsistent oriented block sizes");
    }

    auto block = rotated_block(
        base_block,
        base_scale,
        opposite_block,
        coupling,
        rotation,
        size,
        opposite_size,
        1.0,
        quadratic_scale
    );
    subtract_frame_margin(
        block,
        rotation,
        size,
        opposite_size,
        linearization_error_bound
    );

    const auto radius = occupation_block_mu_radius(
        block,
        gram_row_bounds,
        size,
        tolerance
    );
    return OccupationSectorCheck{
        .passed = positive_definite(
            std::move(block),
            size,
            certificate_margin(tolerance)
        ).passed,
        .mu_radius = radius,
    };
}

}  // namespace

OccupationSectorCheck check_unoccupied_sector(
    const VertexBlocks &blocks,
    std::span<const Complex> rotation,
    std::span<const double> gram_row_bounds,
    double linearization_error_bound,
    double tolerance
) {
    return check_oriented_sector(
        blocks.unoccupied_block,
        1.0,
        blocks.occupied_block,
        blocks.coupling_block,
        rotation,
        gram_row_bounds,
        1.0,
        linearization_error_bound,
        tolerance
    );
}

OccupationSectorCheck check_occupied_sector(
    const VertexBlocks &blocks,
    std::span<const Complex> rotation,
    std::span<const double> gram_row_bounds,
    double linearization_error_bound,
    double tolerance
) {
    const auto occupied_size = square_matrix_size(blocks.occupied_block);
    const auto unoccupied_size = square_matrix_size(blocks.unoccupied_block);
    const auto coupling = adjoint_rectangular_copy(
        blocks.coupling_block,
        unoccupied_size,
        occupied_size
    );
    return check_oriented_sector(
        blocks.occupied_block,
        -1.0,
        blocks.unoccupied_block,
        coupling,
        rotation,
        gram_row_bounds,
        -1.0,
        linearization_error_bound,
        tolerance
    );
}

SimplexCertificate make_unresolved_certificate(
    SimplexCertificateStatus status,
    double mu,
    const std::vector<VertexBlocks> &blocks,
    double linearization_error_bound,
    double tolerance
) {
    if (blocks.empty()) {
        throw std::logic_error("occupation certificate: simplex blocks must not be empty");
    }
    const auto nocc = square_matrix_size(blocks.front().occupied_block);
    const auto nunocc = square_matrix_size(blocks.front().unoccupied_block);
    const auto ndof = nocc + nunocc;
    auto result = SimplexCertificate{
        .status = status,
        .occupation_bounds = unconstrained_occupation(ndof),
    };
    std::vector<std::vector<Complex>> unoccupied_blocks;
    std::vector<std::vector<Complex>> occupied_blocks;
    unoccupied_blocks.reserve(blocks.size());
    occupied_blocks.reserve(blocks.size());
    for (const auto &block : blocks) {
        unoccupied_blocks.push_back(block.unoccupied_block);

        auto occupied_block = block.occupied_block;
        for (auto &value : occupied_block) {
            value = -value;
        }
        occupied_blocks.push_back(std::move(occupied_block));
    }

    const auto occupied_estimate =
        estimate_ordered_subset_rank_with_mu_radius(
            occupied_blocks,
            nocc,
            linearization_error_bound,
            tolerance
        );
    const auto unoccupied_estimate =
        estimate_ordered_subset_rank_with_mu_radius(
            unoccupied_blocks,
            nunocc,
            linearization_error_bound,
            tolerance
        );
    const auto lower = occupied_estimate.rank;
    const auto upper = ndof - unoccupied_estimate.rank;
    if (lower <= upper) {
        result.occupation_bounds = OccupationBounds{.lower = lower, .upper = upper};
        if (
            status == SimplexCertificateStatus::Inconclusive &&
            lower == upper
        ) {
            result.status = SimplexCertificateStatus::CertifiedGapped;
        }
        result.mu_interval = MuInterval{
            .lower = mu - occupied_estimate.mu_radius,
            .upper = mu + unoccupied_estimate.mu_radius,
        };
    }
    return result;
}

}  // namespace lineartetrahedron::certification::detail
