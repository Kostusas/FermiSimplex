#include "certificate/simplex/occupation_certificate.h"

#include "certificate/linalg/cholesky.h"
#include "certificate/bounds/mu_bounds.h"
#include "certificate/bounds/occupation_bounds.h"
#include "certificate/linalg/rotated_blocks.h"

#include <span>
#include <utility>
#include <vector>

namespace lineartetrahedron::simplex_certificate::detail {
namespace {

SimplexCertificate certificate(SimplexCertificateStatus status, OccupationBounds bounds) {
    return SimplexCertificate{
        .status = status,
        .occupation_bounds = bounds,
    };
}

OccupationBounds unconstrained_occupation(size_t ndof) {
    return OccupationBounds{.lower = 0, .upper = ndof};
}

}  // namespace

size_t sector_size(OccupationSector sector, size_t ndof, size_t nocc) {
    return sector == OccupationSector::Unoccupied ? ndof - nocc : nocc;
}

size_t opposite_sector_size(OccupationSector sector, size_t ndof, size_t nocc) {
    return sector == OccupationSector::Unoccupied ? nocc : ndof - nocc;
}

OccupationSectorCheck check_occupation_sector(
    OccupationSector sector,
    const VertexBlocks &blocks,
    std::span<const Complex> rotation,
    std::span<const double> gram_row_bounds,
    size_t size,
    size_t opposite_size,
    double margin,
    double tol
) {
    auto coupling_storage = std::vector<Complex>{};
    auto base_scale = 1.0;
    auto quadratic_scale = 1.0;
    auto base_block = std::span<const Complex>(
        blocks.unoccupied_block.data(),
        blocks.unoccupied_block.size()
    );
    auto opposite_block = std::span<const Complex>(
        blocks.occupied_block.data(),
        blocks.occupied_block.size()
    );
    auto coupling = std::span<const Complex>(
        blocks.coupling_block.data(),
        blocks.coupling_block.size()
    );

    if (sector == OccupationSector::Occupied) {
        base_block = std::span<const Complex>(
            blocks.occupied_block.data(),
            blocks.occupied_block.size()
        );
        opposite_block = std::span<const Complex>(
            blocks.unoccupied_block.data(),
            blocks.unoccupied_block.size()
        );
        coupling_storage = adjoint_rectangular_copy(
            std::span<const Complex>(
                blocks.coupling_block.data(),
                blocks.coupling_block.size()
            ),
            opposite_size,
            size
        );
        coupling = std::span<const Complex>(
            coupling_storage.data(),
            coupling_storage.size()
        );
        base_scale = -1.0;
        quadratic_scale = -1.0;
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
    subtract_frame_margin(block, rotation, size, opposite_size, margin);

    const auto radius = occupation_block_mu_radius(block, gram_row_bounds, size, tol);
    return OccupationSectorCheck{
        .passed = positive_definite(
            std::move(block),
            size,
            certificate_margin(tol)
        ).passed,
        .mu_radius = radius,
    };
}

SimplexCertificate occupation_bounded_certificate(
    SimplexCertificateStatus status,
    double mu,
    size_t ndof,
    size_t nocc,
    const std::vector<VertexBlocks> &blocks,
    double tol,
    double margin,
    bool estimate_occupation_bounds
) {
    auto result = certificate(
        status,
        unconstrained_occupation(ndof)
    );
    if (!estimate_occupation_bounds) {
        return result;
    }

    const auto nunocc = ndof - nocc;
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
        estimate_ordered_subset_rank_with_mu_radius(occupied_blocks, nocc, tol, margin);
    const auto unoccupied_estimate =
        estimate_ordered_subset_rank_with_mu_radius(unoccupied_blocks, nunocc, tol, margin);
    const auto lower = occupied_estimate.rank;
    const auto upper = ndof - unoccupied_estimate.rank;
    if (lower <= upper) {
        result.occupation_bounds = OccupationBounds{.lower = lower, .upper = upper};
        result.mu_interval = MuInterval{
            .lower = mu - occupied_estimate.mu_radius,
            .upper = mu + unoccupied_estimate.mu_radius,
        };
    }
    return result;
}

SimplexCertificate occupation_bounded_inconclusive(
    double mu,
    size_t ndof,
    size_t nocc,
    const std::vector<VertexBlocks> &blocks,
    double tol,
    double margin,
    bool estimate_occupation_bounds
) {
    return occupation_bounded_certificate(
        SimplexCertificateStatus::Inconclusive,
        mu,
        ndof,
        nocc,
        blocks,
        tol,
        margin,
        estimate_occupation_bounds
    );
}

}  // namespace lineartetrahedron::simplex_certificate::detail
