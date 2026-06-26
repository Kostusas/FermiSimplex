#include "certificate/simplex_certificate.h"

#include "certificate/simplex/anchor_selection.h"
#include "certificate/bounds/mu_bounds.h"
#include "certificate/simplex/occupation_certificate.h"
#include "certificate/simplex/simplex_blocks.h"

#include <algorithm>
#include <limits>
#include <span>
#include <stdexcept>

namespace lineartetrahedron::simplex_certificate {
namespace {

SimplexCertificate certificate(SimplexCertificateStatus status, OccupationBounds bounds) {
    return SimplexCertificate{
        .status = status,
        .occupation_bounds = bounds,
    };
}

OccupationBounds exact_occupation(size_t occupation) {
    return OccupationBounds{.lower = occupation, .upper = occupation};
}

}  // namespace

SimplexCertificate certify_simplex_gap(
    const core::Geometry &geometry,
    core::SimplexId simplex_id,
    const core::VertexCache<VertexSpectra> &vertex_cache,
    double mu,
    double margin,
    double tol,
    bool estimate_occupation_bounds
) {
    using namespace detail;

    const auto &simplex = geometry.simplices().simplex(simplex_id);
    if (simplex.vertex_ids.empty()) {
        throw std::runtime_error("certify_simplex_gap: simplex must not be empty");
    }

    const auto anchor_selection_result =
        choose_anchor_vertex(mu, simplex, vertex_cache, tol);
    if (anchor_selection_result.early_certificate.has_value()) {
        return *anchor_selection_result.early_certificate;
    }
    const auto anchor_selection = anchor_selection_result.selection;

    const auto &anchor = vertex_cache.get(anchor_selection.vertex_id);
    auto split_result = split_anchor_spectrum(anchor, mu, tol, anchor_selection.ndof);
    if (split_result.early_certificate.has_value()) {
        return *split_result.early_certificate;
    }
    const auto &split = split_result.split;

    const auto nocc = anchor_selection.nocc;
    const auto ndof = anchor_selection.ndof;
    const auto nunocc = ndof - nocc;
    const auto anchor_vectors =
        std::span<const Complex>(anchor.eigenvectors.data(), anchor.eigenvectors.size());
    const auto blocks = build_simplex_blocks(
        mu,
        simplex,
        vertex_cache,
        anchor_vectors,
        ndof,
        nocc
    );
    const auto rotation = perturbative_rotation(
        std::span<const Complex>(blocks.average_coupling.data(), blocks.average_coupling.size()),
        std::span<const double>(split.unoccupied_gaps.data(), split.unoccupied_gaps.size()),
        std::span<const double>(split.occupied_gaps.data(), split.occupied_gaps.size())
    );

    const auto rotation_span = std::span<const Complex>(rotation.data(), rotation.size());
    const auto occupied_rotation = adjoint_rectangular_copy(rotation_span, nunocc, nocc);
    const auto occupied_rotation_span =
        std::span<const Complex>(occupied_rotation.data(), occupied_rotation.size());
    const auto unoccupied_gram_row_bounds =
        frame_gram_row_bounds(rotation_span, nunocc, nocc);
    const auto occupied_gram_row_bounds =
        frame_gram_row_bounds(occupied_rotation_span, nocc, nunocc);
    auto unoccupied_mu_radius = std::numeric_limits<double>::infinity();
    auto occupied_mu_radius = std::numeric_limits<double>::infinity();

    const auto unoccupied_size = sector_size(OccupationSector::Unoccupied, ndof, nocc);
    const auto unoccupied_opposite_size =
        opposite_sector_size(OccupationSector::Unoccupied, ndof, nocc);
    const auto occupied_size = sector_size(OccupationSector::Occupied, ndof, nocc);
    const auto occupied_opposite_size =
        opposite_sector_size(OccupationSector::Occupied, ndof, nocc);

    for (const auto &vertex_blocks : blocks.vertices) {
        const auto unoccupied = check_occupation_sector(
            OccupationSector::Unoccupied,
            vertex_blocks,
            rotation_span,
            std::span<const double>(
                unoccupied_gram_row_bounds.data(),
                unoccupied_gram_row_bounds.size()
            ),
            unoccupied_size,
            unoccupied_opposite_size,
            margin,
            tol
        );
        unoccupied_mu_radius = std::min(unoccupied_mu_radius, unoccupied.mu_radius);
        if (!unoccupied.passed) {
            return occupation_bounded_inconclusive(
                mu,
                ndof,
                nocc,
                blocks.vertices,
                tol,
                estimate_occupation_bounds
            );
        }

        const auto occupied = check_occupation_sector(
            OccupationSector::Occupied,
            vertex_blocks,
            occupied_rotation_span,
            std::span<const double>(
                occupied_gram_row_bounds.data(),
                occupied_gram_row_bounds.size()
            ),
            occupied_size,
            occupied_opposite_size,
            margin,
            tol
        );
        occupied_mu_radius = std::min(occupied_mu_radius, occupied.mu_radius);
        if (!occupied.passed) {
            return occupation_bounded_inconclusive(
                mu,
                ndof,
                nocc,
                blocks.vertices,
                tol,
                estimate_occupation_bounds
            );
        }
    }

    auto result = certificate(
        SimplexCertificateStatus::CertifiedGapped,
        exact_occupation(nocc)
    );
    result.mu_interval = MuInterval{
        .lower = mu - occupied_mu_radius,
        .upper = mu + unoccupied_mu_radius,
    };
    return result;
}

}  // namespace lineartetrahedron::simplex_certificate
