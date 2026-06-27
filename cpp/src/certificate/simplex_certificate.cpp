#include "certificate/simplex_certificate.h"

#include "certificate/simplex/anchor_selection.h"
#include "certificate/bounds/mu_bounds.h"
#include "certificate/simplex/occupation_certificate.h"
#include "certificate/simplex/simplex_blocks.h"

#include <algorithm>
#include <complex>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

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

void validate_simplex_spectra(
    std::span<const std::span<const double>> eigenvalues,
    std::span<const std::span<const std::complex<double>>> eigenvectors
) {
    if (eigenvalues.empty()) {
        throw std::runtime_error("certify_simplex: simplex must not be empty");
    }
    if (eigenvalues.size() != eigenvectors.size()) {
        throw std::runtime_error("certify_simplex: eigenvalue/eigenvector vertex counts differ");
    }

    const auto ndof = eigenvalues.front().size();
    if (ndof == 0) {
        throw std::runtime_error("certify_simplex: vertex spectra must not be empty");
    }
    for (size_t vertex = 0; vertex < eigenvalues.size(); ++vertex) {
        if (eigenvalues[vertex].size() != ndof) {
            throw std::runtime_error("certify_simplex: all vertices must have the same ndof");
        }
        if (eigenvectors[vertex].size() != ndof * ndof) {
            throw std::runtime_error(
                "certify_simplex: each eigenvector block must have size ndof * ndof"
            );
        }
    }
}

}  // namespace

SimplexCertificate certify_simplex(
    std::span<const std::span<const double>> eigenvalues,
    std::span<const std::span<const std::complex<double>>> eigenvectors,
    double mu,
    double margin,
    double tol,
    bool estimate_occupation_bounds
) {
    using namespace detail;

    validate_simplex_spectra(eigenvalues, eigenvectors);

    const auto anchor_selection_result =
        choose_anchor_spectrum(mu, eigenvalues, tol);
    if (anchor_selection_result.early_certificate.has_value()) {
        return *anchor_selection_result.early_certificate;
    }
    const auto anchor_selection = anchor_selection_result.selection;

    const auto anchor_eigenvalues = eigenvalues[anchor_selection.vertex_index];
    const auto anchor_eigenvectors = eigenvectors[anchor_selection.vertex_index];
    auto split_result = split_anchor_spectrum(anchor_eigenvalues, mu, tol, anchor_selection.ndof);
    if (split_result.early_certificate.has_value()) {
        return *split_result.early_certificate;
    }
    const auto &split = split_result.split;

    const auto nocc = anchor_selection.nocc;
    const auto ndof = anchor_selection.ndof;
    const auto nunocc = ndof - nocc;
    const auto blocks = build_simplex_blocks(
        mu,
        eigenvalues,
        eigenvectors,
        anchor_eigenvectors,
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

SimplexCertificate certify_mesh_simplex(
    const core::Geometry &geometry,
    core::SimplexId simplex_id,
    const core::VertexCache<VertexSpectra> &vertex_cache,
    double mu,
    double margin,
    double tol,
    bool estimate_occupation_bounds
) {
    const auto &simplex = geometry.simplices().simplex(simplex_id);
    if (simplex.vertex_ids.empty()) {
        throw std::runtime_error("certify_mesh_simplex: simplex must not be empty");
    }

    std::vector<std::span<const double>> eigenvalues;
    std::vector<std::span<const std::complex<double>>> eigenvectors;
    eigenvalues.reserve(simplex.vertex_ids.size());
    eigenvectors.reserve(simplex.vertex_ids.size());
    for (const auto vertex_id : simplex.vertex_ids) {
        const auto &spectra = vertex_cache.get(vertex_id);
        eigenvalues.push_back(
            std::span<const double>(spectra.eigenvalues.data(), spectra.eigenvalues.size())
        );
        eigenvectors.push_back(
            std::span<const std::complex<double>>(
                spectra.eigenvectors.data(),
                spectra.eigenvectors.size()
            )
        );
    }
    return certify_simplex(
        std::span<const std::span<const double>>(eigenvalues.data(), eigenvalues.size()),
        std::span<const std::span<const std::complex<double>>>(
            eigenvectors.data(),
            eigenvectors.size()
        ),
        mu,
        margin,
        tol,
        estimate_occupation_bounds
    );
}

}  // namespace lineartetrahedron::simplex_certificate
