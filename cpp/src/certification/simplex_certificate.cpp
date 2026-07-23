#include "certification/bounds/mu_bounds.h"
#include "certification/mesh_certificate.h"
#include "certification/simplex/anchor_selection.h"
#include "certification/simplex/occupation_certificate.h"
#include "certification/simplex/simplex_blocks.h"

#include <algorithm>
#include <complex>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace fermisimplex::certification {
namespace core = adaptivesimplex::core;
namespace {

OccupationBounds exact_occupation(size_t occupation) {
    return OccupationBounds{.lower = occupation, .upper = occupation};
}

void validate_simplex_structure(
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
    double linearization_error_bound,
    double tolerance
) {
    using namespace detail;

    if (!std::isfinite(mu)) {
        throw std::runtime_error("certify_simplex: mu must be finite");
    }
    if (linearization_error_bound < 0.0 || !std::isfinite(linearization_error_bound)) {
        throw std::runtime_error(
            "certify_simplex: linearization_error_bound must be finite and non-negative"
        );
    }
    if (tolerance < 0.0 || !std::isfinite(tolerance)) {
        throw std::runtime_error(
            "certify_simplex: tolerance must be finite and non-negative"
        );
    }
    validate_simplex_structure(eigenvalues, eigenvectors);

    const auto analysis = analyze_vertex_spectra(mu, eigenvalues, tolerance);
    const auto &anchor_selection = analysis.anchor;

    const auto anchor_eigenvalues = eigenvalues[anchor_selection.vertex_index];
    const auto anchor_eigenvectors = eigenvectors[anchor_selection.vertex_index];
    const auto nocc = anchor_selection.nocc;
    const auto ndof = anchor_eigenvalues.size();
    const auto nunocc = ndof - nocc;
    const auto blocks = build_simplex_blocks(
        mu,
        eigenvalues,
        eigenvectors,
        anchor_eigenvectors,
        ndof,
        nocc,
        anchor_selection.vertex_index
    );

    if (analysis.classification == VertexSpectraClassification::VisibleGapless) {
        return make_unresolved_certificate(
            SimplexCertificateStatus::VisibleGapless,
            mu,
            blocks.vertices,
            linearization_error_bound,
            tolerance
        );
    }

    const auto split = split_anchor_spectrum(anchor_eigenvalues, mu, nocc);
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

    for (const auto &vertex_blocks : blocks.vertices) {
        const auto unoccupied = check_unoccupied_sector(
            vertex_blocks,
            rotation_span,
            std::span<const double>(
                unoccupied_gram_row_bounds.data(),
                unoccupied_gram_row_bounds.size()
            ),
            linearization_error_bound,
            tolerance
        );
        const auto occupied = check_occupied_sector(
            vertex_blocks,
            occupied_rotation_span,
            std::span<const double>(
                occupied_gram_row_bounds.data(),
                occupied_gram_row_bounds.size()
            ),
            linearization_error_bound,
            tolerance
        );

        if (!unoccupied.passed || !occupied.passed) {
            return make_unresolved_certificate(
                SimplexCertificateStatus::Inconclusive,
                mu,
                blocks.vertices,
                linearization_error_bound,
                tolerance
            );
        }
        unoccupied_mu_radius = std::min(unoccupied_mu_radius, unoccupied.mu_radius);
        occupied_mu_radius = std::min(occupied_mu_radius, occupied.mu_radius);
    }

    auto result = SimplexCertificate{
        .status = SimplexCertificateStatus::CertifiedGapped,
        .occupation_bounds = exact_occupation(nocc),
    };
    // The occupied-sector radius limits lowering mu; the unoccupied-sector
    // radius limits raising it. These differ for asymmetric certificates.
    result.mu_interval = MuInterval{
        .lower = mu - occupied_mu_radius,
        .upper = mu + unoccupied_mu_radius,
    };
    return result;
}

SimplexCertificate certify_mesh_simplex(
    const SpectralMesh &mesh,
    core::SimplexId simplex_id,
    double mu,
    double linearization_error_bound,
    double tolerance
) {
    const auto &geometry = mesh.geometry();
    const auto &simplex = geometry.simplices().simplex(simplex_id);
    if (simplex.vertex_ids.empty()) {
        throw std::runtime_error("certify_mesh_simplex: simplex must not be empty");
    }
    std::vector<std::span<const double>> eigenvalues;
    std::vector<std::span<const std::complex<double>>> eigenvectors;
    eigenvalues.reserve(simplex.vertex_ids.size());
    eigenvectors.reserve(simplex.vertex_ids.size());
    for (const auto vertex_id : simplex.vertex_ids) {
        const auto &spectra = mesh.eigensystems().get(vertex_id);
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
        linearization_error_bound,
        tolerance
    );
}

}  // namespace fermisimplex::certification
