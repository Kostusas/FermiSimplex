#include "certification/bounds/mu_bounds.h"
#include "certification/linalg/cholesky.h"
#include "certification/mesh_certificate.h"
#include "certification/simplex/anchor_selection.h"
#include "certification/simplex/occupation_certificate.h"
#include "certification/simplex/simplex_blocks.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
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

SimplexCertificate certify_scalar_simplex(
    const detail::VertexSpectraAnalysis &analysis,
    std::span<const double> values,
    double mu,
    double linearization_error_bound,
    double tolerance
) {
    const auto nocc = analysis.anchor.nocc;
    const auto orientation = nocc == 0 ? 1.0 : -1.0;
    auto worst_oriented_value = std::numeric_limits<double>::infinity();
    for (const auto value : values) {
        worst_oriented_value = std::min(
            worst_oriented_value,
            orientation * (value - mu)
        );
    }

    const auto total_margin =
        detail::certificate_margin(tolerance) +
        linearization_error_bound;
    const auto sector_radius = std::max(
        0.0,
        worst_oriented_value - total_margin
    );
    const auto sector_passed = worst_oriented_value > total_margin;
    const auto certified =
        analysis.classification ==
            detail::VertexSpectraClassification::NeedsCertificate &&
        sector_passed;

    auto result = SimplexCertificate{
        .status = certified
            ? SimplexCertificateStatus::CertifiedGapped
            : analysis.classification ==
                    detail::VertexSpectraClassification::VisibleGapless
                ? SimplexCertificateStatus::VisibleGapless
                : SimplexCertificateStatus::Inconclusive,
        .occupation_bounds = certified
            ? exact_occupation(nocc)
            : OccupationBounds{.lower = 0, .upper = 1},
    };
    result.mu_interval = nocc == 0
        ? MuInterval{
            .lower = -std::numeric_limits<double>::infinity(),
            .upper = mu + sector_radius,
        }
        : MuInterval{
            .lower = mu - sector_radius,
            .upper = std::numeric_limits<double>::infinity(),
        };
    return result;
}

}  // namespace

struct PreparedSimplexCertificate::Impl {
    double mu = 0.0;
    double tolerance = 0.0;
    detail::VertexSpectraAnalysis analysis;
    std::size_t nocc = 0;
    std::size_t nunocc = 0;
    detail::SimplexBlocks blocks;
    std::vector<detail::Complex> rotation;
    std::vector<detail::Complex> occupied_rotation;
    std::vector<double> unoccupied_gram_row_bounds;
    std::vector<double> occupied_gram_row_bounds;
    std::vector<double> scalar_values;

    Impl(
        std::span<const std::span<const double>> eigenvalues,
        std::span<const std::span<const std::complex<double>>> eigenvectors,
        double mu_,
        double tolerance_
    ) : mu(mu_), tolerance(tolerance_) {
        using namespace detail;

        if (!std::isfinite(mu)) {
            throw std::runtime_error("certify_simplex: mu must be finite");
        }
        if (tolerance < 0.0 || !std::isfinite(tolerance)) {
            throw std::runtime_error(
                "certify_simplex: tolerance must be finite and non-negative"
            );
        }
        validate_simplex_structure(eigenvalues, eigenvectors);

        analysis = analyze_vertex_spectra(mu, eigenvalues, tolerance);
        const auto &anchor = analysis.anchor;
        const auto anchor_eigenvalues = eigenvalues[anchor.vertex_index];
        const auto anchor_eigenvectors = eigenvectors[anchor.vertex_index];
        nocc = anchor.nocc;
        const auto ndof = anchor_eigenvalues.size();
        if (ndof == 1) {
            scalar_values.reserve(eigenvalues.size());
            for (const auto values : eigenvalues) {
                scalar_values.push_back(values.front());
            }
            return;
        }
        nunocc = ndof - nocc;
        blocks = build_simplex_blocks(
            mu,
            eigenvalues,
            eigenvectors,
            anchor_eigenvectors,
            ndof,
            nocc,
            anchor.vertex_index
        );

        if (analysis.classification ==
            VertexSpectraClassification::VisibleGapless) {
            return;
        }

        const auto split = split_anchor_spectrum(
            anchor_eigenvalues, mu, nocc
        );
        rotation = perturbative_rotation(
            blocks.average_coupling,
            split.unoccupied_gaps,
            split.occupied_gaps
        );
        occupied_rotation = adjoint_rectangular_copy(
            rotation, nunocc, nocc
        );
        unoccupied_gram_row_bounds =
            frame_gram_row_bounds(rotation, nunocc, nocc);
        occupied_gram_row_bounds =
            frame_gram_row_bounds(occupied_rotation, nocc, nunocc);
    }

    SimplexCertificate certify(double linearization_error_bound) const {
        using namespace detail;

        if (linearization_error_bound < 0.0 ||
            !std::isfinite(linearization_error_bound)) {
            throw std::runtime_error(
                "certify_simplex: linearization_error_bound must be finite and non-negative"
            );
        }
        if (!scalar_values.empty()) {
            return certify_scalar_simplex(
                analysis,
                scalar_values,
                mu,
                linearization_error_bound,
                tolerance
            );
        }
        if (analysis.classification ==
            VertexSpectraClassification::VisibleGapless) {
            return make_unresolved_certificate(
                SimplexCertificateStatus::VisibleGapless,
                mu,
                blocks.vertices,
                linearization_error_bound,
                tolerance
            );
        }

        const auto rotation_span = std::span<const Complex>{rotation};
        const auto occupied_rotation_span =
            std::span<const Complex>{occupied_rotation};
        auto unoccupied_mu_radius =
            std::numeric_limits<double>::infinity();
        auto occupied_mu_radius =
            std::numeric_limits<double>::infinity();

        for (const auto &vertex_blocks : blocks.vertices) {
            const auto unoccupied = check_unoccupied_sector(
                vertex_blocks,
                rotation_span,
                std::span<const double>(unoccupied_gram_row_bounds),
                linearization_error_bound,
                tolerance
            );
            const auto occupied = check_occupied_sector(
                vertex_blocks,
                occupied_rotation_span,
                std::span<const double>(occupied_gram_row_bounds),
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
            unoccupied_mu_radius =
                std::min(unoccupied_mu_radius, unoccupied.mu_radius);
            occupied_mu_radius =
                std::min(occupied_mu_radius, occupied.mu_radius);
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

    OccupationBounds occupation_bounds(
        double linearization_error_bound
    ) const {
        using namespace detail;
        if (linearization_error_bound < 0.0 ||
            !std::isfinite(linearization_error_bound)) {
            throw std::runtime_error(
                "certify_simplex: linearization_error_bound must be finite and non-negative"
            );
        }
        if (!scalar_values.empty()) {
            return certify_scalar_simplex(
                analysis,
                scalar_values,
                mu,
                linearization_error_bound,
                tolerance
            ).occupation_bounds;
        }
        if (analysis.classification ==
            VertexSpectraClassification::VisibleGapless) {
            return make_unresolved_occupation_bounds(
                blocks.vertices, linearization_error_bound, tolerance
            );
        }
        const auto rotation_span = std::span<const Complex>{rotation};
        const auto occupied_rotation_span =
            std::span<const Complex>{occupied_rotation};
        for (const auto &vertex_blocks : blocks.vertices) {
            if (!unoccupied_sector_passes(
                    vertex_blocks,
                    rotation_span,
                    linearization_error_bound,
                    tolerance
                ) ||
                !occupied_sector_passes(
                    vertex_blocks,
                    occupied_rotation_span,
                    linearization_error_bound,
                    tolerance
                )) {
                return make_unresolved_occupation_bounds(
                    blocks.vertices, linearization_error_bound, tolerance
                );
            }
        }
        return exact_occupation(nocc);
    }
};

PreparedSimplexCertificate::PreparedSimplexCertificate(
    std::shared_ptr<const Impl> impl
) : impl_(std::move(impl)) {
    if (impl_ == nullptr) {
        throw std::invalid_argument(
            "prepared simplex certificate implementation must not be null"
        );
    }
}

SimplexCertificate PreparedSimplexCertificate::certify(
    double linearization_error_bound
) const {
    return impl_->certify(linearization_error_bound);
}

OccupationBounds PreparedSimplexCertificate::occupation_bounds(
    double linearization_error_bound
) const {
    return impl_->occupation_bounds(linearization_error_bound);
}

SimplexCertificate certify_simplex(
    std::span<const std::span<const double>> eigenvalues,
    std::span<const std::span<const std::complex<double>>> eigenvectors,
    double mu,
    double linearization_error_bound,
    double tolerance
) {
    return prepare_simplex_certificate(
        eigenvalues, eigenvectors, mu, tolerance
    ).certify(linearization_error_bound);
}

PreparedSimplexCertificate prepare_simplex_certificate(
    std::span<const std::span<const double>> eigenvalues,
    std::span<const std::span<const std::complex<double>>> eigenvectors,
    double mu,
    double tolerance
) {
    return PreparedSimplexCertificate{
        std::make_shared<PreparedSimplexCertificate::Impl>(
            eigenvalues, eigenvectors, mu, tolerance
        )
    };
}

PreparedSimplexCertificate prepare_mesh_simplex_certificate(
    const SpectralMesh &mesh,
    core::SimplexId simplex_id,
    double mu,
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
    return PreparedSimplexCertificate{
        std::make_shared<PreparedSimplexCertificate::Impl>(
            std::span<const std::span<const double>>{eigenvalues},
            std::span<const std::span<const std::complex<double>>>{eigenvectors},
            mu,
            tolerance
        )
    };
}

SimplexCertificate certify_mesh_simplex(
    const SpectralMesh &mesh,
    core::SimplexId simplex_id,
    double mu,
    double linearization_error_bound,
    double tolerance
) {
    return prepare_mesh_simplex_certificate(
        mesh, simplex_id, mu, tolerance
    ).certify(linearization_error_bound);
}

}  // namespace fermisimplex::certification
