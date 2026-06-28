#pragma once

#include "core/vertex_spectra.h"

#include <adaptivesimplex/core/geometry.h>
#include <adaptivesimplex/core/types.h>
#include <adaptivesimplex/core/vertex_cache.h>

#include <cstddef>
#include <limits>
#include <span>
#include <complex>

namespace lineartetrahedron::simplex_certificate {

namespace core = adaptivesimplex::core;

enum class SimplexCertificateStatus {
    CertifiedGapped,
    VisibleGapless,
    Inconclusive,
};

constexpr double kDefaultTolerance = 1e-14;

struct OccupationBounds {
    size_t lower = 0;
    size_t upper = 0;
};

struct MuInterval {
    // Empty by default: no reusable chemical-potential certification range.
    double lower = std::numeric_limits<double>::infinity();
    double upper = -std::numeric_limits<double>::infinity();
};

struct SimplexCertificate {
    SimplexCertificateStatus status = SimplexCertificateStatus::Inconclusive;
    OccupationBounds occupation_bounds;
    MuInterval mu_interval;
    double energy_bound = 0.0;
};

inline bool has_mu_interval(const SimplexCertificate &certificate) {
    return certificate.mu_interval.lower <= certificate.mu_interval.upper;
}

inline bool reusable_at(const SimplexCertificate &certificate, double mu) {
    return certificate.mu_interval.lower <= mu && mu <= certificate.mu_interval.upper;
}

inline size_t occupation_width(const SimplexCertificate &certificate) {
    return certificate.occupation_bounds.upper - certificate.occupation_bounds.lower;
}

SimplexCertificate certify_simplex(
    std::span<const std::span<const double>> eigenvalues,
    std::span<const std::span<const std::complex<double>>> eigenvectors,
    double mu = 0.0,
    double margin = 0.0,
    double tol = kDefaultTolerance,
    bool estimate_occupation_bounds = false
);

SimplexCertificate certify_mesh_simplex(
    const core::Geometry &geometry,
    core::SimplexId simplex_id,
    const core::VertexCache<VertexSpectra> &vertex_cache,
    double mu = 0.0,
    double hessian_bound = 0.0,
    double anharmonicity_bound = 0.0,
    double tol = kDefaultTolerance,
    bool estimate_occupation_bounds = false
);

SimplexCertificate certify_mesh_simplex_with_energy_bound(
    const core::Geometry &geometry,
    core::SimplexId simplex_id,
    const core::VertexCache<VertexSpectra> &vertex_cache,
    double mu = 0.0,
    double energy_bound = 0.0,
    double tol = kDefaultTolerance,
    bool estimate_occupation_bounds = false
);

}  // namespace lineartetrahedron::simplex_certificate
