#pragma once

#include <complex>
#include <cstddef>
#include <optional>
#include <span>

namespace fermisimplex::certification {

enum class SimplexCertificateStatus {
    CertifiedGapped,
    VisibleGapless,
    Inconclusive,
};

constexpr double kDefaultTolerance = 1e-14;

struct OccupationBounds {
    // Every point in the simplex has between lower and upper occupied states.
    std::size_t lower = 0;
    std::size_t upper = 0;
};

struct MuInterval {
    // occupation_bounds remain valid for every mu in [lower, upper]. The
    // certificate status can change within this interval.
    double lower = 0.0;
    double upper = 0.0;
};

struct SimplexCertificate {
    SimplexCertificateStatus status = SimplexCertificateStatus::Inconclusive;
    OccupationBounds occupation_bounds;
    std::optional<MuInterval> mu_interval;
};

inline bool occupation_bounds_valid_at(
    const SimplexCertificate &certificate,
    double mu
) {
    return certificate.mu_interval.has_value() &&
           certificate.mu_interval->lower <= mu &&
           mu <= certificate.mu_interval->upper;
}

inline std::size_t occupation_width(const SimplexCertificate &certificate) {
    return certificate.occupation_bounds.upper - certificate.occupation_bounds.lower;
}

SimplexCertificate certify_simplex(
    // One sorted spectrum and one column-major orthonormal eigenbasis per vertex.
    std::span<const std::span<const double>> eigenvalues,
    std::span<const std::span<const std::complex<double>>> eigenvectors,
    double mu,
    // Uniform spectral/operator-norm bound on the difference between the
    // actual Hamiltonian and its vertex-linear interpolant over this simplex.
    double linearization_error_bound,
    double tolerance = kDefaultTolerance
);

}  // namespace fermisimplex::certification
