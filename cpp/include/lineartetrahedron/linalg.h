#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace lineartetrahedron {

struct LanczosStats {
    std::uint64_t calls = 0;
    std::uint64_t iterations = 0;
    std::uint64_t ritz_checks = 0;
    std::uint64_t converged_by_uncertainty = 0;
    std::uint64_t converged_at_full_dimension = 0;
    std::uint64_t converged_by_zero_residual = 0;
    std::uint64_t lanczos_nanoseconds = 0;
    std::uint64_t derivative_calls = 0;
    std::uint64_t derivative_assembly_nanoseconds = 0;
    std::uint64_t derivative_total_nanoseconds = 0;
};

void reset_lanczos_stats();

LanczosStats lanczos_stats();

void record_derivative_spectral_norm_timing(
    std::uint64_t assembly_nanoseconds,
    std::uint64_t total_nanoseconds
);

double hermitian_spectral_norm_lanczos(
    std::span<const std::complex<double>> matrix,
    size_t size,
    double absolute_uncertainty
);

double hermitian_min_eigenvalue_lanczos(
    std::span<const std::complex<double>> matrix,
    size_t size,
    double absolute_uncertainty
);

double matrix_spectral_norm_lanczos(
    std::span<const std::complex<double>> matrix,
    size_t size,
    double absolute_uncertainty
);

}  // namespace lineartetrahedron
