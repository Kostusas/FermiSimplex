#pragma once

#include <complex>
#include <cstddef>
#include <span>
#include <vector>

namespace lineartetrahedron {

double hermitian_spectral_norm_lanczos(
    std::span<const std::complex<double>> matrix,
    size_t size
);

double matrix_spectral_norm_lanczos(
    std::span<const std::complex<double>> matrix,
    size_t size
);

}  // namespace lineartetrahedron
