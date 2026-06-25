#pragma once

#include <complex>
#include <cstddef>
#include <vector>

namespace lineartetrahedron {

void diagonalize_hermitian_in_place(
    std::vector<std::complex<double>> &matrix,
    std::vector<double> &eigenvalues,
    size_t size,
    bool compute_vectors,
    const char *context
);

}  // namespace lineartetrahedron
