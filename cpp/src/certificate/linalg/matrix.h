#pragma once

#include "core/vertex_spectra.h"

#include <complex>
#include <cstddef>
#include <span>
#include <vector>

namespace lineartetrahedron::simplex_certificate::detail {

using Complex = std::complex<double>;

inline size_t column_major_index(size_t row, size_t column, size_t rows) {
    return row + column * rows;
}

inline double signed_eigenvalue(const VertexSpectra &spectra, size_t band, double mu) {
    return spectra.eigenvalues[band] - mu;
}

inline std::vector<Complex> adjoint_rectangular_copy(
    std::span<const Complex> matrix,
    size_t rows,
    size_t columns
) {
    std::vector<Complex> result(columns * rows, Complex{0.0, 0.0});
    for (size_t column = 0; column < columns; ++column) {
        for (size_t row = 0; row < rows; ++row) {
            result[column_major_index(column, row, columns)] =
                std::conj(matrix[column_major_index(row, column, rows)]);
        }
    }
    return result;
}

}  // namespace lineartetrahedron::simplex_certificate::detail
