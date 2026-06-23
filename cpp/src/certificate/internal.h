#pragma once

#include "lineartetrahedron/vertex_spectra.h"

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

struct VertexBlocks {
    std::vector<Complex> positive;
    std::vector<Complex> negative;
    std::vector<Complex> mixed;
};

void gemm(
    char transa,
    char transb,
    size_t m,
    size_t n,
    size_t k,
    Complex alpha,
    const Complex *a,
    size_t lda,
    const Complex *b,
    size_t ldb,
    Complex beta,
    Complex *c,
    size_t ldc
);

void her2k(
    char trans,
    size_t n,
    size_t k,
    Complex alpha,
    const Complex *a,
    size_t lda,
    const Complex *b,
    size_t ldb,
    double beta,
    Complex *c,
    size_t ldc
);

void hemm(
    char side,
    size_t m,
    size_t n,
    Complex alpha,
    const Complex *a,
    size_t lda,
    const Complex *b,
    size_t ldb,
    Complex beta,
    Complex *c,
    size_t ldc
);

bool positive_definite(std::vector<Complex> block, size_t size, double tol);

void negate_in_place(std::vector<Complex> &matrix);

VertexBlocks build_vertex_blocks(
    std::span<const Complex> anchor_vectors,
    size_t ndof,
    size_t npos,
    size_t nneg,
    const VertexSpectra &target,
    double mu
);

std::vector<Complex> rotated_positive_block(
    const VertexBlocks &blocks,
    std::span<const Complex> rotation,
    size_t npos,
    size_t nneg
);

std::vector<Complex> rotated_negative_block(
    const VertexBlocks &blocks,
    std::span<const Complex> rotation,
    size_t npos,
    size_t nneg
);

}  // namespace lineartetrahedron::simplex_certificate::detail
