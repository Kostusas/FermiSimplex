#pragma once

#include "certificate/linalg/matrix.h"

#include <cstddef>

namespace lineartetrahedron::simplex_certificate::detail {

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

int potrf(Complex *matrix, size_t size);

}  // namespace lineartetrahedron::simplex_certificate::detail
