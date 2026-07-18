#include "linalg/blas_lapack.h"

#include <limits>
#include <stdexcept>

#ifndef LINEARTETRAHEDRON_BLAS_ZGEMM
#define LINEARTETRAHEDRON_BLAS_ZGEMM zgemm_
#endif

#ifndef LINEARTETRAHEDRON_BLAS_ZHER2K
#define LINEARTETRAHEDRON_BLAS_ZHER2K zher2k_
#endif

#ifndef LINEARTETRAHEDRON_BLAS_ZHEMM
#define LINEARTETRAHEDRON_BLAS_ZHEMM zhemm_
#endif

extern "C" {
void LINEARTETRAHEDRON_BLAS_ZGEMM(
    const char *transa,
    const char *transb,
    const int *m,
    const int *n,
    const int *k,
    const std::complex<double> *alpha,
    const std::complex<double> *a,
    const int *lda,
    const std::complex<double> *b,
    const int *ldb,
    const std::complex<double> *beta,
    std::complex<double> *c,
    const int *ldc
);

void LINEARTETRAHEDRON_BLAS_ZHER2K(
    const char *uplo,
    const char *trans,
    const int *n,
    const int *k,
    const std::complex<double> *alpha,
    const std::complex<double> *a,
    const int *lda,
    const std::complex<double> *b,
    const int *ldb,
    const double *beta,
    std::complex<double> *c,
    const int *ldc
);

void LINEARTETRAHEDRON_BLAS_ZHEMM(
    const char *side,
    const char *uplo,
    const int *m,
    const int *n,
    const std::complex<double> *alpha,
    const std::complex<double> *a,
    const int *lda,
    const std::complex<double> *b,
    const int *ldb,
    const std::complex<double> *beta,
    std::complex<double> *c,
    const int *ldc
);
}

namespace lineartetrahedron::linalg {
namespace {

int blas_dimension(size_t value) {
    if (value > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("BLAS dimension exceeds the LP64 range");
    }
    return static_cast<int>(value);
}

void scale_matrix(
    Complex *matrix,
    size_t rows,
    size_t columns,
    size_t leading_dimension,
    Complex scale
) {
    for (size_t column = 0; column < columns; ++column) {
        for (size_t row = 0; row < rows; ++row) {
            matrix[row + column * leading_dimension] *= scale;
        }
    }
}

}  // namespace

void matrix_multiply(
    char left_operation,
    char right_operation,
    size_t rows,
    size_t columns,
    size_t inner_dimension,
    Complex alpha,
    const Complex *left,
    size_t left_leading_dimension,
    const Complex *right,
    size_t right_leading_dimension,
    Complex beta,
    Complex *result,
    size_t result_leading_dimension
) {
    if (rows == 0 || columns == 0) {
        return;
    }
    if (inner_dimension == 0) {
        scale_matrix(result, rows, columns, result_leading_dimension, beta);
        return;
    }

    const auto m = blas_dimension(rows);
    const auto n = blas_dimension(columns);
    const auto k = blas_dimension(inner_dimension);
    const auto lda = blas_dimension(left_leading_dimension);
    const auto ldb = blas_dimension(right_leading_dimension);
    const auto ldc = blas_dimension(result_leading_dimension);
    LINEARTETRAHEDRON_BLAS_ZGEMM(
        &left_operation,
        &right_operation,
        &m,
        &n,
        &k,
        &alpha,
        left,
        &lda,
        right,
        &ldb,
        &beta,
        result,
        &ldc
    );
}

void hermitian_rank_2k_update(
    char operation,
    size_t size,
    size_t inner_dimension,
    Complex alpha,
    const Complex *left,
    size_t left_leading_dimension,
    const Complex *right,
    size_t right_leading_dimension,
    double beta,
    Complex *result,
    size_t result_leading_dimension
) {
    if (size == 0) {
        return;
    }

    const char lower_triangle = 'L';
    const auto n = blas_dimension(size);
    const auto k = blas_dimension(inner_dimension);
    const auto lda = blas_dimension(left_leading_dimension);
    const auto ldb = blas_dimension(right_leading_dimension);
    const auto ldc = blas_dimension(result_leading_dimension);
    LINEARTETRAHEDRON_BLAS_ZHER2K(
        &lower_triangle,
        &operation,
        &n,
        &k,
        &alpha,
        left,
        &lda,
        right,
        &ldb,
        &beta,
        result,
        &ldc
    );
}

void hermitian_matrix_multiply(
    char side,
    size_t rows,
    size_t columns,
    Complex alpha,
    const Complex *hermitian,
    size_t hermitian_leading_dimension,
    const Complex *other,
    size_t other_leading_dimension,
    Complex beta,
    Complex *result,
    size_t result_leading_dimension
) {
    if (rows == 0 || columns == 0) {
        return;
    }

    const char lower_triangle = 'L';
    const auto m = blas_dimension(rows);
    const auto n = blas_dimension(columns);
    const auto lda = blas_dimension(hermitian_leading_dimension);
    const auto ldb = blas_dimension(other_leading_dimension);
    const auto ldc = blas_dimension(result_leading_dimension);
    LINEARTETRAHEDRON_BLAS_ZHEMM(
        &side,
        &lower_triangle,
        &m,
        &n,
        &alpha,
        hermitian,
        &lda,
        other,
        &ldb,
        &beta,
        result,
        &ldc
    );
}

}  // namespace lineartetrahedron::linalg
