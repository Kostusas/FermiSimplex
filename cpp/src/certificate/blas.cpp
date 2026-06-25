#include "internal.h"

#include <algorithm>
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

#ifndef LINEARTETRAHEDRON_LAPACK_ZPOTRF
#define LINEARTETRAHEDRON_LAPACK_ZPOTRF zpotrf_
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

void LINEARTETRAHEDRON_LAPACK_ZPOTRF(
    const char *uplo,
    const int *n,
    std::complex<double> *a,
    const int *lda,
    int *info
);
}

namespace lineartetrahedron::simplex_certificate::detail {
namespace {

constexpr double kBlockMargin = 1e-10;

void check_lp64(size_t value) {
    if (value > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("simplex certificate: BLAS dimension exceeds LP64 range");
    }
}

}  // namespace

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
) {
    if (m == 0 || n == 0) {
        return;
    }
    if (k == 0) {
        std::fill(c, c + ldc * n, Complex{0.0, 0.0});
        return;
    }
    check_lp64(m);
    check_lp64(n);
    check_lp64(k);
    check_lp64(lda);
    check_lp64(ldb);
    check_lp64(ldc);

    const int im = static_cast<int>(m);
    const int in = static_cast<int>(n);
    const int ik = static_cast<int>(k);
    const int ilda = static_cast<int>(lda);
    const int ildb = static_cast<int>(ldb);
    const int ildc = static_cast<int>(ldc);
    LINEARTETRAHEDRON_BLAS_ZGEMM(
        &transa,
        &transb,
        &im,
        &in,
        &ik,
        &alpha,
        a,
        &ilda,
        b,
        &ildb,
        &beta,
        c,
        &ildc
    );
}

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
) {
    if (n == 0) {
        return;
    }
    check_lp64(n);
    check_lp64(k);
    check_lp64(lda);
    check_lp64(ldb);
    check_lp64(ldc);

    const char uplo = 'L';
    const int in = static_cast<int>(n);
    const int ik = static_cast<int>(k);
    const int ilda = static_cast<int>(lda);
    const int ildb = static_cast<int>(ldb);
    const int ildc = static_cast<int>(ldc);
    LINEARTETRAHEDRON_BLAS_ZHER2K(
        &uplo,
        &trans,
        &in,
        &ik,
        &alpha,
        a,
        &ilda,
        b,
        &ildb,
        &beta,
        c,
        &ildc
    );
}

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
) {
    if (m == 0 || n == 0) {
        return;
    }
    check_lp64(m);
    check_lp64(n);
    check_lp64(lda);
    check_lp64(ldb);
    check_lp64(ldc);

    const char uplo = 'L';
    const int im = static_cast<int>(m);
    const int in = static_cast<int>(n);
    const int ilda = static_cast<int>(lda);
    const int ildb = static_cast<int>(ldb);
    const int ildc = static_cast<int>(ldc);
    LINEARTETRAHEDRON_BLAS_ZHEMM(
        &side,
        &uplo,
        &im,
        &in,
        &alpha,
        a,
        &ilda,
        b,
        &ildb,
        &beta,
        c,
        &ildc
    );
}

bool positive_definite(std::vector<Complex> block, size_t size, double tol) {
    if (size == 0) {
        return true;
    }
    const auto margin = positive_definite_margin(tol);
    for (size_t index = 0; index < size; ++index) {
        block[column_major_index(index, index, size)] =
            Complex{std::real(block[column_major_index(index, index, size)]), 0.0};
        block[column_major_index(index, index, size)] -= margin;
    }
    if (size > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("simplex certificate: LAPACK dimension exceeds LP64 range");
    }

    const char uplo = 'L';
    const int n = static_cast<int>(size);
    const int lda = std::max(1, n);
    auto info = 0;
    LINEARTETRAHEDRON_LAPACK_ZPOTRF(&uplo, &n, block.data(), &lda, &info);
    if (info < 0) {
        throw std::runtime_error("simplex certificate: zpotrf failed");
    }
    return info == 0;
}

size_t positive_definite_prefix(std::vector<Complex> block, size_t size, double tol) {
    if (size == 0) {
        return 0;
    }
    const auto margin = positive_definite_margin(tol);
    for (size_t index = 0; index < size; ++index) {
        block[column_major_index(index, index, size)] =
            Complex{std::real(block[column_major_index(index, index, size)]), 0.0};
        block[column_major_index(index, index, size)] -= margin;
    }
    if (size > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("simplex certificate: LAPACK dimension exceeds LP64 range");
    }

    const char uplo = 'L';
    const int n = static_cast<int>(size);
    const int lda = std::max(1, n);
    auto info = 0;
    LINEARTETRAHEDRON_LAPACK_ZPOTRF(&uplo, &n, block.data(), &lda, &info);
    if (info < 0) {
        throw std::runtime_error("simplex certificate: zpotrf failed");
    }
    if (info == 0) {
        return size;
    }
    return static_cast<size_t>(info - 1);
}

double positive_definite_margin(double tol) {
    return std::max(kBlockMargin, tol);
}

}  // namespace lineartetrahedron::simplex_certificate::detail
