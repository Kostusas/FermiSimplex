#include "linalg/blas_lapack.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#ifndef FERMISIMPLEX_LAPACK_ZHEEVD
#define FERMISIMPLEX_LAPACK_ZHEEVD zheevd_
#endif

#ifndef FERMISIMPLEX_LAPACK_ZPOTRF
#define FERMISIMPLEX_LAPACK_ZPOTRF zpotrf_
#endif

extern "C" {
void FERMISIMPLEX_LAPACK_ZHEEVD(
    const char *jobz,
    const char *uplo,
    const int *n,
    std::complex<double> *a,
    const int *lda,
    double *w,
    std::complex<double> *work,
    const int *lwork,
    double *rwork,
    const int *lrwork,
    int *iwork,
    const int *liwork,
    int *info
);

void FERMISIMPLEX_LAPACK_ZPOTRF(
    const char *uplo,
    const int *n,
    std::complex<double> *a,
    const int *lda,
    int *info
);
}

namespace fermisimplex::linalg {
namespace {

std::string error_prefix(const char *context) {
    if (context == nullptr || context[0] == '\0') {
        return "linear algebra: ";
    }
    return std::string(context) + ": ";
}

int lapack_dimension(size_t value) {
    if (value > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("LAPACK dimension exceeds the LP64 range");
    }
    return static_cast<int>(value);
}

int workspace_size(double value, const std::string &prefix) {
    if (!std::isfinite(value) || value > static_cast<double>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(prefix + "invalid LAPACK workspace size");
    }
    return value < 1.0 ? 1 : static_cast<int>(value);
}

}  // namespace

int cholesky_factor_lower(Complex *matrix, size_t size) {
    if (size == 0) {
        return 0;
    }

    const char lower_triangle = 'L';
    const auto n = lapack_dimension(size);
    const auto lda = std::max(1, n);
    auto info = 0;
    FERMISIMPLEX_LAPACK_ZPOTRF(&lower_triangle, &n, matrix, &lda, &info);
    if (info < 0) {
        throw std::runtime_error(
            "Cholesky factorization: zpotrf received an invalid argument (info=" +
            std::to_string(info) + ")"
        );
    }
    return info;
}

void diagonalize_hermitian_in_place(
    std::vector<Complex> &matrix,
    std::vector<double> &eigenvalues,
    size_t size,
    bool compute_vectors,
    const char *context
) {
    const auto prefix = error_prefix(context);
    if (size != 0 && (matrix.size() / size != size || matrix.size() != size * size)) {
        throw std::runtime_error(prefix + "Hermitian matrix shape mismatch");
    }

    const char jobz = compute_vectors ? 'V' : 'N';
    const char lower_triangle = 'L';
    const auto n = lapack_dimension(size);
    const auto lda = std::max(1, n);
    eigenvalues.resize(size);
    if (n == 0) {
        return;
    }

    auto lwork = -1;
    auto lrwork = -1;
    auto liwork = -1;
    Complex work_query = 0.0;
    auto rwork_query = 0.0;
    auto iwork_query = 0;
    auto info = 0;
    FERMISIMPLEX_LAPACK_ZHEEVD(
        &jobz,
        &lower_triangle,
        &n,
        matrix.data(),
        &lda,
        eigenvalues.data(),
        &work_query,
        &lwork,
        &rwork_query,
        &lrwork,
        &iwork_query,
        &liwork,
        &info
    );
    if (info != 0) {
        throw std::runtime_error(
            prefix + "zheevd workspace query failed with info=" + std::to_string(info)
        );
    }

    lwork = workspace_size(std::real(work_query), prefix);
    lrwork = workspace_size(rwork_query, prefix);
    liwork = workspace_size(static_cast<double>(iwork_query), prefix);
    std::vector<Complex> work(static_cast<size_t>(lwork));
    std::vector<double> rwork(static_cast<size_t>(lrwork));
    std::vector<int> iwork(static_cast<size_t>(liwork));
    FERMISIMPLEX_LAPACK_ZHEEVD(
        &jobz,
        &lower_triangle,
        &n,
        matrix.data(),
        &lda,
        eigenvalues.data(),
        work.data(),
        &lwork,
        rwork.data(),
        &lrwork,
        iwork.data(),
        &liwork,
        &info
    );
    if (info != 0) {
        throw std::runtime_error(prefix + "zheevd failed with info=" + std::to_string(info));
    }
}

}  // namespace fermisimplex::linalg
