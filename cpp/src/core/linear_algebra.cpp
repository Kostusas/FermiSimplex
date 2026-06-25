#include "core/linear_algebra.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

#ifndef LINEARTETRAHEDRON_LAPACK_ZHEEVD
#define LINEARTETRAHEDRON_LAPACK_ZHEEVD zheevd_
#endif

extern "C" {
void LINEARTETRAHEDRON_LAPACK_ZHEEVD(
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
}

namespace lineartetrahedron {
namespace {

std::string prefix(const char *context) {
    if (context == nullptr || context[0] == '\0') {
        return "linear algebra: ";
    }
    return std::string(context) + ": ";
}

}  // namespace

void diagonalize_hermitian_in_place(
    std::vector<std::complex<double>> &matrix,
    std::vector<double> &eigenvalues,
    size_t size,
    bool compute_vectors,
    const char *context
) {
    const auto error_prefix = prefix(context);
    if (size > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(error_prefix + "LAPACK matrix dimension exceeds LP64 range");
    }

    const char jobz = compute_vectors ? 'V' : 'N';
    const char uplo = 'L';
    const int n = static_cast<int>(size);
    const int lda = std::max(1, n);
    auto info = 0;
    eigenvalues.resize(size);

    if (n == 0) {
        return;
    }

    auto lwork = -1;
    auto lrwork = -1;
    auto liwork = -1;
    std::complex<double> work_query = 0.0;
    auto rwork_query = 0.0;
    auto iwork_query = 0;
    LINEARTETRAHEDRON_LAPACK_ZHEEVD(
        &jobz,
        &uplo,
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
            error_prefix + "zheevd workspace query failed with info=" + std::to_string(info)
        );
    }

    lwork = std::max(1, static_cast<int>(std::real(work_query)));
    lrwork = std::max(1, static_cast<int>(rwork_query));
    liwork = std::max(1, iwork_query);
    std::vector<std::complex<double>> work(static_cast<size_t>(lwork));
    std::vector<double> rwork(static_cast<size_t>(lrwork));
    std::vector<int> iwork(static_cast<size_t>(liwork));
    LINEARTETRAHEDRON_LAPACK_ZHEEVD(
        &jobz,
        &uplo,
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
        throw std::runtime_error(error_prefix + "zheevd failed with info=" + std::to_string(info));
    }
}

}  // namespace lineartetrahedron
