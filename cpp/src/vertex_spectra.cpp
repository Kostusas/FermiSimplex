#include "lineartetrahedron/vertex_spectra.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

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

void diagonalize_hermitian_in_place(
    std::vector<std::complex<double>> &matrix,
    std::vector<double> &eigenvalues,
    size_t ndof
) {
    if (ndof > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("VertexSpectraEvaluator: LAPACK matrix dimension exceeds LP64 range");
    }

    const char jobz = 'V';
    const char uplo = 'L';
    const int n = static_cast<int>(ndof);
    const int lda = std::max(1, n);
    int info = 0;
    eigenvalues.resize(ndof);

    if (n == 0) {
        return;
    }

    int lwork = -1;
    int lrwork = -1;
    int liwork = -1;
    std::complex<double> work_query = 0.0;
    double rwork_query = 0.0;
    int iwork_query = 0;
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
            "VertexSpectraEvaluator: zheevd workspace query failed with info=" +
            std::to_string(info)
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
        throw std::runtime_error(
            "VertexSpectraEvaluator: zheevd failed with info=" + std::to_string(info)
        );
    }
}

}  // namespace

VertexSpectraEvaluator::VertexSpectraEvaluator(
    std::shared_ptr<TightBindingModel> model
) : model_(std::move(model)) {
    if (!model_) {
        throw std::runtime_error("VertexSpectraEvaluator: model must not be null");
    }
}

VertexSpectra VertexSpectraEvaluator::evaluate(
    const adaptivesimplex::core::Geometry &geometry,
    adaptivesimplex::core::VertexId vertex_id
) {
    const auto reduced_point = geometry.vertices().dyadic_vertex(vertex_id).to_point();
    return diagonalize_reduced_point(reduced_point);
}

VertexSpectra VertexSpectraEvaluator::diagonalize_reduced_point(
    const std::vector<double> &reduced_point
) {
    std::vector<double> k_point(model_->ndim());
    for (size_t axis = 0; axis < model_->ndim(); ++axis) {
        k_point[axis] = 2.0 * kPi * reduced_point[axis] - kPi;
    }

    auto h = model_->evaluate_point_raw(k_point.data());
    VertexSpectra entry;
    diagonalize_hermitian_in_place(h, entry.eigenvalues, model_->ndof());
    entry.eigenvectors = std::move(h);
    return entry;
}

}  // namespace lineartetrahedron
