#include "lineartetrahedron/simplex_certificate.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

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

namespace lineartetrahedron::simplex_certificate {

namespace {

using Complex = std::complex<double>;

constexpr double kBlockMargin = 1e-10;

size_t column_major_index(size_t row, size_t column, size_t rows) {
    return row + column * rows;
}

double signed_eigenvalue(const VertexSpectra &spectra, size_t band, double mu) {
    return spectra.eigenvalues[band] - mu;
}

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
    if (
        m > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        n > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        k > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        lda > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        ldb > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        ldc > static_cast<size_t>(std::numeric_limits<int>::max())
    ) {
        throw std::runtime_error("simplex certificate: BLAS dimension exceeds LP64 range");
    }

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
    if (
        n > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        k > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        lda > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        ldb > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        ldc > static_cast<size_t>(std::numeric_limits<int>::max())
    ) {
        throw std::runtime_error("simplex certificate: BLAS dimension exceeds LP64 range");
    }

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
    if (
        m > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        n > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        lda > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        ldb > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        ldc > static_cast<size_t>(std::numeric_limits<int>::max())
    ) {
        throw std::runtime_error("simplex certificate: BLAS dimension exceeds LP64 range");
    }

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

std::vector<Complex> full_overlap(
    std::span<const Complex> anchor_vectors,
    size_t ndof,
    std::span<const Complex> target_vectors
) {
    std::vector<Complex> result(ndof * ndof, Complex{0.0, 0.0});
    gemm(
        'C',
        'N',
        ndof,
        ndof,
        ndof,
        Complex{1.0, 0.0},
        anchor_vectors.data(),
        ndof,
        target_vectors.data(),
        ndof,
        Complex{0.0, 0.0},
        result.data(),
        ndof
    );
    return result;
}

std::vector<Complex> scale_overlap_columns(
    const Complex *overlap,
    size_t overlap_rows,
    size_t rows,
    size_t columns,
    const VertexSpectra &target,
    double mu
) {
    std::vector<Complex> result(rows * columns, Complex{0.0, 0.0});
    for (size_t column = 0; column < columns; ++column) {
        const auto scale = signed_eigenvalue(target, column, mu);
        for (size_t row = 0; row < rows; ++row) {
            result[column_major_index(row, column, rows)] =
                scale * overlap[column_major_index(row, column, overlap_rows)];
        }
    }
    return result;
}

std::vector<Complex> submatrix_copy(
    std::span<const Complex> matrix,
    size_t rows,
    size_t row_offset,
    size_t column_offset,
    size_t sub_rows,
    size_t sub_columns
) {
    std::vector<Complex> result(sub_rows * sub_columns, Complex{0.0, 0.0});
    for (size_t column = 0; column < sub_columns; ++column) {
        for (size_t row = 0; row < sub_rows; ++row) {
            result[column_major_index(row, column, sub_rows)] =
                matrix[column_major_index(row + row_offset, column + column_offset, rows)];
        }
    }
    return result;
}

void hermitize_square(std::vector<Complex> &matrix, size_t size) {
    for (size_t column = 0; column < size; ++column) {
        matrix[column_major_index(column, column, size)] =
            Complex{std::real(matrix[column_major_index(column, column, size)]), 0.0};
        for (size_t row = column + 1; row < size; ++row) {
            const auto lower = matrix[column_major_index(row, column, size)];
            const auto upper = std::conj(matrix[column_major_index(column, row, size)]);
            const auto value = 0.5 * (lower + upper);
            matrix[column_major_index(row, column, size)] = value;
            matrix[column_major_index(column, row, size)] = std::conj(value);
        }
    }
}

void add_lower_triangle(
    std::vector<Complex> &target,
    std::span<const Complex> value,
    size_t size
) {
    for (size_t column = 0; column < size; ++column) {
        for (size_t row = column; row < size; ++row) {
            target[column_major_index(row, column, size)] +=
                value[column_major_index(row, column, size)];
        }
    }
}

void negate_in_place(std::vector<Complex> &matrix) {
    for (auto &value : matrix) {
        value = -value;
    }
}

bool positive_definite(
    std::vector<Complex> block,
    size_t size,
    double tol
) {
    if (size == 0) {
        return true;
    }
    const auto margin = std::max(kBlockMargin, tol);
    for (size_t index = 0; index < size; ++index) {
        block[column_major_index(index, index, size)] =
            Complex{std::real(block[column_major_index(index, index, size)]), 0.0};
    }
    for (size_t index = 0; index < size; ++index) {
        block[column_major_index(index, index, size)] -= margin;
    }
    if (size > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("simplex certificate: LAPACK dimension exceeds LP64 range");
    }
    const char uplo = 'L';
    const int n = static_cast<int>(size);
    const int lda = std::max(1, n);
    auto info = 0;
    LINEARTETRAHEDRON_LAPACK_ZPOTRF(
        &uplo,
        &n,
        block.data(),
        &lda,
        &info
    );
    if (info < 0) {
        throw std::runtime_error("simplex certificate: zpotrf failed");
    }
    return info == 0;
}

struct VertexBlocks {
    std::vector<Complex> positive;
    std::vector<Complex> negative;
    std::vector<Complex> mixed;
};

VertexBlocks build_vertex_blocks(
    std::span<const Complex> anchor_vectors,
    size_t ndof,
    size_t npos,
    size_t nneg,
    const VertexSpectra &target,
    double mu
) {
    const auto target_vectors =
        std::span<const Complex>(target.eigenvectors.data(), target.eigenvectors.size());
    const auto overlap = full_overlap(
        anchor_vectors,
        ndof,
        target_vectors
    );
    const auto scaled_overlap = scale_overlap_columns(
        overlap.data(),
        ndof,
        ndof,
        ndof,
        target,
        mu
    );

    std::vector<Complex> anchor_matrix(ndof * ndof, Complex{0.0, 0.0});
    gemm(
        'N',
        'C',
        ndof,
        ndof,
        ndof,
        Complex{1.0, 0.0},
        scaled_overlap.data(),
        ndof,
        overlap.data(),
        ndof,
        Complex{0.0, 0.0},
        anchor_matrix.data(),
        ndof
    );

    VertexBlocks blocks;
    blocks.positive = submatrix_copy(
        std::span<const Complex>(anchor_matrix.data(), anchor_matrix.size()),
        ndof,
        nneg,
        nneg,
        npos,
        npos
    );
    blocks.negative = submatrix_copy(
        std::span<const Complex>(anchor_matrix.data(), anchor_matrix.size()),
        ndof,
        0,
        0,
        nneg,
        nneg
    );
    blocks.mixed = submatrix_copy(
        std::span<const Complex>(anchor_matrix.data(), anchor_matrix.size()),
        ndof,
        nneg,
        0,
        npos,
        nneg
    );
    hermitize_square(blocks.positive, npos);
    hermitize_square(blocks.negative, nneg);
    return blocks;
}

std::vector<Complex> rotated_positive_block(
    const VertexBlocks &blocks,
    std::span<const Complex> rotation,
    size_t npos,
    size_t nneg
) {
    auto result = blocks.positive;
    if (npos == 0 || nneg == 0) {
        return result;
    }

    her2k(
        'N',
        npos,
        nneg,
        Complex{1.0, 0.0},
        blocks.mixed.data(),
        npos,
        rotation.data(),
        npos,
        1.0,
        result.data(),
        npos
    );

    std::vector<Complex> product(npos * nneg, Complex{0.0, 0.0});
    hemm(
        'R',
        npos,
        nneg,
        Complex{1.0, 0.0},
        blocks.negative.data(),
        nneg,
        rotation.data(),
        npos,
        Complex{0.0, 0.0},
        product.data(),
        npos
    );
    std::vector<Complex> quadratic(npos * npos, Complex{0.0, 0.0});
    gemm(
        'N',
        'C',
        npos,
        npos,
        nneg,
        Complex{1.0, 0.0},
        product.data(),
        npos,
        rotation.data(),
        npos,
        Complex{0.0, 0.0},
        quadratic.data(),
        npos
    );
    add_lower_triangle(result, std::span<const Complex>(quadratic.data(), quadratic.size()), npos);
    return result;
}

std::vector<Complex> rotated_negative_block(
    const VertexBlocks &blocks,
    std::span<const Complex> rotation,
    size_t npos,
    size_t nneg
) {
    auto result = blocks.negative;
    if (npos == 0 || nneg == 0) {
        return result;
    }

    her2k(
        'C',
        nneg,
        npos,
        Complex{-1.0, 0.0},
        blocks.mixed.data(),
        npos,
        rotation.data(),
        npos,
        1.0,
        result.data(),
        nneg
    );

    std::vector<Complex> product(npos * nneg, Complex{0.0, 0.0});
    hemm(
        'L',
        npos,
        nneg,
        Complex{1.0, 0.0},
        blocks.positive.data(),
        npos,
        rotation.data(),
        npos,
        Complex{0.0, 0.0},
        product.data(),
        npos
    );
    std::vector<Complex> quadratic(nneg * nneg, Complex{0.0, 0.0});
    gemm(
        'C',
        'N',
        nneg,
        nneg,
        npos,
        Complex{1.0, 0.0},
        rotation.data(),
        npos,
        product.data(),
        npos,
        Complex{0.0, 0.0},
        quadratic.data(),
        nneg
    );
    add_lower_triangle(result, std::span<const Complex>(quadratic.data(), quadratic.size()), nneg);
    return result;
}

}  // namespace

InertiaDecision classify_rotated_vertex_frame_simplex(
    double mu,
    const core::Geometry &geometry,
    core::SimplexId simplex_id,
    const core::VertexCache<VertexSpectra> &vertex_cache,
    double tol
) {
    const auto &simplex = geometry.simplices().simplex(simplex_id);
    if (simplex.vertex_ids.empty()) {
        return InertiaDecision::Inconclusive;
    }

    const auto &first = vertex_cache.get(simplex.vertex_ids.front());
    const auto ndof = first.eigenvalues.size();
    auto has_reference_occupation = false;
    auto reference_occupation = size_t{0};
    auto best_anchor = simplex.vertex_ids.front();
    auto best_gap = -std::numeric_limits<double>::infinity();

    for (const auto vertex_id : simplex.vertex_ids) {
        const auto &spectra = vertex_cache.get(vertex_id);
        auto vertex_occupation = size_t{0};
        auto min_gap = std::numeric_limits<double>::infinity();
        for (size_t band = 0; band < ndof; ++band) {
            const auto d = signed_eigenvalue(spectra, band, mu);
            const auto gap = std::abs(d);
            if (gap <= tol) {
                return InertiaDecision::VisibleCut;
            }
            min_gap = std::min(min_gap, gap);
            if (d < -tol) {
                ++vertex_occupation;
            }
        }
        if (!has_reference_occupation) {
            reference_occupation = vertex_occupation;
            has_reference_occupation = true;
        } else if (vertex_occupation != reference_occupation) {
            return InertiaDecision::VisibleCut;
        }
        if (min_gap > best_gap) {
            best_gap = min_gap;
            best_anchor = vertex_id;
        }
    }

    const auto &anchor = vertex_cache.get(best_anchor);
    std::vector<size_t> positive_indices;
    std::vector<size_t> negative_indices;
    std::vector<double> positive_gaps;
    std::vector<double> negative_gaps;
    positive_indices.reserve(ndof);
    negative_indices.reserve(ndof);
    positive_gaps.reserve(ndof);
    negative_gaps.reserve(ndof);

    for (size_t band = 0; band < ndof; ++band) {
        const auto d = signed_eigenvalue(anchor, band, mu);
        if (d > tol) {
            positive_indices.push_back(band);
            positive_gaps.push_back(d);
        } else if (d < -tol) {
            negative_indices.push_back(band);
            negative_gaps.push_back(-d);
        } else {
            return InertiaDecision::Inconclusive;
        }
    }

    const auto npos = positive_indices.size();
    const auto nneg = negative_indices.size();
    const auto anchor_vectors =
        std::span<const Complex>(anchor.eigenvectors.data(), anchor.eigenvectors.size());

    std::vector<VertexBlocks> blocks;
    blocks.reserve(simplex.vertex_ids.size());
    std::vector<Complex> average_mixed(npos * nneg, Complex{0.0, 0.0});
    for (const auto vertex_id : simplex.vertex_ids) {
        blocks.push_back(build_vertex_blocks(
            anchor_vectors,
            ndof,
            npos,
            nneg,
            vertex_cache.get(vertex_id),
            mu
        ));
        for (size_t index = 0; index < average_mixed.size(); ++index) {
            average_mixed[index] += blocks.back().mixed[index];
        }
    }

    const auto average_scale = 1.0 / static_cast<double>(simplex.vertex_ids.size());
    for (auto &value : average_mixed) {
        value *= average_scale;
    }

    std::vector<Complex> rotation(npos * nneg, Complex{0.0, 0.0});
    for (size_t neg = 0; neg < nneg; ++neg) {
        for (size_t pos = 0; pos < npos; ++pos) {
            rotation[column_major_index(pos, neg, npos)] =
                average_mixed[column_major_index(pos, neg, npos)] /
                (positive_gaps[pos] + negative_gaps[neg]);
        }
    }

    const auto rotation_span =
        std::span<const Complex>(rotation.data(), rotation.size());
    for (const auto &vertex_blocks : blocks) {
        auto positive_block =
            rotated_positive_block(vertex_blocks, rotation_span, npos, nneg);
        if (!positive_definite(std::move(positive_block), npos, tol)) {
            return InertiaDecision::Inconclusive;
        }

        auto negative_block =
            rotated_negative_block(vertex_blocks, rotation_span, npos, nneg);
        negate_in_place(negative_block);
        if (!positive_definite(std::move(negative_block), nneg, tol)) {
            return InertiaDecision::Inconclusive;
        }
    }

    return InertiaDecision::CertifiedSafe;
}

}  // namespace lineartetrahedron::simplex_certificate
