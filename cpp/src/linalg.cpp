#include "lineartetrahedron/linalg.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#ifndef LINEARTETRAHEDRON_BLAS_ZHEMV
#define LINEARTETRAHEDRON_BLAS_ZHEMV zhemv_
#endif

#ifndef LINEARTETRAHEDRON_BLAS_ZGEMV
#define LINEARTETRAHEDRON_BLAS_ZGEMV zgemv_
#endif

extern "C" {
void LINEARTETRAHEDRON_BLAS_ZHEMV(
    const char *uplo,
    const int *n,
    const std::complex<double> *alpha,
    const std::complex<double> *a,
    const int *lda,
    const std::complex<double> *x,
    const int *incx,
    const std::complex<double> *beta,
    std::complex<double> *y,
    const int *incy
);

void LINEARTETRAHEDRON_BLAS_ZGEMV(
    const char *trans,
    const int *m,
    const int *n,
    const std::complex<double> *alpha,
    const std::complex<double> *a,
    const int *lda,
    const std::complex<double> *x,
    const int *incx,
    const std::complex<double> *beta,
    std::complex<double> *y,
    const int *incy
);
}

namespace lineartetrahedron {
namespace {

using Complex = std::complex<double>;

double vector_norm(std::span<const Complex> values) {
    auto squared = 0.0;
    for (const auto value : values) {
        squared += std::norm(value);
    }
    return std::sqrt(squared);
}

Complex inner_product(
    std::span<const Complex> left,
    std::span<const Complex> right
) {
    auto result = Complex{0.0, 0.0};
    for (size_t index = 0; index < left.size(); ++index) {
        result += std::conj(left[index]) * right[index];
    }
    return result;
}

void normalize(std::vector<Complex> &values) {
    const auto norm = vector_norm(values);
    if (norm == 0.0) {
        throw std::runtime_error("Lanczos: zero starting vector");
    }
    for (auto &value : values) {
        value /= norm;
    }
}

std::vector<Complex> deterministic_start(size_t size) {
    std::vector<Complex> values(size);
    for (size_t index = 0; index < size; ++index) {
        values[index] = Complex{
            1.0 + static_cast<double>((index * 37U) % 11U),
            0.5 + static_cast<double>((index * 17U) % 7U),
        };
    }
    normalize(values);
    return values;
}

void hermitian_matvec(
    std::span<const Complex> matrix,
    size_t size,
    std::span<const Complex> x,
    std::vector<Complex> &y
) {
    const char uplo = 'L';
    const int n = static_cast<int>(size);
    const int inc = 1;
    const auto alpha = Complex{1.0, 0.0};
    const auto beta = Complex{0.0, 0.0};
    y.assign(size, Complex{0.0, 0.0});
    LINEARTETRAHEDRON_BLAS_ZHEMV(
        &uplo,
        &n,
        &alpha,
        matrix.data(),
        &n,
        x.data(),
        &inc,
        &beta,
        y.data(),
        &inc
    );
}

void dilation_matvec(
    std::span<const Complex> matrix,
    size_t size,
    std::span<const Complex> x,
    std::vector<Complex> &y
) {
    const int n = static_cast<int>(size);
    const int inc = 1;
    const auto alpha = Complex{1.0, 0.0};
    const auto beta = Complex{0.0, 0.0};
    y.assign(2 * size, Complex{0.0, 0.0});

    const char normal = 'N';
    LINEARTETRAHEDRON_BLAS_ZGEMV(
        &normal,
        &n,
        &n,
        &alpha,
        matrix.data(),
        &n,
        x.data() + size,
        &inc,
        &beta,
        y.data(),
        &inc
    );

    const char adjoint = 'C';
    LINEARTETRAHEDRON_BLAS_ZGEMV(
        &adjoint,
        &n,
        &n,
        &alpha,
        matrix.data(),
        &n,
        x.data(),
        &inc,
        &beta,
        y.data() + size,
        &inc
    );
}

double largest_abs_eigenvalue_jacobi(std::vector<double> matrix, size_t size) {
    if (size == 0) {
        return 0.0;
    }
    const auto at = [size](size_t row, size_t col) {
        return row * size + col;
    };
    const auto tolerance = 64.0 * std::numeric_limits<double>::epsilon();
    const size_t max_sweeps = std::max<size_t>(32, 32 * size * size);

    for (size_t sweep = 0; sweep < max_sweeps; ++sweep) {
        auto changed = false;
        for (size_t p = 0; p < size; ++p) {
            for (size_t q = p + 1; q < size; ++q) {
                const auto apq = matrix[at(p, q)];
                if (std::abs(apq) <= tolerance) {
                    continue;
                }
                changed = true;
                const auto app = matrix[at(p, p)];
                const auto aqq = matrix[at(q, q)];
                const auto tau = (aqq - app) / (2.0 * apq);
                const auto sign = tau >= 0.0 ? 1.0 : -1.0;
                const auto t = sign / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
                const auto c = 1.0 / std::sqrt(1.0 + t * t);
                const auto s = t * c;

                matrix[at(p, p)] = app - t * apq;
                matrix[at(q, q)] = aqq + t * apq;
                matrix[at(p, q)] = 0.0;
                matrix[at(q, p)] = 0.0;

                for (size_t r = 0; r < size; ++r) {
                    if (r == p || r == q) {
                        continue;
                    }
                    const auto arp = matrix[at(r, p)];
                    const auto arq = matrix[at(r, q)];
                    matrix[at(r, p)] = c * arp - s * arq;
                    matrix[at(p, r)] = matrix[at(r, p)];
                    matrix[at(r, q)] = s * arp + c * arq;
                    matrix[at(q, r)] = matrix[at(r, q)];
                }
            }
        }
        if (!changed) {
            break;
        }
    }

    auto result = 0.0;
    for (size_t index = 0; index < size; ++index) {
        result = std::max(result, std::abs(matrix[at(index, index)]));
    }
    return result;
}

double tridiagonal_spectral_radius(
    std::span<const double> diagonal,
    std::span<const double> offdiagonal
) {
    const auto size = diagonal.size();
    std::vector<double> dense(size * size, 0.0);
    for (size_t index = 0; index < size; ++index) {
        dense[index * size + index] = diagonal[index];
        if (index + 1 < size) {
            dense[index * size + index + 1] = offdiagonal[index];
            dense[(index + 1) * size + index] = offdiagonal[index];
        }
    }
    return largest_abs_eigenvalue_jacobi(std::move(dense), size);
}

template <class Matvec>
double spectral_norm_lanczos(size_t size, Matvec matvec) {
    if (size == 0) {
        return 0.0;
    }
    if (size > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Lanczos: matrix dimension exceeds LP64 range");
    }

    constexpr auto residual_tolerance = 1e-12;
    auto q = deterministic_start(size);
    std::vector<std::vector<Complex>> basis;
    std::vector<double> diagonal;
    std::vector<double> offdiagonal;
    basis.reserve(size);
    diagonal.reserve(size);
    offdiagonal.reserve(size > 0 ? size - 1 : 0);

    auto beta_previous = 0.0;
    std::vector<Complex> previous(size, Complex{0.0, 0.0});
    for (size_t iteration = 0; iteration < size; ++iteration) {
        basis.push_back(q);
        std::vector<Complex> z;
        matvec(q, z);
        if (iteration > 0) {
            for (size_t index = 0; index < size; ++index) {
                z[index] -= beta_previous * previous[index];
            }
        }

        const auto alpha = std::real(inner_product(q, z));
        diagonal.push_back(alpha);
        for (size_t index = 0; index < size; ++index) {
            z[index] -= alpha * q[index];
        }

        for (const auto &basis_vector : basis) {
            const auto overlap = inner_product(basis_vector, z);
            for (size_t index = 0; index < size; ++index) {
                z[index] -= overlap * basis_vector[index];
            }
        }

        const auto beta = vector_norm(z);
        if (iteration + 1 == size || beta <= residual_tolerance) {
            break;
        }

        offdiagonal.push_back(beta);
        previous = std::move(q);
        q = std::move(z);
        for (auto &value : q) {
            value /= beta;
        }
        beta_previous = beta;
    }

    return tridiagonal_spectral_radius(diagonal, offdiagonal);
}

}  // namespace

double hermitian_spectral_norm_lanczos(
    std::span<const Complex> matrix,
    size_t size
) {
    if (matrix.size() != size * size) {
        throw std::runtime_error("hermitian_spectral_norm_lanczos: matrix size mismatch");
    }
    return spectral_norm_lanczos(
        size,
        [&](std::span<const Complex> x, std::vector<Complex> &y) {
            hermitian_matvec(matrix, size, x, y);
        }
    );
}

double matrix_spectral_norm_lanczos(
    std::span<const Complex> matrix,
    size_t size
) {
    if (matrix.size() != size * size) {
        throw std::runtime_error("matrix_spectral_norm_lanczos: matrix size mismatch");
    }
    return spectral_norm_lanczos(
        2 * size,
        [&](std::span<const Complex> x, std::vector<Complex> &y) {
            dilation_matvec(matrix, size, x, y);
        }
    );
}

}  // namespace lineartetrahedron
