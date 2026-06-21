#include "lineartetrahedron/linalg.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

#ifndef LINEARTETRAHEDRON_LAPACK_DSTEV
#define LINEARTETRAHEDRON_LAPACK_DSTEV dstev_
#endif

#ifndef LINEARTETRAHEDRON_BLAS_ZHEMV
#define LINEARTETRAHEDRON_BLAS_ZHEMV zhemv_
#endif

#ifndef LINEARTETRAHEDRON_BLAS_ZGEMV
#define LINEARTETRAHEDRON_BLAS_ZGEMV zgemv_
#endif

extern "C" {
void LINEARTETRAHEDRON_LAPACK_DSTEV(
    const char *jobz,
    const int *n,
    double *d,
    double *e,
    double *z,
    const int *ldz,
    double *work,
    int *info
);

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

thread_local LanczosStats lanczos_stats_;

using Clock = std::chrono::steady_clock;

std::uint64_t nanoseconds_since(Clock::time_point start) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count()
    );
}

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

void axpy(
    Complex alpha,
    std::span<const Complex> x,
    std::span<Complex> y
) {
    if (x.size() != y.size()) {
        throw std::runtime_error("Lanczos: axpy size mismatch");
    }
    for (size_t index = 0; index < x.size(); ++index) {
        y[index] += alpha * x[index];
    }
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

struct RitzNormBounds {
    double lower = 0.0;
    double upper = 0.0;
};

struct RitzMinEigenvalueBounds {
    double lower = 0.0;
    double upper = 0.0;
};

RitzNormBounds tridiagonal_spectral_norm_bounds(
    std::span<const double> diagonal,
    std::span<const double> offdiagonal,
    double residual_beta,
    std::vector<double> &eigenvalues,
    std::vector<double> &offdiagonal_copy,
    std::vector<double> &eigenvectors,
    std::vector<double> &work
) {
    const auto size = diagonal.size();
    if (size == 0) {
        return {};
    }
    if (offdiagonal.size() + 1 != size) {
        throw std::runtime_error("Lanczos: tridiagonal offdiagonal size mismatch");
    }

    eigenvalues.assign(diagonal.begin(), diagonal.end());
    offdiagonal_copy.assign(offdiagonal.begin(), offdiagonal.end());
    eigenvectors.assign(size * size, 0.0);
    work.assign(std::max<size_t>(1, 2 * size - 2), 0.0);

    const char jobz = 'V';
    const int n = static_cast<int>(size);
    const int ldz = std::max(1, n);
    auto info = 0;
    LINEARTETRAHEDRON_LAPACK_DSTEV(
        &jobz,
        &n,
        eigenvalues.data(),
        offdiagonal_copy.data(),
        eigenvectors.data(),
        &ldz,
        work.data(),
        &info
    );
    if (info != 0) {
        throw std::runtime_error("Lanczos: dstev failed with info=" + std::to_string(info));
    }

    const auto min_index = size_t{0};
    const auto max_index = size - 1;
    const auto theta_min = eigenvalues[min_index];
    const auto theta_max = eigenvalues[max_index];
    const auto eigenvector_at = [ldz](size_t row, size_t col, const std::vector<double> &values) {
        return values[row + col * static_cast<size_t>(ldz)];
    };
    const auto r_min =
        std::abs(residual_beta) *
        std::abs(eigenvector_at(size - 1, min_index, eigenvectors));
    const auto r_max =
        std::abs(residual_beta) *
        std::abs(eigenvector_at(size - 1, max_index, eigenvectors));
    const auto lower = std::max(theta_max, -theta_min);
    const auto upper = std::max(theta_max + r_max, -theta_min + r_min);
    return RitzNormBounds{.lower = lower, .upper = std::max(lower, upper)};
}

RitzMinEigenvalueBounds tridiagonal_min_eigenvalue_bounds(
    std::span<const double> diagonal,
    std::span<const double> offdiagonal,
    double residual_beta,
    std::vector<double> &eigenvalues,
    std::vector<double> &offdiagonal_copy,
    std::vector<double> &eigenvectors,
    std::vector<double> &work
) {
    const auto size = diagonal.size();
    if (size == 0) {
        return {};
    }
    if (offdiagonal.size() + 1 != size) {
        throw std::runtime_error("Lanczos: tridiagonal offdiagonal size mismatch");
    }

    eigenvalues.assign(diagonal.begin(), diagonal.end());
    offdiagonal_copy.assign(offdiagonal.begin(), offdiagonal.end());
    eigenvectors.assign(size * size, 0.0);
    work.assign(std::max<size_t>(1, 2 * size - 2), 0.0);

    const char jobz = 'V';
    const int n = static_cast<int>(size);
    const int ldz = std::max(1, n);
    auto info = 0;
    LINEARTETRAHEDRON_LAPACK_DSTEV(
        &jobz,
        &n,
        eigenvalues.data(),
        offdiagonal_copy.data(),
        eigenvectors.data(),
        &ldz,
        work.data(),
        &info
    );
    if (info != 0) {
        throw std::runtime_error("Lanczos: dstev failed with info=" + std::to_string(info));
    }

    const auto min_index = size_t{0};
    const auto theta_min = eigenvalues[min_index];
    const auto eigenvector_at = [ldz](size_t row, size_t col, const std::vector<double> &values) {
        return values[row + col * static_cast<size_t>(ldz)];
    };
    const auto residual =
        std::abs(residual_beta) *
        std::abs(eigenvector_at(size - 1, min_index, eigenvectors));
    return RitzMinEigenvalueBounds{
        .lower = theta_min - residual,
        .upper = theta_min + residual,
    };
}

template <class Matvec>
double spectral_norm_lanczos(
    size_t size,
    Matvec matvec,
    double absolute_uncertainty
) {
    if (size == 0) {
        return 0.0;
    }
    if (size > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Lanczos: matrix dimension exceeds LP64 range");
    }
    if (!(absolute_uncertainty >= 0.0) || !std::isfinite(absolute_uncertainty)) {
        throw std::runtime_error("Lanczos: absolute uncertainty must be finite and nonnegative");
    }

    constexpr auto residual_tolerance = 1e-12;
    constexpr size_t min_iterations = 8;
    constexpr size_t residual_check_period = 8;
    ++lanczos_stats_.calls;
    const auto lanczos_start = Clock::now();
    auto q = deterministic_start(size);
    std::vector<Complex> basis(size * size, Complex{0.0, 0.0});
    std::vector<double> diagonal;
    std::vector<double> offdiagonal;
    diagonal.reserve(size);
    offdiagonal.reserve(size > 0 ? size - 1 : 0);
    std::vector<double> ritz_eigenvalues;
    std::vector<double> ritz_offdiagonal;
    std::vector<double> ritz_eigenvectors;
    std::vector<double> ritz_work;
    ritz_eigenvalues.reserve(size);
    ritz_offdiagonal.reserve(size > 0 ? size - 1 : 0);
    ritz_eigenvectors.reserve(size * size);
    ritz_work.reserve(std::max<size_t>(1, 2 * size - 2));

    auto beta_previous = 0.0;
    std::vector<Complex> previous(size, Complex{0.0, 0.0});
    std::vector<Complex> z(size, Complex{0.0, 0.0});
    RitzNormBounds bounds;
    for (size_t iteration = 0; iteration < size; ++iteration) {
        ++lanczos_stats_.iterations;
        std::copy(q.begin(), q.end(), basis.begin() + static_cast<std::ptrdiff_t>(iteration * size));
        matvec(q, z);
        if (iteration > 0) {
            axpy(
                Complex{-beta_previous, 0.0},
                std::span<const Complex>(previous.data(), previous.size()),
                std::span<Complex>(z.data(), z.size())
            );
        }

        const auto alpha = std::real(inner_product(q, z));
        diagonal.push_back(alpha);
        axpy(
            Complex{-alpha, 0.0},
            std::span<const Complex>(q.data(), q.size()),
            std::span<Complex>(z.data(), z.size())
        );

        for (size_t basis_index = 0; basis_index <= iteration; ++basis_index) {
            const auto basis_vector = std::span<const Complex>(
                basis.data() + static_cast<std::ptrdiff_t>(basis_index * size),
                size
            );
            const auto overlap = inner_product(basis_vector, z);
            axpy(
                -overlap,
                basis_vector,
                std::span<Complex>(z.data(), z.size())
            );
        }

        const auto beta = vector_norm(z);
        const auto iteration_count = iteration + 1;
        const auto final_iteration = iteration + 1 == size || beta <= residual_tolerance;
        const auto can_check =
            final_iteration ||
            (iteration_count >= std::min(min_iterations, size) &&
             iteration_count % residual_check_period == 0);
        if (can_check) {
            ++lanczos_stats_.ritz_checks;
            bounds = tridiagonal_spectral_norm_bounds(
                diagonal,
                offdiagonal,
                beta,
                ritz_eigenvalues,
                ritz_offdiagonal,
                ritz_eigenvectors,
                ritz_work
            );
            const auto uncertainty = bounds.upper - bounds.lower;
            if (uncertainty <= absolute_uncertainty || final_iteration) {
                if (uncertainty <= absolute_uncertainty) {
                    ++lanczos_stats_.converged_by_uncertainty;
                } else if (beta <= residual_tolerance) {
                    ++lanczos_stats_.converged_by_zero_residual;
                } else {
                    ++lanczos_stats_.converged_at_full_dimension;
                }
                lanczos_stats_.lanczos_nanoseconds += nanoseconds_since(lanczos_start);
                return bounds.upper;
            }
        }
        if (final_iteration) {
            if (beta <= residual_tolerance) {
                ++lanczos_stats_.converged_by_zero_residual;
            } else {
                ++lanczos_stats_.converged_at_full_dimension;
            }
            lanczos_stats_.lanczos_nanoseconds += nanoseconds_since(lanczos_start);
            return bounds.upper;
        }

        offdiagonal.push_back(beta);
        previous = q;
        for (size_t index = 0; index < size; ++index) {
            q[index] = z[index] / beta;
        }
        beta_previous = beta;
    }

    lanczos_stats_.lanczos_nanoseconds += nanoseconds_since(lanczos_start);
    return bounds.upper;
}

template <class Matvec>
double min_eigenvalue_lanczos(
    size_t size,
    Matvec matvec,
    double absolute_uncertainty
) {
    if (size == 0) {
        return std::numeric_limits<double>::infinity();
    }
    if (size > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Lanczos: matrix dimension exceeds LP64 range");
    }
    if (!(absolute_uncertainty >= 0.0) || !std::isfinite(absolute_uncertainty)) {
        throw std::runtime_error("Lanczos: absolute uncertainty must be finite and nonnegative");
    }

    constexpr auto residual_tolerance = 1e-12;
    constexpr size_t min_iterations = 8;
    constexpr size_t residual_check_period = 8;
    ++lanczos_stats_.calls;
    const auto lanczos_start = Clock::now();
    auto q = deterministic_start(size);
    std::vector<Complex> basis(size * size, Complex{0.0, 0.0});
    std::vector<double> diagonal;
    std::vector<double> offdiagonal;
    diagonal.reserve(size);
    offdiagonal.reserve(size > 0 ? size - 1 : 0);
    std::vector<double> ritz_eigenvalues;
    std::vector<double> ritz_offdiagonal;
    std::vector<double> ritz_eigenvectors;
    std::vector<double> ritz_work;
    ritz_eigenvalues.reserve(size);
    ritz_offdiagonal.reserve(size > 0 ? size - 1 : 0);
    ritz_eigenvectors.reserve(size * size);
    ritz_work.reserve(std::max<size_t>(1, 2 * size - 2));

    auto beta_previous = 0.0;
    std::vector<Complex> previous(size, Complex{0.0, 0.0});
    std::vector<Complex> z(size, Complex{0.0, 0.0});
    RitzMinEigenvalueBounds bounds;
    for (size_t iteration = 0; iteration < size; ++iteration) {
        ++lanczos_stats_.iterations;
        std::copy(q.begin(), q.end(), basis.begin() + static_cast<std::ptrdiff_t>(iteration * size));
        matvec(q, z);
        if (iteration > 0) {
            axpy(
                Complex{-beta_previous, 0.0},
                std::span<const Complex>(previous.data(), previous.size()),
                std::span<Complex>(z.data(), z.size())
            );
        }

        const auto alpha = std::real(inner_product(q, z));
        diagonal.push_back(alpha);
        axpy(
            Complex{-alpha, 0.0},
            std::span<const Complex>(q.data(), q.size()),
            std::span<Complex>(z.data(), z.size())
        );

        for (size_t basis_index = 0; basis_index <= iteration; ++basis_index) {
            const auto basis_vector = std::span<const Complex>(
                basis.data() + static_cast<std::ptrdiff_t>(basis_index * size),
                size
            );
            const auto overlap = inner_product(basis_vector, z);
            axpy(
                -overlap,
                basis_vector,
                std::span<Complex>(z.data(), z.size())
            );
        }

        const auto beta = vector_norm(z);
        const auto iteration_count = iteration + 1;
        const auto final_iteration = iteration + 1 == size || beta <= residual_tolerance;
        const auto can_check =
            final_iteration ||
            (iteration_count >= std::min(min_iterations, size) &&
             iteration_count % residual_check_period == 0);
        if (can_check) {
            ++lanczos_stats_.ritz_checks;
            bounds = tridiagonal_min_eigenvalue_bounds(
                diagonal,
                offdiagonal,
                beta,
                ritz_eigenvalues,
                ritz_offdiagonal,
                ritz_eigenvectors,
                ritz_work
            );
            const auto uncertainty = bounds.upper - bounds.lower;
            if (uncertainty <= absolute_uncertainty || final_iteration) {
                if (uncertainty <= absolute_uncertainty) {
                    ++lanczos_stats_.converged_by_uncertainty;
                } else if (beta <= residual_tolerance) {
                    ++lanczos_stats_.converged_by_zero_residual;
                } else {
                    ++lanczos_stats_.converged_at_full_dimension;
                }
                lanczos_stats_.lanczos_nanoseconds += nanoseconds_since(lanczos_start);
                return bounds.lower;
            }
        }
        if (final_iteration) {
            if (beta <= residual_tolerance) {
                ++lanczos_stats_.converged_by_zero_residual;
            } else {
                ++lanczos_stats_.converged_at_full_dimension;
            }
            lanczos_stats_.lanczos_nanoseconds += nanoseconds_since(lanczos_start);
            return bounds.lower;
        }

        offdiagonal.push_back(beta);
        previous = q;
        for (size_t index = 0; index < size; ++index) {
            q[index] = z[index] / beta;
        }
        beta_previous = beta;
    }

    lanczos_stats_.lanczos_nanoseconds += nanoseconds_since(lanczos_start);
    return bounds.lower;
}

}  // namespace

void reset_lanczos_stats() {
    lanczos_stats_ = LanczosStats{};
}

LanczosStats lanczos_stats() {
    return lanczos_stats_;
}

void record_derivative_spectral_norm_timing(
    std::uint64_t assembly_nanoseconds,
    std::uint64_t total_nanoseconds
) {
    ++lanczos_stats_.derivative_calls;
    lanczos_stats_.derivative_assembly_nanoseconds += assembly_nanoseconds;
    lanczos_stats_.derivative_total_nanoseconds += total_nanoseconds;
}

double hermitian_spectral_norm_lanczos(
    std::span<const Complex> matrix,
    size_t size,
    double absolute_uncertainty
) {
    if (matrix.size() != size * size) {
        throw std::runtime_error("hermitian_spectral_norm_lanczos: matrix size mismatch");
    }
    return spectral_norm_lanczos(
        size,
        [&](std::span<const Complex> x, std::vector<Complex> &y) {
            hermitian_matvec(matrix, size, x, y);
        },
        absolute_uncertainty
    );
}

double hermitian_min_eigenvalue_lanczos(
    std::span<const Complex> matrix,
    size_t size,
    double absolute_uncertainty
) {
    if (matrix.size() != size * size) {
        throw std::runtime_error("hermitian_min_eigenvalue_lanczos: matrix size mismatch");
    }
    return min_eigenvalue_lanczos(
        size,
        [&](std::span<const Complex> x, std::vector<Complex> &y) {
            hermitian_matvec(matrix, size, x, y);
        },
        absolute_uncertainty
    );
}

double matrix_spectral_norm_lanczos(
    std::span<const Complex> matrix,
    size_t size,
    double absolute_uncertainty
) {
    if (matrix.size() != size * size) {
        throw std::runtime_error("matrix_spectral_norm_lanczos: matrix size mismatch");
    }
    return spectral_norm_lanczos(
        2 * size,
        [&](std::span<const Complex> x, std::vector<Complex> &y) {
            dilation_matvec(matrix, size, x, y);
        },
        absolute_uncertainty
    );
}

}  // namespace lineartetrahedron
