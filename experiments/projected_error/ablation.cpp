#include <fermisimplex/certification.h>
#include <fermisimplex/spectral_mesh.h>

#include "linalg/blas_lapack.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

using Complex = std::complex<double>;
using Matrix = std::vector<Complex>;
using Eigensystem = fermisimplex::Eigensystem;

std::size_t cm(std::size_t row, std::size_t column, std::size_t rows) {
    return row + column * rows;
}

Matrix random_symmetric(
    std::mt19937_64 &generator,
    std::size_t n,
    double scale
) {
    auto distribution = std::normal_distribution<double>(0.0, scale);
    auto result = Matrix(n * n, Complex{0.0, 0.0});
    for (std::size_t column = 0; column < n; ++column) {
        for (std::size_t row = column; row < n; ++row) {
            const auto value = distribution(generator);
            result[cm(row, column, n)] = value;
            result[cm(column, row, n)] = value;
        }
    }
    return result;
}

Matrix add_affine(
    const Matrix &constant,
    const Matrix &slope,
    double x
) {
    auto result = constant;
    for (std::size_t index = 0; index < result.size(); ++index) {
        result[index] += x * slope[index];
    }
    return result;
}

Eigensystem spectrum(Matrix matrix, std::size_t n) {
    auto values = std::vector<double>{};
    fermisimplex::linalg::diagonalize_hermitian_in_place(
        matrix,
        values,
        n,
        true,
        "union ablation spectrum"
    );
    return Eigensystem{
        .eigenvalues = std::move(values),
        .eigenvectors = std::move(matrix),
    };
}

Matrix band_vectors(
    const Eigensystem &entry,
    std::size_t lower,
    std::size_t upper
) {
    const auto n = entry.eigenvalues.size();
    const auto count = upper - lower;
    auto result = Matrix(n * count, Complex{0.0, 0.0});
    for (std::size_t column = 0; column < count; ++column) {
        for (std::size_t row = 0; row < n; ++row) {
            result[cm(row, column, n)] =
                entry.eigenvectors[cm(row, lower + column, n)];
        }
    }
    return result;
}

Matrix multiply(
    char left_operation,
    char right_operation,
    std::size_t rows,
    std::size_t columns,
    std::size_t inner,
    const Matrix &left,
    std::size_t left_rows,
    const Matrix &right,
    std::size_t right_rows
) {
    auto result = Matrix(rows * columns, Complex{0.0, 0.0});
    fermisimplex::linalg::matrix_multiply(
        left_operation,
        right_operation,
        rows,
        columns,
        inner,
        Complex{1.0, 0.0},
        left.data(),
        left_rows,
        right.data(),
        right_rows,
        Complex{0.0, 0.0},
        result.data(),
        rows
    );
    return result;
}

Matrix project(
    const Matrix &hamilian,
    const Matrix &vectors,
    std::size_t n,
    std::size_t count
) {
    const auto h_times_v =
        multiply('N', 'N', n, count, n, hamilian, n, vectors, n);
    return multiply(
        'C',
        'N',
        count,
        count,
        n,
        vectors,
        n,
        h_times_v,
        n
    );
}

std::vector<double> projected_values(
    const Matrix &hamiltonian,
    const Matrix &vectors,
    std::size_t n,
    std::size_t count
) {
    auto projected = project(hamiltonian, vectors, n, count);
    auto values = std::vector<double>{};
    fermisimplex::linalg::diagonalize_hermitian_in_place(
        projected,
        values,
        count,
        false,
        "union ablation projected values"
    );
    return values;
}

struct Estimate {
    double negative = 0.0;
    double positive = 0.0;
    std::size_t union_dimension = 0;
};

Estimate compare_to_interpolation(
    std::span<const double> projected,
    const Eigensystem &left,
    const Eigensystem &right,
    std::size_t lower,
    std::size_t upper,
    std::size_t union_dimension
) {
    auto result = Estimate{.union_dimension = union_dimension};
    for (std::size_t band = lower; band < upper; ++band) {
        const auto interpolated =
            0.5 * (left.eigenvalues[band] + right.eigenvalues[band]);
        const auto difference = projected[band - lower] - interpolated;
        result.positive = std::max(result.positive, difference);
        result.negative = std::max(result.negative, -difference);
    }
    return result;
}

double block_margin(
    const Eigensystem &entry,
    std::size_t lower,
    std::size_t upper
) {
    const auto n = entry.eigenvalues.size();
    const auto lower_gap = lower == 0
        ? std::numeric_limits<double>::infinity()
        : entry.eigenvalues[lower] - entry.eigenvalues[lower - 1];
    const auto upper_gap = upper == n
        ? std::numeric_limits<double>::infinity()
        : entry.eigenvalues[upper] - entry.eigenvalues[upper - 1];
    return std::min(lower_gap, upper_gap);
}

Estimate anchor_estimate(
    const Matrix &midpoint_hamiltonian,
    const Eigensystem &left,
    const Eigensystem &right,
    std::size_t lower,
    std::size_t upper
) {
    const auto n = left.eigenvalues.size();
    const auto count = upper - lower;
    const auto &anchor =
        block_margin(left, lower, upper) >=
                block_margin(right, lower, upper)
            ? left
            : right;
    const auto vectors = band_vectors(anchor, lower, upper);
    const auto values =
        projected_values(midpoint_hamiltonian, vectors, n, count);
    return compare_to_interpolation(
        values,
        left,
        right,
        lower,
        upper,
        count
    );
}

Estimate endpoint_envelope_estimate(
    const Matrix &midpoint_hamiltonian,
    const Eigensystem &left,
    const Eigensystem &right,
    std::size_t lower,
    std::size_t upper
) {
    const auto n = left.eigenvalues.size();
    const auto count = upper - lower;
    const auto left_values = projected_values(
        midpoint_hamiltonian,
        band_vectors(left, lower, upper),
        n,
        count
    );
    const auto right_values = projected_values(
        midpoint_hamiltonian,
        band_vectors(right, lower, upper),
        n,
        count
    );
    const auto left_estimate = compare_to_interpolation(
        left_values,
        left,
        right,
        lower,
        upper,
        count
    );
    const auto right_estimate = compare_to_interpolation(
        right_values,
        left,
        right,
        lower,
        upper,
        count
    );
    return Estimate{
        .negative = std::max(
            left_estimate.negative,
            right_estimate.negative
        ),
        .positive = std::max(
            left_estimate.positive,
            right_estimate.positive
        ),
        .union_dimension = count,
    };
}

Matrix union_basis(
    const Matrix &left,
    const Matrix &right,
    std::size_t n,
    std::size_t count
) {
    auto candidates = Matrix(n * 2 * count, Complex{0.0, 0.0});
    for (std::size_t column = 0; column < count; ++column) {
        for (std::size_t row = 0; row < n; ++row) {
            candidates[cm(row, column, n)] = left[cm(row, column, n)];
            candidates[cm(row, count + column, n)] =
                right[cm(row, column, n)];
        }
    }

    auto basis = Matrix{};
    basis.reserve(n * std::min(n, 2 * count));
    constexpr auto threshold = 1e-12;
    for (std::size_t candidate = 0;
         candidate < 2 * count && basis.size() < n * n;
         ++candidate) {
        auto vector = Matrix(n, Complex{0.0, 0.0});
        for (std::size_t row = 0; row < n; ++row) {
            vector[row] = candidates[cm(row, candidate, n)];
        }
        for (std::size_t pass = 0; pass < 2; ++pass) {
            const auto rank = basis.size() / n;
            for (std::size_t column = 0; column < rank; ++column) {
                auto overlap = Complex{0.0, 0.0};
                for (std::size_t row = 0; row < n; ++row) {
                    overlap +=
                        std::conj(basis[cm(row, column, n)]) * vector[row];
                }
                for (std::size_t row = 0; row < n; ++row) {
                    vector[row] -= basis[cm(row, column, n)] * overlap;
                }
            }
        }
        auto norm_squared = 0.0;
        for (const auto value : vector) {
            norm_squared += std::norm(value);
        }
        const auto norm = std::sqrt(norm_squared);
        if (norm <= threshold) {
            continue;
        }
        for (auto value : vector) {
            basis.push_back(value / norm);
        }
    }
    return basis;
}

Matrix projector_coordinates(
    const Matrix &basis,
    const Matrix &block,
    std::size_t n,
    std::size_t rank,
    std::size_t count
) {
    const auto coordinates =
        multiply('C', 'N', rank, count, n, basis, n, block, n);
    return multiply(
        'N',
        'C',
        rank,
        rank,
        count,
        coordinates,
        rank,
        coordinates,
        rank
    );
}

Estimate union_estimate(
    const Matrix &midpoint_hamiltonian,
    const Eigensystem &left,
    const Eigensystem &right,
    std::size_t lower,
    std::size_t upper
) {
    const auto n = left.eigenvalues.size();
    const auto count = upper - lower;
    const auto left_block = band_vectors(left, lower, upper);
    const auto right_block = band_vectors(right, lower, upper);
    const auto basis = union_basis(left_block, right_block, n, count);
    const auto rank = basis.size() / n;
    auto interpolated_projector =
        projector_coordinates(basis, left_block, n, rank, count);
    const auto right_projector =
        projector_coordinates(basis, right_block, n, rank, count);
    for (std::size_t index = 0;
         index < interpolated_projector.size();
         ++index) {
        interpolated_projector[index] =
            0.5 * (interpolated_projector[index] + right_projector[index]);
    }

    auto projector_values = std::vector<double>{};
    fermisimplex::linalg::diagonalize_hermitian_in_place(
        interpolated_projector,
        projector_values,
        rank,
        true,
        "union ablation interpolated projector"
    );
    auto selected_coordinates =
        Matrix(rank * count, Complex{0.0, 0.0});
    for (std::size_t column = 0; column < count; ++column) {
        const auto source = rank - count + column;
        for (std::size_t row = 0; row < rank; ++row) {
            selected_coordinates[cm(row, column, rank)] =
                interpolated_projector[cm(row, source, rank)];
        }
    }
    const auto interpolated_vectors = multiply(
        'N',
        'N',
        n,
        count,
        rank,
        basis,
        n,
        selected_coordinates,
        rank
    );
    const auto values = projected_values(
        midpoint_hamiltonian,
        interpolated_vectors,
        n,
        count
    );
    return compare_to_interpolation(
        values,
        left,
        right,
        lower,
        upper,
        rank
    );
}

Estimate exact_estimate(
    const Eigensystem &midpoint,
    const Eigensystem &left,
    const Eigensystem &right,
    std::size_t lower,
    std::size_t upper
) {
    return compare_to_interpolation(
        std::span<const double>(
            midpoint.eigenvalues.data() + lower,
            upper - lower
        ),
        left,
        right,
        lower,
        upper,
        midpoint.eigenvalues.size()
    );
}

double shortfall(const Estimate &estimate, const Estimate &exact) {
    return std::max(
        exact.positive - estimate.positive,
        exact.negative - estimate.negative
    );
}

struct ScanStats {
    std::size_t cases = 0;
    std::size_t anchor_underestimates = 0;
    std::size_t envelope_underestimates = 0;
    std::size_t union_underestimates = 0;
    std::size_t union_fixes_anchor = 0;
    std::size_t union_regressions = 0;
    double worst_anchor_shortfall = 0.0;
    double worst_envelope_shortfall = 0.0;
    double worst_union_shortfall = 0.0;
    double union_rank_total = 0.0;
    std::size_t maximum_union_rank = 0;
};

void add_case(
    ScanStats &stats,
    const Matrix &midpoint_hamiltonian,
    const Eigensystem &left,
    const Eigensystem &right,
    const Eigensystem &midpoint,
    std::size_t lower,
    std::size_t upper
) {
    const auto anchor = anchor_estimate(
        midpoint_hamiltonian,
        left,
        right,
        lower,
        upper
    );
    const auto union_value = union_estimate(
        midpoint_hamiltonian,
        left,
        right,
        lower,
        upper
    );
    const auto envelope = endpoint_envelope_estimate(
        midpoint_hamiltonian,
        left,
        right,
        lower,
        upper
    );
    const auto exact = exact_estimate(
        midpoint,
        left,
        right,
        lower,
        upper
    );
    const auto anchor_shortfall = shortfall(anchor, exact);
    const auto envelope_shortfall = shortfall(envelope, exact);
    const auto union_shortfall = shortfall(union_value, exact);
    const auto anchor_failed = anchor_shortfall > 1e-8;
    const auto union_failed = union_shortfall > 1e-8;
    ++stats.cases;
    stats.anchor_underestimates += anchor_failed;
    stats.envelope_underestimates += envelope_shortfall > 1e-8;
    stats.union_underestimates += union_failed;
    stats.union_fixes_anchor += anchor_failed && !union_failed;
    stats.union_regressions += !anchor_failed && union_failed;
    stats.worst_anchor_shortfall =
        std::max(stats.worst_anchor_shortfall, anchor_shortfall);
    stats.worst_envelope_shortfall =
        std::max(stats.worst_envelope_shortfall, envelope_shortfall);
    stats.worst_union_shortfall =
        std::max(stats.worst_union_shortfall, union_shortfall);
    stats.union_rank_total +=
        static_cast<double>(union_value.union_dimension);
    stats.maximum_union_rank =
        std::max(stats.maximum_union_rank, union_value.union_dimension);
}

void print_stats(const std::string &label, const ScanStats &stats) {
    std::cout
        << label
        << " cases=" << stats.cases
        << " anchor_under=" << stats.anchor_underestimates
        << " envelope_under=" << stats.envelope_underestimates
        << " union_under=" << stats.union_underestimates
        << " union_fixes=" << stats.union_fixes_anchor
        << " union_regressions=" << stats.union_regressions
        << " worst_anchor=" << stats.worst_anchor_shortfall
        << " worst_envelope=" << stats.worst_envelope_shortfall
        << " worst_union=" << stats.worst_union_shortfall
        << " mean_union_rank="
        << stats.union_rank_total / static_cast<double>(stats.cases)
        << " max_union_rank=" << stats.maximum_union_rank
        << "\n";
}

void correctness_scan(
    std::size_t n,
    std::size_t cases,
    double interval_length
) {
    auto generator = std::mt19937_64{0x5eed1234U};
    auto certificate_stats = ScanStats{};
    auto singleton_stats = ScanStats{};
    for (std::size_t index = 0; index < cases; ++index) {
        const auto constant = random_symmetric(generator, n, 0.8);
        const auto slope = random_symmetric(generator, n, 1.6);
        const auto left_h = add_affine(constant, slope, 0.0);
        const auto right_h =
            add_affine(constant, slope, interval_length);
        const auto midpoint_h =
            add_affine(constant, slope, 0.5 * interval_length);
        const auto left = spectrum(left_h, n);
        const auto right = spectrum(right_h, n);
        const auto midpoint = spectrum(midpoint_h, n);

        const auto eigenvalue_spans =
            std::vector<std::span<const double>>{
                left.eigenvalues,
                right.eigenvalues,
            };
        const auto eigenvector_spans =
            std::vector<std::span<const Complex>>{
                left.eigenvectors,
                right.eigenvectors,
            };
        const auto certificate = fermisimplex::certification::certify_simplex(
            eigenvalue_spans,
            eigenvector_spans,
            0.0,
            0.0,
            1e-14
        );
        const auto lower = certificate.occupation_bounds.lower;
        const auto upper = certificate.occupation_bounds.upper;
        if (lower < upper) {
            add_case(
                certificate_stats,
                midpoint_h,
                left,
                right,
                midpoint,
                lower,
                upper
            );
        }

        add_case(
            singleton_stats,
            midpoint_h,
            left,
            right,
            midpoint,
            n / 2,
            n / 2 + 1
        );
    }
    print_stats(
        "occupation-uncertain interval n=" + std::to_string(n) +
            " h=" + std::to_string(interval_length),
        certificate_stats
    );
    print_stats(
        "central singleton n=" + std::to_string(n) +
            " h=" + std::to_string(interval_length),
        singleton_stats
    );
}

template <class Function>
double median_microseconds(std::size_t iterations, Function &&function) {
    auto measurements = std::vector<double>{};
    measurements.reserve(7);
    volatile double sink = 0.0;
    for (std::size_t sample = 0; sample < 7; ++sample) {
        const auto started = std::chrono::steady_clock::now();
        for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
            sink = sink + function();
        }
        const auto stopped = std::chrono::steady_clock::now();
        measurements.push_back(
            std::chrono::duration<double, std::micro>(
                stopped - started
            ).count() /
            static_cast<double>(iterations)
        );
    }
    std::sort(measurements.begin(), measurements.end());
    return measurements[measurements.size() / 2];
}

void timing_scan(std::size_t n, std::size_t count) {
    auto generator = std::mt19937_64{0x71aeU + n + count};
    const auto constant = random_symmetric(generator, n, 0.8);
    const auto slope = random_symmetric(generator, n, 1.6);
    const auto left_h = add_affine(constant, slope, 0.0);
    const auto right_h = add_affine(constant, slope, 1.0);
    const auto midpoint_h = add_affine(constant, slope, 0.5);
    const auto left = spectrum(left_h, n);
    const auto right = spectrum(right_h, n);
    const auto lower = (n - count) / 2;
    const auto upper = lower + count;
    const auto iterations = n <= 8
        ? std::size_t{5000}
        : n <= 32 ? std::size_t{500} : std::size_t{100};

    const auto anchor_us = median_microseconds(iterations, [&] {
        const auto result = anchor_estimate(
            midpoint_h,
            left,
            right,
            lower,
            upper
        );
        return result.negative + result.positive;
    });
    const auto union_us = median_microseconds(iterations, [&] {
        const auto result = union_estimate(
            midpoint_h,
            left,
            right,
            lower,
            upper
        );
        return result.negative + result.positive;
    });
    const auto envelope_us = median_microseconds(iterations, [&] {
        const auto result = endpoint_envelope_estimate(
            midpoint_h,
            left,
            right,
            lower,
            upper
        );
        return result.negative + result.positive;
    });
    const auto lapack_us = median_microseconds(iterations, [&] {
        const auto result = spectrum(midpoint_h, n);
        return result.eigenvalues[n / 2];
    });
    const auto union_rank = union_estimate(
        midpoint_h,
        left,
        right,
        lower,
        upper
    ).union_dimension;
    std::cout
        << "timing n=" << n
        << " target_bands=" << count
        << " union_rank=" << union_rank
        << " anchor_us=" << anchor_us
        << " envelope_us=" << envelope_us
        << " union_us=" << union_us
        << " full_lapack_us=" << lapack_us
        << " union/anchor=" << union_us / anchor_us
        << " union/lapack=" << union_us / lapack_us
        << "\n";
}

int main() {
    std::cout << std::setprecision(8);
    correctness_scan(4, 1000, 1.0);
    correctness_scan(8, 500, 1.0);
    correctness_scan(8, 500, 0.5);
    correctness_scan(8, 500, 0.25);
    correctness_scan(8, 500, 0.125);
    timing_scan(8, 1);
    timing_scan(8, 3);
    timing_scan(32, 1);
    timing_scan(32, 3);
    timing_scan(60, 1);
    timing_scan(60, 3);
}
