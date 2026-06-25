#include "certificate/internal.h"
#include "core/types.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

using Complex = std::complex<double>;
namespace detail = lineartetrahedron::simplex_certificate::detail;

constexpr double kTol = 1e-12;
using Clock = std::chrono::steady_clock;

struct EstimatorProfile {
    double total_us = 0.0;
    double order_us = 0.0;
    double permute_us = 0.0;
    double prefix_cholesky_us = 0.0;
    double mu_radius_us = 0.0;
    std::vector<double> prefix_cholesky_call_us;
    std::vector<double> prefix_cholesky_certificate_us;
    size_t checksum = 0;
};

struct DistributionStats {
    double mean = 0.0;
    double min = 0.0;
    double p50 = 0.0;
    double p90 = 0.0;
    double p99 = 0.0;
    double max = 0.0;
};

double elapsed_us(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::micro>(end - start).count();
}

DistributionStats distribution_stats(std::vector<double> values) {
    if (values.empty()) {
        return {};
    }
    std::sort(values.begin(), values.end());
    auto sum = 0.0;
    for (const auto value : values) {
        sum += value;
    }
    const auto percentile = [&](double fraction) {
        const auto index = static_cast<size_t>(
            std::min<double>(
                static_cast<double>(values.size() - 1),
                std::floor(fraction * static_cast<double>(values.size() - 1))
            )
        );
        return values[index];
    };
    return DistributionStats{
        .mean = sum / static_cast<double>(values.size()),
        .min = values.front(),
        .p50 = percentile(0.50),
        .p90 = percentile(0.90),
        .p99 = percentile(0.99),
        .max = values.back(),
    };
}

std::vector<Complex> hermitian_2x2(double a, double b, double c) {
    return {
        Complex{a, 0.0},
        Complex{b, 0.0},
        Complex{b, 0.0},
        Complex{c, 0.0},
    };
}

std::vector<Complex> winding_hamiltonian(int winding, double reduced_k) {
    const auto phase = 2.0 * lineartetrahedron::kPi * static_cast<double>(winding) * reduced_k;
    const auto c = std::cos(phase);
    const auto s = std::sin(phase);
    return hermitian_2x2(c, s, -c);
}

std::vector<std::vector<Complex>> winding_blocks(bool positive_side) {
    std::vector<std::vector<Complex>> blocks;
    for (const auto point : {0.0, 0.125, 0.25}) {
        auto block = winding_hamiltonian(3, point);
        const auto value = positive_side
            ? std::real(block[detail::column_major_index(0, 0, 2)])
            : -std::real(block[detail::column_major_index(1, 1, 2)]);
        blocks.push_back({Complex{value, 0.0}});
    }
    return blocks;
}

std::vector<std::vector<Complex>> dense_blocks(size_t size, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::vector<std::vector<Complex>> blocks;
    for (size_t block_index = 0; block_index < 3; ++block_index) {
        std::vector<Complex> block(size * size, Complex{0.0, 0.0});
        for (size_t column = 0; column < size; ++column) {
            for (size_t row = column; row < size; ++row) {
                if (row == column) {
                    const auto value =
                        1.5 + 0.02 * static_cast<double>(column) + 0.01 * normal(rng);
                    block[detail::column_major_index(row, column, size)] = Complex{value, 0.0};
                } else {
                    const auto scale = 0.04 / std::sqrt(static_cast<double>(size));
                    const auto value = scale * Complex{normal(rng), normal(rng)};
                    block[detail::column_major_index(row, column, size)] = value;
                    block[detail::column_major_index(column, row, size)] = std::conj(value);
                }
            }
        }
        blocks.push_back(std::move(block));
    }
    return blocks;
}

std::vector<size_t> candidate_order_by_worst_margin(
    const std::vector<std::vector<Complex>> &blocks,
    size_t size
) {
    std::vector<std::pair<double, size_t>> margins;
    margins.reserve(size);
    for (size_t index = 0; index < size; ++index) {
        auto margin = std::numeric_limits<double>::infinity();
        for (const auto &block : blocks) {
            margin = std::min(
                margin,
                std::real(block[detail::column_major_index(index, index, size)])
            );
        }
        margins.emplace_back(margin, index);
    }

    std::sort(
        margins.begin(),
        margins.end(),
        [](const auto &left, const auto &right) {
            if (left.first != right.first) {
                return left.first > right.first;
            }
            return left.second < right.second;
        }
    );

    std::vector<size_t> order;
    order.reserve(size);
    for (const auto &[unused_margin, index] : margins) {
        (void)unused_margin;
        order.push_back(index);
    }
    return order;
}

std::vector<Complex> permuted_block(
    const std::vector<Complex> &block,
    const std::vector<size_t> &order,
    size_t size
) {
    std::vector<Complex> result(size * size, Complex{0.0, 0.0});
    for (size_t column = 0; column < size; ++column) {
        const auto source_column = order[column];
        for (size_t row = 0; row < size; ++row) {
            result[detail::column_major_index(row, column, size)] =
                block[detail::column_major_index(order[row], source_column, size)];
        }
    }
    return result;
}

Complex hermitian_entry(
    const std::vector<Complex> &matrix,
    size_t size,
    size_t row,
    size_t column
) {
    if (row >= column) {
        return matrix[detail::column_major_index(row, column, size)];
    }
    return std::conj(matrix[detail::column_major_index(column, row, size)]);
}

double leading_prefix_mu_radius(
    const std::vector<std::vector<Complex>> &permuted_blocks,
    size_t size,
    size_t rank
) {
    if (rank == 0) {
        return std::numeric_limits<double>::infinity();
    }

    auto radius = std::numeric_limits<double>::infinity();
    const auto margin = detail::positive_definite_margin(kTol);
    for (const auto &block : permuted_blocks) {
        for (size_t row = 0; row < rank; ++row) {
            auto row_bound = std::real(block[detail::column_major_index(row, row, size)]);
            for (size_t column = 0; column < rank; ++column) {
                if (column != row) {
                    row_bound -= std::abs(hermitian_entry(block, size, row, column));
                }
            }
            radius = std::min(radius, row_bound - margin);
        }
    }
    return std::max(0.0, radius);
}

size_t estimate_common_rank_only(
    const std::vector<std::vector<Complex>> &blocks,
    size_t size
) {
    const auto order = candidate_order_by_worst_margin(blocks, size);
    auto rank = size;
    for (const auto &block : blocks) {
        rank = std::min(
            rank,
            detail::positive_definite_prefix(permuted_block(block, order, size), size, kTol)
        );
    }
    return rank;
}

EstimatorProfile profile_estimator(
    const std::vector<std::vector<Complex>> &blocks,
    size_t size,
    size_t iterations
) {
    auto profile = EstimatorProfile{};
    profile.prefix_cholesky_call_us.reserve(iterations * blocks.size());
    profile.prefix_cholesky_certificate_us.reserve(iterations);
    for (size_t iter = 0; iter < iterations; ++iter) {
        const auto total_start = Clock::now();

        const auto order_start = Clock::now();
        const auto order = candidate_order_by_worst_margin(blocks, size);
        const auto order_end = Clock::now();
        profile.order_us += elapsed_us(order_start, order_end);

        auto rank = size;
        auto certificate_prefix_cholesky_us = 0.0;
        std::vector<std::vector<Complex>> permuted_blocks;
        permuted_blocks.reserve(blocks.size());
        for (const auto &block : blocks) {
            const auto permute_start = Clock::now();
            auto permuted = permuted_block(block, order, size);
            const auto permute_end = Clock::now();
            profile.permute_us += elapsed_us(permute_start, permute_end);

            const auto prefix_start = Clock::now();
            rank = std::min(
                rank,
                detail::positive_definite_prefix(permuted, size, kTol)
            );
            const auto prefix_end = Clock::now();
            const auto prefix_us = elapsed_us(prefix_start, prefix_end);
            profile.prefix_cholesky_us += prefix_us;
            certificate_prefix_cholesky_us += prefix_us;
            profile.prefix_cholesky_call_us.push_back(prefix_us);
            permuted_blocks.push_back(std::move(permuted));
        }
        profile.prefix_cholesky_certificate_us.push_back(certificate_prefix_cholesky_us);

        const auto mu_radius_start = Clock::now();
        const auto mu_radius = leading_prefix_mu_radius(permuted_blocks, size, rank);
        const auto mu_radius_end = Clock::now();
        profile.mu_radius_us += elapsed_us(mu_radius_start, mu_radius_end);

        const auto total_end = Clock::now();
        profile.total_us += elapsed_us(total_start, total_end);
        profile.checksum += rank;
        if (mu_radius < 0.0) {
            std::cerr << "";
        }
    }
    return profile;
}

double microseconds_for(
    const std::vector<std::vector<Complex>> &blocks,
    size_t size,
    size_t iterations,
    bool estimate_common_rank
) {
    auto checksum = size_t{0};
    const auto start = Clock::now();
    for (size_t iter = 0; iter < iterations; ++iter) {
        for (const auto &block : blocks) {
            auto candidate = block;
            checksum += detail::positive_definite(std::move(candidate), size, kTol) ? 1 : 0;
        }
        if (estimate_common_rank) {
            checksum += detail::estimate_common_rank(blocks, size, kTol);
        }
    }
    const auto end = Clock::now();
    if (checksum == 0 && estimate_common_rank) {
        std::cerr << "";
    }
    return elapsed_us(start, end);
}

double microseconds_for_rank_only_bounds(
    const std::vector<std::vector<Complex>> &blocks,
    size_t size,
    size_t iterations
) {
    auto checksum = size_t{0};
    const auto start = Clock::now();
    for (size_t iter = 0; iter < iterations; ++iter) {
        for (const auto &block : blocks) {
            auto candidate = block;
            checksum += detail::positive_definite(std::move(candidate), size, kTol) ? 1 : 0;
        }
        checksum += estimate_common_rank_only(blocks, size);
    }
    const auto end = Clock::now();
    if (checksum == 0) {
        std::cerr << "";
    }
    return elapsed_us(start, end);
}

void run_case(
    const std::string &name,
    const std::vector<std::vector<Complex>> &blocks,
    size_t size,
    size_t iterations
) {
    const auto baseline_us = microseconds_for(blocks, size, iterations, false);
    const auto rank_only_bounded_us =
        microseconds_for_rank_only_bounds(blocks, size, iterations);
    const auto bounded_us = microseconds_for(blocks, size, iterations, true);
    const auto baseline_each = baseline_us / static_cast<double>(iterations);
    const auto rank_only_bounded_each =
        rank_only_bounded_us / static_cast<double>(iterations);
    const auto bounded_each = bounded_us / static_cast<double>(iterations);
    const auto rank = detail::estimate_common_rank(blocks, size, kTol);
    const auto profile = profile_estimator(blocks, size, iterations);
    if (profile.checksum == 0 && rank != 0) {
        std::cerr << "";
    }
    const auto profile_scale = 1.0 / static_cast<double>(iterations);
    const auto profiled_estimator_each = profile.total_us * profile_scale;
    const auto order_each = profile.order_us * profile_scale;
    const auto permute_each = profile.permute_us * profile_scale;
    const auto prefix_cholesky_each = profile.prefix_cholesky_us * profile_scale;
    const auto mu_radius_each = profile.mu_radius_us * profile_scale;
    const auto unaccounted_each =
        profiled_estimator_each - order_each - permute_each - prefix_cholesky_each -
        mu_radius_each;
    const auto prefix_call_stats =
        distribution_stats(profile.prefix_cholesky_call_us);
    const auto prefix_certificate_stats =
        distribution_stats(profile.prefix_cholesky_certificate_us);
    std::cout
        << "{\"case\":\"" << name
        << "\",\"certificates\":" << iterations
        << ",\"baseline_us_per_certificate\":" << std::setprecision(8) << baseline_each
        << ",\"rank_only_bounded_us_per_certificate\":" << rank_only_bounded_each
        << ",\"bounded_us_per_certificate\":" << bounded_each
        << ",\"rank_only_estimator_extra_us_per_certificate\":"
        << (rank_only_bounded_each - baseline_each)
        << ",\"estimator_extra_us_per_certificate\":" << (bounded_each - baseline_each)
        << ",\"mu_bounds_extra_vs_rank_only_us_per_certificate\":"
        << (bounded_each - rank_only_bounded_each)
        << ",\"profiled_estimator_us_per_certificate\":" << profiled_estimator_each
        << ",\"order_us_per_certificate\":" << order_each
        << ",\"permute_us_per_certificate\":" << permute_each
        << ",\"prefix_cholesky_us_per_certificate\":" << prefix_cholesky_each
        << ",\"mu_radius_us_per_certificate\":" << mu_radius_each
        << ",\"prefix_cholesky_call_mean_us\":" << prefix_call_stats.mean
        << ",\"prefix_cholesky_call_min_us\":" << prefix_call_stats.min
        << ",\"prefix_cholesky_call_p50_us\":" << prefix_call_stats.p50
        << ",\"prefix_cholesky_call_p90_us\":" << prefix_call_stats.p90
        << ",\"prefix_cholesky_call_p99_us\":" << prefix_call_stats.p99
        << ",\"prefix_cholesky_call_max_us\":" << prefix_call_stats.max
        << ",\"prefix_cholesky_certificate_mean_us\":" << prefix_certificate_stats.mean
        << ",\"prefix_cholesky_certificate_min_us\":" << prefix_certificate_stats.min
        << ",\"prefix_cholesky_certificate_p50_us\":" << prefix_certificate_stats.p50
        << ",\"prefix_cholesky_certificate_p90_us\":" << prefix_certificate_stats.p90
        << ",\"prefix_cholesky_certificate_p99_us\":" << prefix_certificate_stats.p99
        << ",\"prefix_cholesky_certificate_max_us\":" << prefix_certificate_stats.max
        << ",\"profile_unaccounted_us_per_certificate\":" << unaccounted_each
        << ",\"overhead_ratio\":"
        << (baseline_each == 0.0 ? 0.0 : bounded_each / baseline_each)
        << ",\"mu_bounds_overhead_ratio_vs_rank_only\":"
        << (
            rank_only_bounded_each == 0.0
                ? 0.0
                : bounded_each / rank_only_bounded_each
        )
        << ",\"rank\":" << rank
        << "}\n";
}

}  // namespace

int main() {
    run_case("winding-2-band-plus", winding_blocks(true), 1, 20000);
    run_case("winding-2-band-minus", winding_blocks(false), 1, 20000);
    run_case("dense-10-band", dense_blocks(10, 10), 10, 4000);
    run_case("dense-60-band", dense_blocks(60, 60), 60, 1000);
    return 0;
}
