#include "certificate/linalg/cholesky.h"
#include "certificate/bounds/occupation_bounds.h"
#include "core/types.h"

#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

using Complex = std::complex<double>;
namespace detail = lineartetrahedron::simplex_certificate::detail;

constexpr double kTol = 1e-12;
using Clock = std::chrono::steady_clock;

double elapsed_us(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::micro>(end - start).count();
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

std::vector<std::vector<Complex>> winding_blocks(bool unoccupied_sector) {
    std::vector<std::vector<Complex>> blocks;
    for (const auto point : {0.0, 0.125, 0.25}) {
        auto block = winding_hamiltonian(3, point);
        const auto value = unoccupied_sector
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

double microseconds_for(
    const std::vector<std::vector<Complex>> &blocks,
    size_t size,
    size_t iterations,
    bool estimate_occupation_rank
) {
    auto checksum = size_t{0};
    const auto start = Clock::now();
    for (size_t iter = 0; iter < iterations; ++iter) {
        for (const auto &block : blocks) {
            auto candidate = block;
            checksum += detail::positive_definite(
                std::move(candidate),
                size,
                detail::certificate_margin(kTol)
            ).passed ? 1 : 0;
        }
        if (estimate_occupation_rank) {
            checksum += detail::estimate_ordered_subset_rank(blocks, size, kTol);
        }
    }
    const auto end = Clock::now();
    if (checksum == 0 && estimate_occupation_rank) {
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
    const auto bounded_us = microseconds_for(blocks, size, iterations, true);
    const auto baseline_each = baseline_us / static_cast<double>(iterations);
    const auto bounded_each = bounded_us / static_cast<double>(iterations);
    const auto rank = detail::estimate_ordered_subset_rank(blocks, size, kTol);
    std::cout
        << "{\"case\":\"" << name
        << "\",\"certificates\":" << iterations
        << ",\"baseline_us_per_certificate\":" << std::setprecision(8) << baseline_each
        << ",\"bounded_us_per_certificate\":" << bounded_each
        << ",\"estimator_extra_us_per_certificate\":" << (bounded_each - baseline_each)
        << ",\"overhead_ratio\":"
        << (baseline_each == 0.0 ? 0.0 : bounded_each / baseline_each)
        << ",\"rank\":" << rank
        << "}\n";
}

}  // namespace

int main() {
    run_case("winding-2-band-unoccupied", winding_blocks(true), 1, 20000);
    run_case("winding-2-band-occupied", winding_blocks(false), 1, 20000);
    run_case("dense-10-band", dense_blocks(10, 10), 10, 4000);
    run_case("dense-60-band", dense_blocks(60, 60), 60, 1000);
    return 0;
}
