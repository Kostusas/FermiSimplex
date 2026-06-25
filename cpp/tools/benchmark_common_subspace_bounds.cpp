#include "certificate/internal.h"
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

std::vector<Complex> negated(std::vector<Complex> matrix) {
    for (auto &value : matrix) {
        value = -value;
    }
    return matrix;
}

std::vector<std::vector<Complex>> winding_blocks(bool positive_side) {
    std::vector<std::vector<Complex>> blocks;
    for (const auto point : {0.0, 0.125, 0.25}) {
        auto block = winding_hamiltonian(3, point);
        blocks.push_back(positive_side ? std::move(block) : negated(std::move(block)));
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
    bool enabled
) {
    auto checksum = size_t{0};
    const auto start = std::chrono::steady_clock::now();
    for (size_t iter = 0; iter < iterations; ++iter) {
        for (const auto &block : blocks) {
            auto candidate = block;
            checksum += detail::positive_definite(std::move(candidate), size, kTol) ? 1 : 0;
        }
        if (enabled) {
            checksum += detail::estimate_common_rank(blocks, size, kTol);
        }
    }
    const auto end = std::chrono::steady_clock::now();
    if (checksum == 0 && enabled) {
        std::cerr << "";
    }
    return std::chrono::duration<double, std::micro>(end - start).count();
}

void run_case(
    const std::string &name,
    const std::vector<std::vector<Complex>> &blocks,
    size_t size,
    size_t iterations
) {
    const auto off_us = microseconds_for(blocks, size, iterations, false);
    const auto on_us = microseconds_for(blocks, size, iterations, true);
    const auto off_each = off_us / static_cast<double>(iterations);
    const auto on_each = on_us / static_cast<double>(iterations);
    std::cout
        << "{\"case\":\"" << name
        << "\",\"certificates\":" << iterations
        << ",\"off_us_per_certificate\":" << std::setprecision(8) << off_each
        << ",\"on_us_per_certificate\":" << on_each
        << ",\"overhead_ratio\":" << (off_each == 0.0 ? 0.0 : on_each / off_each)
        << "}\n";
}

}  // namespace

int main() {
    run_case("winding-2-band-plus", winding_blocks(true), 2, 20000);
    run_case("winding-2-band-minus", winding_blocks(false), 2, 20000);
    run_case("dense-10-band", dense_blocks(10, 10), 10, 4000);
    run_case("dense-60-band", dense_blocks(60, 60), 60, 200);
    return 0;
}
