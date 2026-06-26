#include "certificate/linalg/cholesky.h"

#include "certificate/linalg/lapack.h"

#include <algorithm>

namespace lineartetrahedron::simplex_certificate::detail {
namespace {

constexpr double kBlockMargin = 1e-10;

}  // namespace

double certificate_margin(double tol) {
    return std::max(kBlockMargin, tol);
}

PositiveDefiniteResult positive_definite(
    std::vector<Complex> block,
    size_t size,
    double margin
) {
    if (size == 0) {
        return PositiveDefiniteResult{.passed = true, .accepted_rank = 0};
    }

    for (size_t index = 0; index < size; ++index) {
        block[column_major_index(index, index, size)] =
            Complex{std::real(block[column_major_index(index, index, size)]), 0.0};
        block[column_major_index(index, index, size)] -= margin;
    }

    const auto info = potrf(block.data(), size);
    if (info == 0) {
        return PositiveDefiniteResult{.passed = true, .accepted_rank = size};
    }
    return PositiveDefiniteResult{
        .passed = false,
        .accepted_rank = static_cast<size_t>(info - 1),
    };
}

}  // namespace lineartetrahedron::simplex_certificate::detail
