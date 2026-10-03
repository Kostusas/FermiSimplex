#include "occupation/model.h"

#include <limits>
#include <numeric>

namespace fermisimplex::occupation_detail {

Sectors sign_sectors(const Polynomial &polynomial, double allowance,
                     ChargeErrorStats &stats) {
    const auto n = polynomial.size;
    if (n == 0) return {};
    const auto sector_indices = [&](std::size_t count, bool negative) {
        std::vector<std::size_t> sector(count);
        std::iota(sector.begin(), sector.end(), negative ? 0 : n - count);
        // Cholesky must visit the most positive states first for a suffix.
        if (!negative) std::reverse(sector.begin(), sector.end());
        return sector;
    };
    const auto row_margin = [&](const Matrix &control, std::size_t count,
                                bool negative) {
        const auto first = negative ? 0 : n - count;
        std::vector<double> lower(count);
        for (std::size_t row = 0; row < count; ++row)
            lower[row] = (negative ? -1. : 1.) *
                control[first + row + (first + row) * n].real() - allowance;
        for (std::size_t col = 0; col < count; ++col)
            for (std::size_t row = 0; row < count; ++row)
                if (row != col)
                    lower[row] -= std::abs(control[first + row + (first + col) * n]);
        return lower.empty() ? std::numeric_limits<double>::infinity() :
            *std::min_element(lower.begin(), lower.end());
    };
    const auto largest = [&](bool negative, std::size_t maximum) {
        // Definiteness requires every diagonal to have the right sign.
        for (const auto &control : polynomial.controls)
            for (std::size_t rank = 0; rank < maximum; ++rank) {
                const auto band = negative ? rank : n - 1 - rank;
                const auto diagonal = control[band + band * n].real();
                if ((negative ? -diagonal : diagonal) <= allowance) {
                    maximum = rank;
                    break;
                }
            }
        auto margin = std::numeric_limits<double>::infinity();
        bool needs_margin = false;
        for (const auto &control : polynomial.controls) {
            const auto lower = row_margin(control, maximum, negative);
            if (lower > 0) {
                margin = std::min(margin, lower);
                continue;
            }
            auto sector = sector_indices(maximum, negative);
            auto matrix = block(control, n, sector, sector);
            if (negative) for (auto &value : matrix) value = -value;
            for (std::size_t k = 0; k < maximum; ++k)
                matrix[k + k * maximum] -= allowance;
            const auto failure = linalg::cholesky_factor_lower(matrix.data(), maximum);
            // A failed pivot k leaves exactly k-1 positive leading states.
            // Earlier controls stay definite on this smaller principal block.
            if (failure > 0) maximum = static_cast<std::size_t>(failure - 1);
            needs_margin = true;
        }
        // The common Gershgorin case needs only one pass. After any factorization,
        // establish the final margin on the retained block, at most once/control.
        if (needs_margin) {
            margin = std::numeric_limits<double>::infinity();
            const auto sector = sector_indices(maximum, negative);
            for (const auto &control : polynomial.controls) {
                auto lower = row_margin(control, maximum, negative);
                if (lower <= 0) {
                    auto matrix = block(control, n, sector, sector);
                    if (negative) for (auto &value : matrix) value = -value;
                    std::vector<double> eigenvalues;
                    linalg::diagonalize_hermitian_in_place(matrix, eigenvalues,
                        maximum, false, "occupation sector margin");
                    ++stats.norm_eigensystems;
                    lower = eigenvalues.front() - allowance;
                }
                margin = std::min(margin, lower);
            }
        }
        return std::pair{maximum, margin};
    };
    const auto [negative, negative_gap] = largest(true, n);
    const auto [positive, positive_gap] = largest(false, n - negative);
    const auto gap = std::min(negative_gap, positive_gap);
    // A near-singular Cholesky test and an eigenvalue-based margin can disagree
    // at roundoff. Only a positive final margin permits a safe-block inverse.
    if (!std::isfinite(gap) || gap <= 0) return {};
    return {negative, positive, gap, negative_gap, positive_gap};
}

}  // namespace fermisimplex::occupation_detail
