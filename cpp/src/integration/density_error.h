#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace fermisimplex::integration_detail {

// Refined cells replace much larger earlier corrections. Compensated sums
// prevent their cancellation residue from becoming a false refinement floor.
class DensityError {
    struct Sum {
        long double value = 0, correction = 0;

        void add(long double term) {
            const auto next = value + term;
            correction += std::abs(value) >= std::abs(term)
                ? (value - next) + term : (term - next) + value;
            value = next;
        }
        long double total() const { return value + correction; }
    };

    Sum squared_, roundoff_;
    std::vector<Sum> real_, imaginary_;

public:
    explicit DensityError(std::size_t components) : real_(components), imaginary_(components) {}

    template <class Value>
    void update(const Value &change, double roundoff, int sign) {
        const auto error = static_cast<long double>(change.max_abs());
        squared_.add(sign * error * error);
        roundoff_.add(sign * static_cast<long double>(roundoff));
        for (std::size_t i = 0; i < change.size(); ++i) {
            real_[i].add(sign * static_cast<long double>(change[i].real()));
            imaginary_[i].add(sign * static_cast<long double>(change[i].imag()));
        }
    }

    double estimate() const {
        auto error = std::max(std::sqrt(std::max(0.L, squared_.total())), roundoff_.total());
        for (std::size_t i = 0; i < real_.size(); ++i)
            error = std::max(error, std::hypot(real_[i].total(), imaginary_[i].total()));
        return static_cast<double>(error);
    }
};

}  // namespace fermisimplex::integration_detail
