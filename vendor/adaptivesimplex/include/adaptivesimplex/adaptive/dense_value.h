#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace adaptivesimplex::adaptive {

template <class T>
class DenseValue {
public:
    DenseValue() = default;

    explicit DenseValue(size_t size)
        : values_(size, T{}) {}

    explicit DenseValue(std::vector<T> values)
        : values_(std::move(values)) {}

    size_t size() const noexcept { return values_.size(); }
    const std::vector<T> &values() const noexcept { return values_; }

    T &operator[](size_t index) noexcept { return values_[index]; }
    const T &operator[](size_t index) const noexcept { return values_[index]; }

    DenseValue &operator+=(const DenseValue &other) {
        for (size_t index = 0; index < values_.size(); ++index) {
            values_[index] += other.values_[index];
        }
        return *this;
    }

    DenseValue &operator-=(const DenseValue &other) {
        for (size_t index = 0; index < values_.size(); ++index) {
            values_[index] -= other.values_[index];
        }
        return *this;
    }

    double max_abs() const {
        double result = 0.0;
        for (const auto &value : values_) {
            result = std::max(result, std::abs(value));
        }
        return result;
    }

private:
    std::vector<T> values_;
};

}  // namespace adaptivesimplex::adaptive
