#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace adaptivesimplex::core {

class DyadicVertex {
public:
    DyadicVertex(std::vector<std::int64_t> coords, std::uint32_t level)
        : coords_(std::move(coords)), level_(level) {
            if (level_ >= kMaxLevel) {
                throw std::runtime_error("DyadicVertex: dyadic level out of range");
            }    
            canonicalize();
    }

    std::span<const std::int64_t> coords() const noexcept { return coords_; }
    std::uint32_t level() const noexcept { return level_; }

    long double squared_distance_to(const DyadicVertex &other) const {
        if (coords_.size() != other.coords_.size()) {
            throw std::runtime_error("DyadicVertex: distance dimension mismatch");
        }

        const auto level = std::max(level_, other.level_);
        auto distance = 0.0L;
        for (auto axis = size_t{0}; axis < coords_.size(); ++axis) {
            const auto left_coord = checked_scaled_coord(coords_[axis], level - level_);
            const auto right_coord = checked_scaled_coord(other.coords_[axis], level - other.level_);
            const auto delta =
                static_cast<long double>(left_coord) - static_cast<long double>(right_coord);
            distance += delta * delta;
        }
        return std::ldexp(distance, -static_cast<int>(2 * level));
    }

    std::vector<double> to_point() const {
        std::vector<double> values(coords_.size());
        for (size_t axis = 0; axis < coords_.size(); ++axis) {
            values[axis] = std::ldexp(static_cast<double>(coords_[axis]), -static_cast<int>(level_));
        }
        return values;
    }

    static DyadicVertex midpoint(const DyadicVertex &left, const DyadicVertex &right) {
        if (left.coords_.size() != right.coords_.size()) {
            throw std::runtime_error("DyadicVertex: midpoint dimension mismatch");
        }
        const auto level = std::max(left.level_, right.level_);

        std::vector<std::int64_t> coords(left.coords_.size());
        for (auto axis = size_t{0}; axis < coords.size(); ++axis) {
            const auto left_coord = checked_scaled_coord(left.coords_[axis], level - left.level_);
            const auto right_coord = checked_scaled_coord(right.coords_[axis], level - right.level_);
            coords[axis] = checked_add(left_coord, right_coord);
        }
        return DyadicVertex(std::move(coords), level + 1);
    }

    bool operator==(const DyadicVertex &other) const {
        return level_ == other.level_ && coords_ == other.coords_;
    }

    struct Hash {
        size_t operator()(const DyadicVertex &vertex) const noexcept {
            size_t seed = std::hash<std::uint32_t>{}(vertex.level_);
            for (const auto coord : vertex.coords_) {
                seed ^= std::hash<std::int64_t>{}(coord) +
                        0x9e3779b97f4a7c15ULL +
                        (seed << 6) +
                        (seed >> 2);
            }
            return seed;
        }
    };

private:
    static constexpr std::uint32_t kMaxLevel = 62;

    static std::int64_t checked_power_of_two(std::uint32_t exponent) {
        if (exponent >= kMaxLevel) {
            throw std::runtime_error("DyadicVertex: dyadic level out of range");
        }
        return std::int64_t{1} << exponent;
    }

    static std::int64_t checked_scaled_coord(std::int64_t coord, std::uint32_t shift) {
        const std::int64_t scale = checked_power_of_two(shift);
        if (coord > 0 && coord > std::numeric_limits<std::int64_t>::max() / scale) {
            throw std::runtime_error("DyadicVertex: dyadic coordinate overflow");
        }
        if (coord < 0 && coord < std::numeric_limits<std::int64_t>::min() / scale) {
            throw std::runtime_error("DyadicVertex: dyadic coordinate overflow");
        }
        return coord * scale;
    }

    static std::int64_t checked_add(std::int64_t left, std::int64_t right) {
        if (right > 0 && left > std::numeric_limits<std::int64_t>::max() - right) {
            throw std::runtime_error("DyadicVertex: dyadic coordinate overflow");
        }
        if (right < 0 && left < std::numeric_limits<std::int64_t>::min() - right) {
            throw std::runtime_error("DyadicVertex: dyadic coordinate overflow");
        }
        return left + right;
    }

    static bool all_coords_even(const std::vector<std::int64_t> &coords) {
        for (const auto coord : coords) {
            if (coord % 2 != 0) {
                return false;
            }
        }
        return true;
    }

    void canonicalize() {
        while (level_ > 0 && all_coords_even(coords_)) {
            --level_;
            for (auto &coord : coords_) {
                coord /= 2;
            }
        }
    }

    std::vector<std::int64_t> coords_;
    std::uint32_t level_ = 0;
};

}  // namespace adaptivesimplex::core
