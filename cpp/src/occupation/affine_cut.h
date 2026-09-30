#pragma once

#include <algorithm>
#include <cmath>
#include <span>
#include <utility>
#include <vector>

namespace fermisimplex::occupation_detail {

enum class CutKind { empty, full, partial, on_level };

struct CutClassification {
    CutKind kind;
    double epsilon;
};

// Values are energies relative to mu. Use this convention for both constant
// bands and varying affine cuts, independent of the absolute energy origin.
inline CutClassification classify_cut(std::span<const double> values, double tolerance) {
    double scale = 1;
    for (const auto value : values) scale = std::max(scale, std::abs(value));
    const auto epsilon = tolerance * scale;
    bool negative = false, positive = false;
    for (const auto value : values) {
        negative |= value < -epsilon;
        positive |= value > epsilon;
    }
    if (!negative && !positive) return {CutKind::on_level, epsilon};
    if (!positive) return {CutKind::full, epsilon};
    if (!negative) return {CutKind::empty, epsilon};
    return {CutKind::partial, epsilon};
}

// Define the reported affine field once on its root simplex. Restrictions of
// this field use zero level tolerance so subdivision cannot change the cut.
inline CutKind snap_cut_to_level(std::span<double> values, double tolerance) {
    const auto classification = classify_cut(values, tolerance);
    for (auto &value : values)
        if (std::abs(value) <= classification.epsilon) value = 0;
    return classification.kind;
}

// Occupied fraction of a uniform simplex under one affine energy cut.
// Sorted vertex energies are its knots. Both recurrences use nonnegative
// terms and handle repeated knots without merging nearby distinct energies.
class AffineCut {
public:
    AffineCut(std::vector<double> values, double tolerance)
        : knots_(std::move(values)) {
        std::sort(knots_.begin(), knots_.end());
        outside_range_ = knots_.front() >= 0 || knots_.back() < 0;
        // Geometric clipping treats near-level vertices as lying on the cut.
        // Snap only to the cut, never merge nearby off-level knots.
        kind_ = snap_cut_to_level(knots_, tolerance);
    }

    CutKind kind() const { return kind_; }

    double fraction() const {
        if (kind_ == CutKind::full) return 1;
        if (kind_ == CutKind::empty) return 0;
        if (kind_ == CutKind::on_level) return .5;
        std::vector<double> fractions(knots_.size());
        for (std::size_t i = 0; i < knots_.size(); ++i)
            fractions[i] = knots_[i] <= 0 ? 1. : 0.;
        for (std::size_t order = 1; order < knots_.size(); ++order)
            for (std::size_t i = 0; i + order < knots_.size(); ++i) {
                const auto low = knots_[i], high = knots_[i + order];
                if (high <= 0) fractions[i] = 1;
                else if (low >= 0) fractions[i] = 0;
                else fractions[i] = (-low / (high - low)) * fractions[i] +
                                     (high / (high - low)) * fractions[i + 1];
            }
        return std::clamp(fractions.front(), 0., 1.);
    }

    double derivative() const {
        if (kind_ == CutKind::on_level || outside_range_)
            return 0;
        const auto dimension = knots_.size() - 1;
        std::vector<double> basis(dimension);
        // Left derivative at a knot, including the upper endpoint in 1D.
        for (std::size_t i = 0; i < dimension; ++i)
            basis[i] = knots_[i] < 0 && knots_[i + 1] >= 0 ? 1. : 0.;
        for (std::size_t degree = 1; degree < dimension; ++degree)
            for (std::size_t i = 0; i + degree < dimension; ++i) {
                double value = 0;
                if (basis[i] != 0 && knots_[i + degree] > knots_[i])
                    value += (-knots_[i] / (knots_[i + degree] - knots_[i])) * basis[i];
                if (basis[i + 1] != 0 && knots_[i + degree + 1] > knots_[i + 1])
                    value += (knots_[i + degree + 1] /
                        (knots_[i + degree + 1] - knots_[i + 1])) * basis[i + 1];
                basis[i] = value;
            }
        return dimension * basis.front() / (knots_.back() - knots_.front());
    }

private:
    std::vector<double> knots_;
    CutKind kind_;
    bool outside_range_;
};

inline double occupied_volume(double volume, std::span<const double> values,
                              double tolerance, CutKind kind) {
    if (kind == CutKind::full) return volume;
    if (kind == CutKind::empty) return 0;
    if (kind == CutKind::on_level) return .5 * volume;
    return volume * AffineCut({values.begin(), values.end()}, tolerance).fraction();
}

}  // namespace fermisimplex::occupation_detail
