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
// Sorted vertex energies are its knots. Differentiate the same cumulative
// recurrence, holding tolerance-snapped knots at zero. Distinct off-level
// energies remain distinct, including clustered knots.
class AffineCut {
public:
    struct Integral { double fraction = 0, derivative = 0; };

    AffineCut(std::vector<double> values, double tolerance)
        : knots_(std::move(values)), pin_level_(tolerance > 0) {
        std::sort(knots_.begin(), knots_.end());
        // Geometric clipping treats near-level vertices as lying on the cut.
        // Snap only to the cut, never merge nearby off-level knots.
        kind_ = snap_cut_to_level(knots_, tolerance);
    }

    double fraction() const { return evaluate<false>().fraction; }
    Integral fraction_and_derivative() const { return evaluate<true>(); }

private:
    template <bool Derivative>
    Integral evaluate() const {
        if (kind_ == CutKind::empty) return {};
        if (kind_ == CutKind::on_level) return {.fraction = .5};
        if (kind_ == CutKind::full && (!Derivative || pin_level_ || knots_.back() < 0))
            return {.fraction = 1};
        std::vector<double> fractions(knots_.size());
        std::vector<double> derivatives(Derivative ? knots_.size() : 0);
        for (std::size_t i = 0; i < knots_.size(); ++i)
            fractions[i] = knots_[i] < 0 ? 1. : 0.;
        for (std::size_t order = 1; order < knots_.size(); ++order)
            for (std::size_t i = 0; i + order < knots_.size(); ++i) {
                const auto low = knots_[i], high = knots_[i + order];
                // With zero tolerance, use the left derivative at an exact
                // knot. A positive tolerance instead makes level knots fixed.
                if (high < 0 || (pin_level_ && high == 0)) {
                    fractions[i] = 1;
                    if constexpr (Derivative) derivatives[i] = 0;
                } else if (low >= 0) {
                    fractions[i] = 0;
                    if constexpr (Derivative) derivatives[i] = 0;
                } else {
                    const auto span = high - low;
                    const auto left = -low / span, right = high / span;
                    if constexpr (Derivative)
                        derivatives[i] = left * derivatives[i] + right * derivatives[i + 1] +
                            (fractions[i] - fractions[i + 1]) / span;
                    fractions[i] = left * fractions[i] + right * fractions[i + 1];
                }
            }
        return {std::clamp(fractions.front(), 0., 1.),
                Derivative ? derivatives.front() : 0.};
    }

    std::vector<double> knots_;
    CutKind kind_;
    bool pin_level_;
};

inline double occupied_volume(double volume, std::span<const double> values,
                              double tolerance, CutKind kind) {
    if (kind == CutKind::full) return volume;
    if (kind == CutKind::empty) return 0;
    if (kind == CutKind::on_level) return .5 * volume;
    return volume * AffineCut({values.begin(), values.end()}, tolerance).fraction();
}

}  // namespace fermisimplex::occupation_detail
