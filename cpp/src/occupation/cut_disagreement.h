#pragma once

#include "occupation/affine_cut.h"

#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

namespace fermisimplex::occupation_detail {

// Partition a simplex at the first affine cut. Each resulting inside simplex
// still carries affine values for the second cut, so its occupied volume is
// available from the ordinary one-cut rule.
struct TwoCutVertex {
    double first;
    double second;
};

inline double intersection_volume(
    std::vector<TwoCutVertex> vertices,
    double volume,
    double tolerance
) {
    const auto scale = std::max(1.0, [&] {
        auto maximum = 0.0;
        for (const auto &vertex : vertices) {
            maximum = std::max(maximum, std::abs(vertex.first));
        }
        return maximum;
    }());
    const auto epsilon = tolerance * scale;
    std::size_t negative = vertices.size();
    std::size_t positive = vertices.size();
    for (std::size_t i = 0; i < vertices.size(); ++i) {
        if (vertices[i].first < -epsilon && negative == vertices.size()) {
            negative = i;
        }
        if (vertices[i].first > epsilon && positive == vertices.size()) {
            positive = i;
        }
    }
    if (positive == vertices.size()) {
        auto values = std::vector<double>{};
        values.reserve(vertices.size());
        for (const auto &vertex : vertices) {
            values.push_back(vertex.second);
        }
        return volume * AffineCut(std::move(values), tolerance).fraction();
    }
    if (negative == vertices.size()) {
        return 0.0;
    }

    const auto a = vertices[negative];
    const auto b = vertices[positive];
    const auto fraction = a.first / (a.first - b.first);
    const auto crossing = TwoCutVertex{
        .first = 0.0,
        .second = (1.0 - fraction) * a.second + fraction * b.second,
    };
    vertices[positive] = crossing;
    const auto inside = intersection_volume(
        vertices, fraction * volume, tolerance
    );
    vertices[positive] = b;
    vertices[negative] = crossing;
    return inside + intersection_volume(
        vertices, (1.0 - fraction) * volume, tolerance
    );
}

inline double cut_disagreement(
    double volume,
    std::span<const double> first,
    std::span<const double> second,
    double tolerance
) {
    const auto first_kind = classify_cut(first, tolerance).kind;
    const auto second_kind = classify_cut(second, tolerance).kind;
    if (first_kind == CutKind::on_level ||
        second_kind == CutKind::on_level) {
        return first_kind == CutKind::on_level &&
            second_kind == CutKind::on_level
            ? 0.0 : 0.5 * volume;
    }
    if (first_kind == CutKind::empty) {
        return occupied_volume(volume, second, tolerance, second_kind);
    }
    if (second_kind == CutKind::empty) {
        return occupied_volume(volume, first, tolerance, first_kind);
    }
    if (first_kind == CutKind::full) {
        return volume - occupied_volume(
            volume, second, tolerance, second_kind
        );
    }
    if (second_kind == CutKind::full) {
        return volume - occupied_volume(
            volume, first, tolerance, first_kind
        );
    }

    const auto first_volume = occupied_volume(
        volume, first, tolerance, first_kind
    );
    const auto second_volume = occupied_volume(
        volume, second, tolerance, second_kind
    );

    auto second_above = true;
    auto second_below = true;
    for (std::size_t i = 0; i < first.size(); ++i) {
        second_above &= second[i] >= first[i];
        second_below &= second[i] <= first[i];
    }
    // When the affine difference has one sign, the occupied sets are nested.
    if (second_above || second_below) {
        return std::abs(first_volume - second_volume);
    }

    auto vertices = std::vector<TwoCutVertex>{};
    vertices.reserve(first.size());
    for (std::size_t i = 0; i < first.size(); ++i) {
        vertices.push_back({first[i], second[i]});
    }
    const auto both = intersection_volume(std::move(vertices), volume, tolerance);
    return std::clamp(
        first_volume + second_volume - 2.0 * both, 0.0, volume
    );
}

}  // namespace fermisimplex::occupation_detail
