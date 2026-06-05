#include "lineartetrahedron/band_weights.h"

#include "lineartetrahedron/simplex_rules.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>

namespace lineartetrahedron {
namespace core = adaptivesimplex::core;

namespace {

enum class OccupancyClass : std::uint8_t { Empty, Full, Half, Cut };

std::vector<size_t> sorted_band_order(
    const std::vector<const VertexSpectra *> &entries,
    size_t band
) {
    std::vector<size_t> order(entries.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::stable_sort(
        order.begin(),
        order.end(),
        [&](size_t left, size_t right) {
            return entries[left]->eigenvalues[band] < entries[right]->eigenvalues[band];
        }
    );
    return order;
}

bool strictly_ordered(
    const std::vector<const VertexSpectra *> &entries,
    const std::vector<size_t> &order,
    size_t band,
    double tol
) {
    for (size_t pos = 1; pos < order.size(); ++pos) {
        const double curr = entries[order[pos]]->eigenvalues[band];
        const double prev = entries[order[pos - 1]]->eigenvalues[band];
        if (curr - prev <= tol) {
            return false;
        }
    }
    return true;
}

OccupancyClass classify_band(
    const std::vector<const VertexSpectra *> &entries,
    const std::vector<size_t> &order,
    size_t band,
    double mu,
    double tol
) {
    const double band_min = entries[order.front()]->eigenvalues[band];
    const double band_max = entries[order.back()]->eigenvalues[band];
    bool half_mask = true;
    for (const auto local_vertex : order) {
        if (std::abs(entries[local_vertex]->eigenvalues[band] - mu) > tol) {
            half_mask = false;
            break;
        }
    }

    if (band_min > mu + tol) {
        return OccupancyClass::Empty;
    }
    if (half_mask) {
        return OccupancyClass::Half;
    }
    if (band_max <= mu + tol) {
        return OccupancyClass::Full;
    }
    return OccupancyClass::Cut;
}

}  // namespace

std::vector<const VertexSpectra *> gather_vertex_spectra(
    const core::Geometry &geometry,
    const core::VertexCache<VertexSpectra> &cache,
    core::SimplexId simplex_id
) {
    const auto &simplex = geometry.simplices().simplex(simplex_id);
    std::vector<const VertexSpectra *> entries;
    entries.reserve(simplex.vertex_ids.size());
    for (const auto vertex_id : simplex.vertex_ids) {
        entries.push_back(&cache.get(vertex_id));
    }
    return entries;
}

BandWeights band_weights_on_simplex(
    const std::vector<const VertexSpectra *> &entries,
    size_t band,
    double mu,
    double volume,
    size_t ndim,
    double tol
) {
    const size_t n_vertices = ndim + 1;
    BandWeights result;
    result.vertex_weights.assign(n_vertices, 0.0);
    const auto order = sorted_band_order(entries, band);
    const auto classification = classify_band(entries, order, band, mu, tol);
    if (classification == OccupancyClass::Empty) {
        return result;
    }

    if (classification == OccupancyClass::Full || classification == OccupancyClass::Half) {
        const double factor = classification == OccupancyClass::Half ? 0.5 : 1.0;
        result.charge = factor * volume;
        std::fill(
            result.vertex_weights.begin(),
            result.vertex_weights.end(),
            factor * volume / static_cast<double>(n_vertices)
        );
        return result;
    }

    std::vector<double> sorted_energies(n_vertices, 0.0);
    for (size_t pos = 0; pos < n_vertices; ++pos) {
        sorted_energies[pos] = entries[order[pos]]->eigenvalues[band];
    }

    const auto fractions = simplex_rules::occupied_linear_moment_fractions(
        sorted_energies.data(),
        ndim,
        mu,
        tol
    );
    for (size_t pos = 0; pos < n_vertices; ++pos) {
        result.vertex_weights[order[pos]] = volume * fractions[pos];
    }

    if (strictly_ordered(entries, order, band, tol)) {
        const auto [fraction, derivative] = simplex_rules::simplex_fraction_and_derivative(
            sorted_energies.data(),
            mu,
            ndim,
            tol
        );
        result.charge = volume * fraction;
        result.derivative = volume * derivative;
    } else {
        result.charge = std::accumulate(
            result.vertex_weights.begin(),
            result.vertex_weights.end(),
            0.0
        );
    }
    return result;
}

}  // namespace lineartetrahedron
