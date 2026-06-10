#pragma once

#include "lineartetrahedron/vertex_spectra.h"

#include <adaptivesimplex/core/geometry.h>
#include <adaptivesimplex/core/vertex_cache.h>

#include <cstddef>
#include <span>
#include <vector>

namespace lineartetrahedron {

struct BandWeights {
    double charge = 0.0;
    double derivative = 0.0;
    std::vector<double> vertex_weights;
};

struct BandWeightView {
    double charge = 0.0;
    double derivative = 0.0;
    std::span<const double> vertex_weights;
};

struct BandWeightScratch {
    std::vector<double> vertex_weights;
    std::vector<size_t> order;
    std::vector<double> sorted_energies;
};

std::vector<const VertexSpectra *> gather_vertex_spectra(
    const adaptivesimplex::core::Geometry &geometry,
    const adaptivesimplex::core::VertexCache<VertexSpectra> &cache,
    adaptivesimplex::core::SimplexId simplex_id
);

BandWeights band_weights_on_simplex(
    const std::vector<const VertexSpectra *> &entries,
    size_t band,
    double mu,
    double volume,
    size_t ndim,
    double tol
);

BandWeightView band_weights_on_simplex(
    const std::vector<const VertexSpectra *> &entries,
    size_t band,
    double mu,
    double volume,
    size_t ndim,
    double tol,
    BandWeightScratch &scratch
);

}  // namespace lineartetrahedron
