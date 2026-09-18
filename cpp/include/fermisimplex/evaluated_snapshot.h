#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace fermisimplex {

// Owning, row-major arrays. Connectivity indexes rows of points/eigenvalues.
// IDs remain stable within one SpectralMesh; provenance is current membership.
struct EvaluatedSnapshot {
    std::size_t ndim = 0;
    std::size_t ndof = 0;
    std::size_t active_vertices = 0;
    std::size_t active_simplices = 0;
    std::vector<std::size_t> vertex_ids;
    std::vector<double> points;
    std::vector<std::int64_t> dyadic_numerators;
    std::vector<std::uint32_t> dyadic_levels;
    std::vector<double> eigenvalues;
    std::optional<std::vector<std::complex<double>>> eigenvectors;
    std::vector<std::uint8_t> vertex_is_active;
    std::vector<std::size_t> simplex_ids;
    std::vector<std::size_t> active_ancestor_ids;
    std::vector<std::size_t> simplices;
    std::vector<double> volumes;
};

}  // namespace fermisimplex
