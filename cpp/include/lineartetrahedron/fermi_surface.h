#pragma once

#include "lineartetrahedron/tight_binding.h"
#include "lineartetrahedron/types.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace lineartetrahedron {

struct FermiSurfaceStats {
    std::uint64_t vertex_evaluation_nanoseconds = 0;
    std::uint64_t marking_nanoseconds = 0;
    std::uint64_t refinement_nanoseconds = 0;
    std::uint64_t extraction_nanoseconds = 0;
    std::uint64_t total_nanoseconds = 0;
    std::uint64_t vertex_evaluation_calls = 0;
    std::uint64_t evaluated_vertices = 0;
    std::uint64_t marking_passes = 0;
    std::uint64_t active_simplex_visits = 0;
    std::uint64_t classified_simplices = 0;
    std::uint64_t marked_simplices = 0;
    std::uint64_t refinement_calls = 0;
};

void reset_fermi_surface_stats();

FermiSurfaceStats fermi_surface_stats();

FermiSurfaceResult fermi_surface(
    std::shared_ptr<TightBindingModel> model,
    double mu,
    double min_feature_size,
    std::int64_t max_diagonalizations,
    double tol,
    bool return_nearest_vertex_states = false
);

FermiSurfaceResult fermi_surface_from_model(
    std::shared_ptr<const HamiltonianModel> model,
    double mu,
    double min_feature_size,
    std::int64_t max_diagonalizations,
    double tol,
    bool return_nearest_vertex_states = false
);

std::vector<std::int64_t> product_simplex_triangulation_cells(size_t negative_count, size_t positive_count);

}  // namespace lineartetrahedron
