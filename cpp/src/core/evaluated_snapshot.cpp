#include <fermisimplex/spectral_mesh.h>

#include <adaptivesimplex/core/evaluated_partition.h>

#include <limits>
#include <stdexcept>

namespace fermisimplex {

EvaluatedSnapshot SpectralMesh::evaluated_snapshot(bool include_eigenvectors) const {
    namespace core = adaptivesimplex::core;
    const auto &geometry = this->geometry();
    const auto &cache = eigensystems();
    const auto partition = core::evaluated_partition(geometry, cache);
    auto result = EvaluatedSnapshot{};
    result.ndim = ndim();
    result.ndof = ndof();
    result.active_simplices = geometry.simplices().n_active();
    const auto active_ids = active_vertex_ids();
    result.active_vertices = active_ids.size();
    auto is_active = std::vector<std::uint8_t>(geometry.vertices().size(), 0);
    for (const auto id : active_ids) {
        is_active[id] = 1;
    }
    const auto missing = std::numeric_limits<std::size_t>::max();
    auto rows = std::vector<std::size_t>(geometry.vertices().size(), missing);
    result.vertex_ids.reserve(cache.size());
    result.points.reserve(cache.size() * ndim());
    result.dyadic_numerators.reserve(cache.size() * ndim());
    result.dyadic_levels.reserve(cache.size());
    result.eigenvalues.reserve(cache.size() * ndof());
    result.vertex_is_active.reserve(cache.size());
    if (include_eigenvectors) {
        result.eigenvectors.emplace();
        result.eigenvectors->reserve(cache.size() * ndof() * ndof());
    }
    for (core::VertexId id = 0; id < geometry.vertices().size(); ++id) {
        if (!cache.contains(id)) {
            continue;
        }
        const auto &spectrum = cache.get(id);
        const auto &vertex = geometry.vertices().dyadic_vertex(id);
        const auto point = vertex.to_point();
        rows[id] = result.vertex_ids.size();
        result.vertex_ids.push_back(id);
        result.points.insert(result.points.end(), point.begin(), point.end());
        result.dyadic_numerators.insert(result.dyadic_numerators.end(),
                                        vertex.coords().begin(), vertex.coords().end());
        result.dyadic_levels.push_back(vertex.level());
        result.vertex_is_active.push_back(is_active[id]);
        result.eigenvalues.insert(result.eigenvalues.end(),
                                  spectrum.eigenvalues.begin(), spectrum.eigenvalues.end());
        if (include_eigenvectors) {
            for (std::size_t row = 0; row < ndof(); ++row) {
                for (std::size_t band = 0; band < ndof(); ++band) {
                    result.eigenvectors->push_back(spectrum.eigenvectors[band * ndof() + row]);
                }
            }
        }
    }
    if (result.vertex_ids.size() != cache.size()) {
        throw std::runtime_error("evaluated_snapshot: cache contains vertices outside geometry");
    }
    result.simplex_ids.reserve(partition.size());
    result.active_ancestor_ids.reserve(partition.size());
    result.simplices.reserve(partition.size() * (ndim() + 1));
    result.volumes.reserve(partition.size());
    for (const auto cell : partition) {
        const auto &simplex = geometry.simplices().simplex(cell.simplex_id);
        result.simplex_ids.push_back(cell.simplex_id);
        result.active_ancestor_ids.push_back(cell.active_ancestor_id);
        result.volumes.push_back(simplex.volume);
        for (const auto vertex : simplex.vertex_ids) {
            result.simplices.push_back(rows[vertex]);
        }
    }
    return result;
}

}  // namespace fermisimplex
