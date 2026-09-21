#include <fermisimplex/integration.h>

#include "integration/density.h"
#include "integration/simplex_cubature.h"
#include <adaptivesimplex/cut/simplex_moments.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fermisimplex {
namespace {
using namespace integration_detail;
namespace core = adaptivesimplex::core;
namespace cut = adaptivesimplex::cut;
using Value = DensityRule::Value;

struct Cell {
    core::SimplexId id;
    unsigned level = 0;
    std::vector<double> occupations;
    std::vector<std::vector<double>> vertices;
    std::map<BarycentricNode, Value> samples;
    Value value;
    double error = 0;
};

Value cubature_value(
    Cell &cell, const Cubature &cubature, SpectralMesh &mesh,
    const DensityRule &rule, IntegrationStats &stats
) {
    const auto &simplex = mesh.geometry().simplices().simplex(cell.id);
    std::vector<std::complex<long double>> sum(rule.output_size());
    long double absolute_weight_sum = 0;
    double sample_max = 0;
    for (const auto &[node, weight] : cubature) {
        auto it = cell.samples.find(node);
        if (it == cell.samples.end()) {
            const auto denominator = std::accumulate(node.begin(), node.end(), 0U);
            std::vector<double> point(mesh.ndim(), 0);
            for (std::size_t v = 0; v < node.size(); ++v) {
                for (std::size_t axis = 0; axis < point.size(); ++axis) {
                    point[axis] += cell.vertices[v][axis] * node[v] / denominator;
                }
            }
            auto spectra = mesh.spectrum(point);
            ++stats.evaluations;
            ++stats.cubature_evaluations;
            it = cell.samples.emplace(
                node, rule.at_point(spectra, point, cell.occupations)
            ).first;
        }
        absolute_weight_sum += std::abs(weight);
        sample_max = std::max(sample_max, it->second.max_abs());
        for (std::size_t c = 0; c < sum.size(); ++c) {
            sum[c] += weight * static_cast<std::complex<long double>>(it->second[c]);
        }
    }
    Value result(rule.output_size());
    Value correction(rule.output_size());
    for (std::size_t c = 0; c < sum.size(); ++c) {
        result[c] = static_cast<std::complex<double>>(sum[c] * static_cast<long double>(simplex.volume));
        correction[c] = result[c] - cell.value[c];
        if (!std::isfinite(std::abs(result[c]))) {
            throw std::runtime_error("non-finite density cubature value");
        }
    }
    // Account for cancellation in the signed higher-order rules. This is a
    // floating-point floor, not a bound on the quadrature truncation error.
    const auto roundoff = static_cast<double>(absolute_weight_sum) * sample_max *
        simplex.volume * 32 * std::numeric_limits<double>::epsilon();
    cell.error = std::max(correction.max_abs(), roundoff);
    ++stats.simplex_visits;
    return result;
}
}  // namespace

DensityComponentsResult integrate_density_components_p(
    SpectralMesh &mesh, double mu, std::vector<LatticeVector> lattice_vectors,
    std::vector<DensityComponent> components, double target_error,
    std::int64_t max_refinements, std::uint32_t max_degree
) {
    if (!std::isfinite(mu) || !std::isfinite(target_error) || target_error < 0 ||
        max_refinements < -1 ||
        (max_degree != 2 && (max_degree < 3 || max_degree > 21 || max_degree % 2 == 0))) {
        throw std::invalid_argument("invalid density p-cubature options");
    }
    DensityRule rule(mesh.ndim(), mesh.ndof(), std::move(lattice_vectors),
                     std::move(components));
    DensityComponentsResult result;
    auto &stats = result.stats;
    auto &geometry = mesh.geometry();
    auto &cache = mesh.eigensystems();
    for (const auto vertex : mesh.active_vertex_ids()) {
        if (!cache.contains(vertex)) {
            const auto point = geometry.vertices().dyadic_vertex(vertex).to_point();
            cache.insert(vertex, mesh.spectrum(point));
            ++stats.evaluations;
        }
    }
    const unsigned levels = max_degree == 2 ? 1 : (max_degree + 1) / 2;
    // Construct rules only when first needed; share coefficients across cells.
    std::vector<Cubature> rules(levels);
    rules[0] = vertices_centroid(mesh.ndim());
    std::vector<Cell> cells;
    cells.reserve(geometry.simplices().n_active());
    std::priority_queue<std::pair<double, std::size_t>> pending;
    long double total_error = 0;
    Value total(rule.output_size());
    for (const auto id : geometry.simplices().active_simplices()) {
        const auto &simplex = geometry.simplices().simplex(id);
        Cell cell{.id = id};
        cell.occupations.resize(mesh.ndof());
        bool occupied = false;
        for (std::size_t band = 0; band < mesh.ndof(); ++band) {
            const auto moments = cut::simplex_moments(
                geometry, id,
                [&](core::VertexId vertex) { return cache.get(vertex).eigenvalues[band]; },
                cut::LevelOptions{.level = mu, .level_tolerance = mesh.tolerance()}
            );
            cell.occupations[band] = moments.kind == cut::SimplexCutKind::on_level ? 0.5 :
                std::accumulate(moments.barycentric_moments.begin(),
                                moments.barycentric_moments.end(), 0.0) / simplex.volume;
            occupied = occupied || cell.occupations[band] != 0;
        }
        // Exactly empty in the frozen occupation model: no interior evaluations.
        if (!occupied) continue;
        cell.value = Value(rule.output_size());
        for (std::size_t v = 0; v < simplex.vertex_ids.size(); ++v) {
            const auto vertex = simplex.vertex_ids[v];
            const auto point = geometry.vertices().dyadic_vertex(vertex).to_point();
            cell.vertices.emplace_back(point.begin(), point.end());
            auto value = rule.at_point(cache.get(vertex), cell.vertices.back(), cell.occupations);
            for (std::size_t c = 0; c < value.size(); ++c) {
                cell.value[c] += simplex.volume * value[c] / static_cast<double>(simplex.vertex_ids.size());
            }
            BarycentricNode node(simplex.vertex_ids.size(), 0);
            node[v] = 1;
            cell.samples.emplace(std::move(node), std::move(value));
        }
        cell.value = cubature_value(cell, rules[0], mesh, rule, stats);
        total += cell.value;
        total_error += cell.error;
        if (levels > 1) pending.emplace(cell.error, cells.size());
        cells.push_back(std::move(cell));
    }
    stats.max_degree = 2;
    while (total_error > target_error && !pending.empty() &&
           (max_refinements < 0 || stats.p_refinements < max_refinements)) {
        const auto index = pending.top().second;
        pending.pop();
        auto &cell = cells[index];
        total -= cell.value;
        total_error -= cell.error;
        ++cell.level;
        if (rules[cell.level].empty()) {
            rules[cell.level] = grundmann_moeller(mesh.ndim(), cell.level);
        }
        cell.value = cubature_value(cell, rules[cell.level], mesh, rule, stats);
        total += cell.value;
        total_error += cell.error;
        ++stats.p_refinements;
        stats.max_degree = std::max(stats.max_degree, 2 * cell.level + 1);
        if (cell.level + 1 < levels) pending.emplace(cell.error, index);
    }
    // Re-sum positive local indicators instead of relying on incremental
    // subtraction when the target is near floating-point precision.
    total_error = 0;
    total = Value(rule.output_size());
    for (const auto &cell : cells) {
        total_error += cell.error;
        total += cell.value;
    }
    result.values = total.values();
    result.stopping_error = static_cast<double>(total_error);
    stats.target_reached = result.stopping_error <= target_error;
    stats.cached_vertices = mesh.cached_vertices();
    stats.active_simplices = mesh.active_simplices();
    stats.active_vertices = mesh.active_vertices();
    return result;
}
}  // namespace fermisimplex
