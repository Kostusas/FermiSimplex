#include <fermisimplex/integration.h>

#include "integration/density.h"
#include "integration/density_error.h"
#include "integration/simplex_cubature.h"
#include <adaptivesimplex/adaptive/refinement_queue.h>
#include <adaptivesimplex/cut/simplex_moments.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <exception>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace fermisimplex {
namespace {
using namespace integration_detail;
namespace core = adaptivesimplex::core;
namespace cut = adaptivesimplex::cut;
using Value = DensityRule::Value;

struct Cell {
    core::SimplexId id;
    unsigned level = 0;
    bool active = true;
    std::vector<double> occupations;
    // The charge mesh's vertex-linear band energies, restricted to this cell.
    // Splitting this field preserves the charge-stage occupation exactly.
    std::vector<std::vector<double>> frozen_energies;
    std::vector<std::vector<double>> vertices;
    std::map<BarycentricNode, Value> samples;
    Value value;
    Value cut_correction;
    Value correction;
    double roundoff = 0;
    double error = 0;
};

Value cubature_value(
    Cell &cell, const Cubature &cubature, const core::Geometry &geometry,
    SpectralMesh &mesh, const DensityRule &rule, IntegrationStats &stats
) {
    const auto &simplex = geometry.simplices().simplex(cell.id);
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
        result[c] = static_cast<std::complex<double>>(
            sum[c] * static_cast<long double>(simplex.volume)
        );
        correction[c] = result[c] - cell.value[c];
        if (!std::isfinite(std::abs(result[c]))) {
            throw std::runtime_error("non-finite density cubature value");
        }
    }
    const auto roundoff = static_cast<double>(absolute_weight_sum) * sample_max *
        simplex.volume * 32 * std::numeric_limits<double>::epsilon();
    cell.correction = std::move(correction);
    cell.roundoff = roundoff;
    cell.error = std::max(cell.correction.max_abs(), roundoff);
    ++stats.simplex_visits;
    return result;
}
}  // namespace

DensityComponentsResult integrate_density_components_p(
    SpectralMesh &mesh, double mu, std::vector<LatticeVector> lattice_vectors,
    std::vector<DensityComponent> components, double target_error,
    std::int64_t max_refinements, std::uint32_t max_degree,
    std::int64_t max_h_refinements
) {
    if (!std::isfinite(mu) || !std::isfinite(target_error) || target_error < 0 ||
        max_refinements < -1 || max_h_refinements < -1 ||
        (max_degree != 2 && (max_degree < 3 || max_degree > 21 || max_degree % 2 == 0))) {
        throw std::invalid_argument("invalid density hp-cubature options");
    }
    DensityRule rule(mesh.ndim(), mesh.ndof(), std::move(lattice_vectors),
                     std::move(components));
    DensityComponentsResult result;
    auto &stats = result.stats;
    const core::Geometry *geometry = &mesh.geometry();
    std::optional<core::Geometry> density_geometry;
    auto &cache = mesh.eigensystems();
    for (const auto vertex : mesh.active_vertex_ids()) {
        if (!cache.contains(vertex)) {
            const auto point = geometry->vertices().dyadic_vertex(vertex).to_point();
            cache.insert(vertex, mesh.spectrum(point));
            ++stats.evaluations;
        }
    }
    // Density-only midpoint spectra never enter the charge mesh cache.
    std::unordered_map<core::VertexId, Eigensystem> local_cache;
    const auto vertex_spectrum = [&](core::VertexId vertex) -> const Eigensystem & {
        if (cache.contains(vertex)) return cache.get(vertex);
        if (const auto it = local_cache.find(vertex); it != local_cache.end()) {
            return it->second;
        }
        const auto point = geometry->vertices().dyadic_vertex(vertex).to_point();
        ++stats.evaluations;
        return local_cache.emplace(vertex, mesh.spectrum(point)).first->second;
    };
    const unsigned levels = max_degree == 2 ? 1 : (max_degree + 1) / 2;
    std::vector<Cubature> rules(levels);
    rules[0] = vertices_centroid(mesh.ndim());
    std::vector<Cell> cells;
    cells.reserve(geometry->simplices().n_active());
    adaptivesimplex::adaptive::RefinementQueue pending;
    DensityGlobalError error_policy;
    double squared_p_error = 0;
    Value p_correction_sum(rule.output_size());
    long double roundoff_sum = 0;
    const auto add_error = [&](const Cell &cell) {
        error_policy.add_local_estimate(squared_p_error, cell.correction);
        p_correction_sum += cell.correction;
        roundoff_sum += cell.roundoff;
    };
    const auto remove_error = [&](const Cell &cell) {
        error_policy.remove_local_estimate(squared_p_error, cell.correction);
        p_correction_sum -= cell.correction;
        roundoff_sum -= cell.roundoff;
    };
    const auto global_error = [&]() {
        return std::max({
            error_policy.error(squared_p_error, p_correction_sum),
            static_cast<double>(std::max(0.L, roundoff_sum))
        });
    };
    const auto make_cell = [&](core::SimplexId id, const Cell *parent) -> std::optional<Cell> {
        const auto &simplex = geometry->simplices().simplex(id);
        Cell cell{.id = id};
        cell.occupations.resize(mesh.ndof());
        cell.frozen_energies.reserve(simplex.vertex_ids.size());
        if (parent) {
            const auto &parent_simplex = geometry->simplices().simplex(parent->id);
            const auto &children = *parent_simplex.children;
            for (const auto vertex : simplex.vertex_ids) {
                if (vertex == children.midpoint) {
                    std::vector<double> energies(mesh.ndof());
                    for (std::size_t band = 0; band < mesh.ndof(); ++band) {
                        energies[band] = 0.5 * (
                            parent->frozen_energies[children.split_edge[0]][band] +
                            parent->frozen_energies[children.split_edge[1]][band]
                        );
                    }
                    cell.frozen_energies.push_back(std::move(energies));
                } else {
                    const auto it = std::find(
                        parent_simplex.vertex_ids.begin(),
                        parent_simplex.vertex_ids.end(), vertex
                    );
                    cell.frozen_energies.push_back(
                        parent->frozen_energies[
                            static_cast<std::size_t>(it - parent_simplex.vertex_ids.begin())
                        ]
                    );
                }
            }
        } else {
            for (const auto vertex : simplex.vertex_ids) {
                cell.frozen_energies.push_back(vertex_spectrum(vertex).eigenvalues);
            }
        }
        bool occupied = false;
        std::vector<double> cut_weights;
        for (std::size_t band = 0; band < mesh.ndof(); ++band) {
            const auto moments = cut::simplex_moments(
                *geometry, id,
                [&](core::VertexId vertex) {
                    const auto it = std::find(
                        simplex.vertex_ids.begin(), simplex.vertex_ids.end(), vertex
                    );
                    return cell.frozen_energies[
                        static_cast<std::size_t>(it - simplex.vertex_ids.begin())
                    ][band];
                },
                cut::LevelOptions{.level = mu, .level_tolerance = mesh.tolerance()}
            );
            cell.occupations[band] = moments.kind == cut::SimplexCutKind::on_level ? 0.5 :
                std::accumulate(moments.barycentric_moments.begin(),
                                moments.barycentric_moments.end(), 0.0) / simplex.volume;
            occupied = occupied || cell.occupations[band] != 0;
            if (moments.kind == cut::SimplexCutKind::partial) {
                if (cut_weights.empty()) {
                    cut_weights.resize(simplex.vertex_ids.size() * mesh.ndof(), 0.0);
                }
                const auto average = cell.occupations[band] * simplex.volume /
                    static_cast<double>(simplex.vertex_ids.size());
                for (std::size_t v = 0; v < simplex.vertex_ids.size(); ++v) {
                    cut_weights[v * mesh.ndof() + band] =
                        moments.barycentric_moments[v] - average;
                }
            }
        }
        if (!occupied) return std::nullopt;
        cell.value = Value(rule.output_size());
        cell.cut_correction = Value(rule.output_size());
        for (std::size_t v = 0; v < simplex.vertex_ids.size(); ++v) {
            const auto vertex = simplex.vertex_ids[v];
            const auto point = geometry->vertices().dyadic_vertex(vertex).to_point();
            cell.vertices.emplace_back(point.begin(), point.end());
            const auto &spectrum = vertex_spectrum(vertex);
            auto value = rule.at_point(spectrum, cell.vertices.back(), cell.occupations);
            if (!cut_weights.empty()) {
                cell.cut_correction += rule.at_point(
                    spectrum, cell.vertices.back(),
                    std::span<const double>(
                        cut_weights.data() + v * mesh.ndof(), mesh.ndof()
                    )
                );
            }
            for (std::size_t c = 0; c < value.size(); ++c) {
                cell.value[c] += simplex.volume * value[c] /
                    static_cast<double>(simplex.vertex_ids.size());
            }
            BarycentricNode node(simplex.vertex_ids.size(), 0);
            node[v] = 1;
            cell.samples.emplace(std::move(node), std::move(value));
        }
        // The cut correction uses moments of the original charge simplex
        // restricted to this density child, keeping its total charge fixed.
        cell.value = cubature_value(cell, rules[0], *geometry, mesh, rule, stats);
        return cell;
    };
    const auto enqueue = [&](std::size_t index) {
        const auto &cell = cells[index];
        if (cell.level + 1 < levels || max_h_refinements != 0) {
            pending.push(index, cell.error);
        }
    };
    for (const auto id : geometry->simplices().active_simplices()) {
        auto cell = make_cell(id, nullptr);
        if (!cell) continue;
        add_error(*cell);
        cells.push_back(std::move(*cell));
        enqueue(cells.size() - 1);
    }
    stats.max_degree = 2;
    int threads = 1;
#ifdef _OPENMP
    if (mesh.ndof() >= 32) threads = std::min(omp_get_max_threads(), 16);
#endif
    const auto batch_size = max_h_refinements == 0 && threads > 1 ? 16 : 1;
    while (global_error() > target_error) {
        const auto remaining_p = max_refinements < 0 ?
            -1 : max_refinements - stats.p_refinements;
        if (max_h_refinements == 0 && remaining_p == 0) break;
        const auto selected = pending.select_for_reduction(
            global_error() - target_error,
            max_h_refinements == 0 ? remaining_p : -1, 1, batch_size
        );
        if (selected.empty()) break;
        if (selected.size() == 1) {
            const auto index = static_cast<std::size_t>(selected.front());
            auto &cell = cells[index];
            if (cell.level + 1 < levels) {
                if (remaining_p == 0) continue;
                remove_error(cell);
                ++cell.level;
                if (rules[cell.level].empty()) {
                    rules[cell.level] = grundmann_moeller(mesh.ndim(), cell.level);
                }
                cell.value = cubature_value(
                    cell, rules[cell.level], *geometry, mesh, rule, stats
                );
                add_error(cell);
                ++stats.p_refinements;
                stats.max_degree = std::max(stats.max_degree, 2 * cell.level + 1);
                enqueue(index);
                continue;
            }
            if (max_h_refinements == 0 ||
                (max_h_refinements > 0 && stats.refinements >= max_h_refinements)) {
                continue;
            }
            remove_error(cell);
            Cell parent = std::move(cell);
            cells[index].active = false;
            if (!density_geometry) {
                density_geometry.emplace(mesh.geometry());
                geometry = &*density_geometry;
            }
            const auto child_ids = density_geometry->refine_active({parent.id});
            ++stats.refinements;
            for (const auto child_id : child_ids) {
                auto child = make_cell(child_id, &parent);
                if (!child) continue;
                add_error(*child);
                cells.push_back(std::move(*child));
                enqueue(cells.size() - 1);
            }
            continue;
        }
        // The p-only path preserves the previous parallel batch controller.
        for (const auto index : selected) {
            auto &cell = cells[index];
            remove_error(cell);
            ++cell.level;
            if (rules[cell.level].empty()) {
                rules[cell.level] = grundmann_moeller(mesh.ndim(), cell.level);
            }
        }
        std::vector<IntegrationStats> work(selected.size());
        std::vector<std::exception_ptr> failures(selected.size());
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(static)
#endif
        for (std::int64_t task = 0; task < static_cast<std::int64_t>(selected.size()); ++task) {
            const auto i = static_cast<std::size_t>(task);
            try {
                auto &cell = cells[selected[i]];
                cell.value = cubature_value(
                    cell, rules[cell.level], *geometry, mesh, rule, work[i]
                );
            } catch (...) {
                failures[i] = std::current_exception();
            }
        }
        for (std::size_t i = 0; i < selected.size(); ++i) {
            if (failures[i]) std::rethrow_exception(failures[i]);
            auto &cell = cells[selected[i]];
            stats.evaluations += work[i].evaluations;
            stats.cubature_evaluations += work[i].cubature_evaluations;
            stats.simplex_visits += work[i].simplex_visits;
            add_error(cell);
            ++stats.p_refinements;
            stats.max_degree = std::max(stats.max_degree, 2 * cell.level + 1);
            enqueue(selected[i]);
        }
    }
    squared_p_error = 0;
    p_correction_sum = Value(rule.output_size());
    roundoff_sum = 0;
    Value total(rule.output_size());
    for (const auto &cell : cells) {
        if (!cell.active) continue;
        add_error(cell);
        total += cell.value;
        total += cell.cut_correction;
    }
    result.values = total.values();
    result.stopping_error = global_error();
    stats.target_reached = result.stopping_error <= target_error;
    stats.cached_vertices = mesh.cached_vertices();
    stats.active_simplices = geometry->simplices().n_active();
    stats.active_vertices = geometry->n_active_vertices();
    return result;
}
}  // namespace fermisimplex
