#include <fermisimplex/integration.h>

#include "integration/density.h"
#include "integration/density_error.h"
#include <adaptivesimplex/adaptive/refinement_queue.h>
#include "integration/simplex_cubature.h"
#include <adaptivesimplex/cut/simplex_moments.h>

#include <algorithm>
#include <exception>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cmath>
#include <complex>
#include <limits>
#include <map>
#include <numeric>
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
    Value cut_correction;
    Value correction;
    double error = 0;
    double roundoff = 0;
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
    adaptivesimplex::adaptive::RefinementQueue pending;
    DensityGlobalError error_policy;
    double squared_error = 0;
    Value correction_sum(rule.output_size());
    long double roundoff_sum = 0;
    const auto add_error = [&](const Cell &cell) {
        error_policy.add_local_estimate(squared_error, cell.correction);
        correction_sum += cell.correction;
        roundoff_sum += cell.roundoff;
    };
    const auto remove_error = [&](const Cell &cell) {
        error_policy.remove_local_estimate(squared_error, cell.correction);
        correction_sum -= cell.correction;
        roundoff_sum -= cell.roundoff;
    };
    const auto global_error = [&]() {
        return std::max(error_policy.error(squared_error, correction_sum),
                        static_cast<double>(std::max(0.L, roundoff_sum)));
    };
    Value total(rule.output_size());
    for (const auto id : geometry.simplices().active_simplices()) {
        const auto &simplex = geometry.simplices().simplex(id);
        Cell cell{.id = id};
        cell.occupations.resize(mesh.ndof());
        bool occupied = false;
        std::vector<double> cut_weights;
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
        // Exactly empty in the frozen occupation model: no interior evaluations.
        if (!occupied) continue;
        cell.value = Value(rule.output_size());
        if (!cut_weights.empty()) cell.cut_correction = Value(rule.output_size());
        for (std::size_t v = 0; v < simplex.vertex_ids.size(); ++v) {
            const auto vertex = simplex.vertex_ids[v];
            const auto point = geometry.vertices().dyadic_vertex(vertex).to_point();
            cell.vertices.emplace_back(point.begin(), point.end());
            auto value = rule.at_point(cache.get(vertex), cell.vertices.back(), cell.occupations);
            if (!cut_weights.empty()) {
                cell.cut_correction += rule.at_point(
                    cache.get(vertex), cell.vertices.back(),
                    std::span<const double>(cut_weights.data() + v * mesh.ndof(), mesh.ndof())
                );
            }
            for (std::size_t c = 0; c < value.size(); ++c) {
                cell.value[c] += simplex.volume * value[c] / static_cast<double>(simplex.vertex_ids.size());
            }
            BarycentricNode node(simplex.vertex_ids.size(), 0);
            node[v] = 1;
            cell.samples.emplace(std::move(node), std::move(value));
        }
        // Q_p + integral_cut(L) - fraction*integral_full(L), where L is
        // the vertex-linear projector including its Fourier phase. Keeping
        // this degree-independent correction separate leaves p differences
        // unchanged and requires no new spectral samples.
        cell.value = cubature_value(cell, rules[0], mesh, rule, stats);
        add_error(cell);
        if (levels > 1) pending.push(cells.size(), cell.error);
        cells.push_back(std::move(cell));
    }
    stats.max_degree = 2;
    // Small eigensystems do not consistently amortize parallel scheduling.
    // MeanFi's default thread limit of one retains the serial controller.
    int threads = 1;
#ifdef _OPENMP
    if (mesh.ndof() >= 32) threads = std::min(omp_get_max_threads(), 16);
#endif
    const auto batch_size = threads > 1 ? 16 : 1;
    while (global_error() > target_error &&
           (max_refinements < 0 || stats.p_refinements < max_refinements)) {
        const auto remaining = max_refinements < 0 ? -1 : max_refinements - stats.p_refinements;
        const auto selected = pending.select_for_reduction(
            global_error() - target_error, remaining, 1, batch_size
        );
        if (selected.empty()) break;
        for (const auto index : selected) {
            auto &cell = cells[index];
            remove_error(cell);
            ++cell.level;
            if (rules[cell.level].empty()) {
                rules[cell.level] = grundmann_moeller(mesh.ndim(), cell.level);
            }
        }
        if (selected.size() == 1) {
            auto &cell = cells[selected.front()];
            cell.value = cubature_value(cell, rules[cell.level], mesh, rule, stats);
            add_error(cell);
            ++stats.p_refinements;
            stats.max_degree = std::max(stats.max_degree, 2 * cell.level + 1);
            if (cell.level + 1 < levels) pending.push(selected.front(), cell.error);
            continue;
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
                cell.value = cubature_value(cell, rules[cell.level], mesh, rule, work[i]);
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
            if (cell.level + 1 < levels) pending.push(selected[i], cell.error);
        }
    }
    // Recompute aggregates to limit cancellation in incremental updates.
    squared_error = 0;
    correction_sum = Value(rule.output_size());
    roundoff_sum = 0;
    total = Value(rule.output_size());
    for (const auto &cell : cells) {
        add_error(cell);
        total += cell.value;
        total += cell.cut_correction;
    }
    result.values = total.values();
    result.stopping_error = global_error();
    stats.target_reached = result.stopping_error <= target_error;
    stats.cached_vertices = mesh.cached_vertices();
    stats.active_simplices = mesh.active_simplices();
    stats.active_vertices = mesh.active_vertices();
    return result;
}
}  // namespace fermisimplex
