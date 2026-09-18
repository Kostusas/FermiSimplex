#include "core/evaluated_partition.h"
#include "test_helpers.h"

#include <adaptivesimplex/core/root_mesh.h>
#include <adaptivesimplex/core/vertex_cache.h>

#include <cmath>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>

namespace core = adaptivesimplex::core;
namespace detail = fermisimplex::detail;
using fermisimplex::test::expect;

void cache_vertices(const core::Geometry &geometry, core::VertexCache<double> &cache,
                    const std::vector<core::SimplexId> &ids) {
    for (const auto id : ids) {
        for (const auto vertex : geometry.simplices().simplex(id).vertex_ids) {
            cache.insert(vertex, 1.0);
        }
    }
}

// Every root-to-leaf path must cross exactly one selected cell. This checks
// coverage and ancestor overlap independently of floating-point volume sums.
void check_frontier(const core::Geometry &geometry, core::SimplexId id,
                    const std::set<core::SimplexId> &selected, int ancestors = 0) {
    ancestors += selected.contains(id);
    expect(ancestors <= 1, "no parent/child overlap");
    const auto &simplex = geometry.simplices().simplex(id);
    if (simplex.children) {
        for (const auto child : simplex.children->simplex_ids) {
            check_frontier(geometry, child, selected, ancestors);
        }
    } else {
        expect(ancestors == 1, "every branch is covered");
    }
}

void verify(const core::Geometry &geometry, const core::VertexCache<double> &cache,
            const std::vector<detail::EvaluatedSimplex> &partition) {
    std::set<core::SimplexId> selected;
    double volume = 0.0;
    for (const auto cell : partition) {
        expect(selected.insert(cell.simplex_id).second, "no duplicate cells");
        expect(geometry.simplices().is_active(cell.active_ancestor_id), "ancestor is active");
        const auto &simplex = geometry.simplices().simplex(cell.simplex_id);
        volume += simplex.volume;
        for (const auto vertex : simplex.vertex_ids) {
            expect(cache.contains(vertex), "cell vertex has cached data");
        }
    }
    expect(std::abs(volume - 1.0) < 1e-12, "unit-domain volume");
    for (const auto active : geometry.simplices().active_simplices()) {
        check_frontier(geometry, active, selected);
    }
    std::cout << "dimension=" << geometry.ndim()
              << " volume_error=" << std::abs(volume - 1.0) << '\n';
}

int main() {
    for (std::size_t dimension = 1; dimension <= 4; ++dimension) {
        auto geometry = core::root_geometry(dimension, 0);
        auto cache = core::VertexCache<double>(0);
        const auto active_span = geometry.simplices().active_simplices();
        const std::vector<core::SimplexId> active(active_span.begin(), active_span.end());
        bool threw = false;
        try {
            detail::evaluated_partition(geometry, cache);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        expect(threw, "unevaluated mesh has no complete covering");
        cache_vertices(geometry, cache, active);
        expect(detail::evaluated_partition(geometry, cache).size() == active.size(),
               "depth-zero partition");

        for (const auto id : active) {
            cache_vertices(geometry, cache, geometry.preview_active(id, 2));
        }
        const auto expected = detail::evaluated_partition(geometry, cache);
        expect(expected.size() == 4 * active.size(), "complete depth-two previews");
        verify(geometry, cache, expected);
        // New geometry alone is not evidence of new evaluated data.
        for (const auto id : active) {
            geometry.preview_active(id, 4);
        }
        const auto simplex_count = geometry.simplices().size();
        const auto vertex_count = geometry.vertices().size();
        const auto cache_count = cache.size();
        expect(detail::evaluated_partition(geometry, cache) == expected,
               "repeatable evaluated frontier");
        expect(detail::evaluated_partition(geometry, cache) == expected,
               "repeatable evaluated frontier");
        expect(geometry.simplices().size() == simplex_count, "unchanged simplex tree");
        expect(geometry.vertices().size() == vertex_count, "unchanged vertex table");
        expect(cache.size() == cache_count, "unchanged cache");
        const auto after = geometry.simplices().active_simplices();
        expect(std::vector<core::SimplexId>(after.begin(), after.end()) == active,
               "unchanged active cells");
        verify(geometry, cache, expected);

        auto control = geometry;
        const auto replacements = geometry.refine_active({active.front()}, 1);
        expect(replacements == control.refine_active({active.front()}, 1),
               "unchanged subsequent refinement");
        expect(detail::evaluated_partition(geometry, cache) ==
               detail::evaluated_partition(control, cache),
               "unchanged subsequent partition");
    }

    // Mixed-depth 1D fixture: retain x=1/8 even while x=1/4 is missing.
    auto line = core::root_geometry(1, 0);
    line.preview_active(0, 3);
    auto cache = core::VertexCache<double>(0);
    for (std::size_t vertex = 0; vertex < line.vertices().size(); ++vertex) {
        const auto x = line.vertices().dyadic_vertex(vertex).to_point()[0];
        if (x == 0.0 || x == 0.5 || x == 1.0 || x == 0.125 || x == 0.75) {
            cache.insert(vertex, x);
        }
    }
    const auto partition = detail::evaluated_partition(line, cache);
    expect(partition.size() == 3, "mixed-depth partition");  // [0,1/2], [1/2,3/4], [3/4,1]
    verify(line, cache, partition);
    std::set<core::VertexId> used;
    for (const auto cell : partition) {
        for (const auto vertex : line.simplices().simplex(cell.simplex_id).vertex_ids) {
            used.insert(vertex);
        }
    }
    expect(cache.size() == used.size() + 1, "extra cached point survives incomplete preview");
    return 0;
}
