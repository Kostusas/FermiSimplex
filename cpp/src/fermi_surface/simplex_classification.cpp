#include "fermi_surface/simplex_classification.h"

#include <fermisimplex/occupation.h>
#include "core/simplex_geometry.h"

namespace fermisimplex::fermi_surface_detail {
namespace {

void append_visible(
    core::SimplexId simplex_id,
    bool refinable,
    SimplexClassification &result
) {
    if (refinable) {
        result.refine.push_back(simplex_id);
    } else {
        result.terminal_surface.push_back(simplex_id);
        ++result.terminal_visible;
    }
}

}  // namespace

SimplexClassification classify_frontier(
    const SpectralMesh &mesh,
    const std::vector<core::SimplexId> &frontier,
    double mu,
    double min_feature_size,
    double curvature_bound
) {
    const auto &geometry = mesh.geometry();
    SimplexClassification result;
    for (const auto simplex_id : frontier) {
        const auto refinable = simplex_diameter(geometry, simplex_id) > min_feature_size;
        std::optional<double> remainder;
        if (curvature_bound > 0) {
            // If ||H-L|| <= e for the affine interpolant L, the edge
            // controls of Q-L have norm <= 2e and zero vertex controls.
            // Their weights sum to at most d/(d+1).
            remainder = (1. + 2. * mesh.ndim() / (mesh.ndim() + 1.)) *
                mesh.linearization_error_bound(simplex_id, curvature_bound);
        }
        ChargeErrorStats stats;
        const auto enclosure = enclose_occupation(mesh, simplex_id, mu, 2, stats, remainder);
        if (enclosure.fixed_occupation()) continue;
        if (visible_occupation_change(mesh, simplex_id, mu)) {
            append_visible(simplex_id, refinable, result);
        } else {
            if (refinable) {
                result.refine.push_back(simplex_id);
            } else {
                result.terminal_surface.push_back(simplex_id);
                ++result.terminal_inconclusive;
            }
        }
    }
    return result;
}

}  // namespace fermisimplex::fermi_surface_detail
