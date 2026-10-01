#include "fermi_surface/simplex_classification.h"

#include <fermisimplex/occupation.h>
#include "occupation/enclosure.h"
#include "core/simplex_geometry.h"
#include "core/tight_binding_access.h"

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
    const auto constant = core_detail::TightBindingModelAccess::constant_spectrum_roundoff(
        mesh.model()).has_value();
    SimplexClassification result;
    for (const auto simplex_id : frontier) {
        // An unresolved constant spectrum cannot define a codimension-one
        // surface or become more certain through spatial refinement.
        const auto refinable = !constant &&
            simplex_diameter(geometry, simplex_id) > min_feature_size;
        const auto vertices = constant ? occupation_detail::VertexOccupation::uniform :
            occupation_detail::vertex_occupation(mesh, simplex_id, mu);
        if (vertices == occupation_detail::VertexOccupation::crossing) {
            append_visible(simplex_id, refinable, result);
            continue;
        }
        std::optional<double> remainder;
        if (curvature_bound > 0) {
            // If ||H-L|| <= e for the affine interpolant L, the edge
            // controls of Q-L have norm <= 2e and zero vertex controls.
            // Their weights sum to at most d/(d+1).
            remainder = (1. + 2. * mesh.ndim() / (mesh.ndim() + 1.)) *
                mesh.linearization_error_bound(simplex_id, curvature_bound);
        }
        ChargeErrorStats stats;
        if (occupation_detail::fixed_occupation(mesh, simplex_id, mu, 2, stats, remainder))
            continue;
        if (vertices == occupation_detail::VertexOccupation::touches_level) {
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
