#include "fermi_surface/simplex_classification.h"

#include "certification/mesh_certificate.h"
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
        const auto certificate = certification::certify_mesh_simplex(
            mesh,
            simplex_id,
            mu,
            mesh.linearization_error_bound(simplex_id, curvature_bound),
            mesh.tolerance()
        );
        const auto status = certificate.status;

        switch (status) {
        case certification::SimplexCertificateStatus::CertifiedGapped:
            break;
        case certification::SimplexCertificateStatus::VisibleGapless:
            append_visible(simplex_id, refinable, result);
            break;
        case certification::SimplexCertificateStatus::Inconclusive:
            if (refinable) {
                result.refine.push_back(simplex_id);
            } else {
                result.terminal_surface.push_back(simplex_id);
                ++result.terminal_inconclusive;
            }
            break;
        }
    }
    return result;
}

}  // namespace fermisimplex::fermi_surface_detail
