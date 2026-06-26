#include "fermi_surface/simplex_classification.h"

#include "certificate/simplex_certificate.h"

#include <algorithm>
#include <cstddef>
#include <cmath>

namespace lineartetrahedron::fermi_surface_detail {
namespace {

double max_reduced_edge_length(
    const core::Geometry &geometry,
    core::SimplexId simplex_id
) {
    const auto &simplex = geometry.simplices().simplex(simplex_id);
    auto result = 0.0;
    for (size_t left = 0; left < simplex.vertex_ids.size(); ++left) {
        const auto left_point =
            geometry.vertices().dyadic_vertex(simplex.vertex_ids[left]).to_point();
        for (size_t right = left + 1; right < simplex.vertex_ids.size(); ++right) {
            const auto right_point =
                geometry.vertices().dyadic_vertex(simplex.vertex_ids[right]).to_point();
            auto squared = 0.0;
            for (size_t axis = 0; axis < geometry.ndim(); ++axis) {
                const auto delta = left_point[axis] - right_point[axis];
                squared += delta * delta;
            }
            result = std::max(result, std::sqrt(squared));
        }
    }
    return result;
}

}  // namespace

SimplexClassification classify_frontier(
    const core::Geometry &geometry,
    const SpectraCache &cache,
    const std::vector<core::SimplexId> &frontier,
    double mu,
    double min_feature_size,
    double margin,
    double tol
) {
    SimplexClassification result;
    for (const auto simplex_id : frontier) {
        const auto refinable = max_reduced_edge_length(geometry, simplex_id) > min_feature_size;
        const auto certificate = simplex_certificate::certify_simplex_gap(
            geometry,
            simplex_id,
            cache,
            mu,
            margin,
            tol
        );
        const auto status = certificate.status;

        if (status == simplex_certificate::SimplexCertificateStatus::CertifiedGapped) {
            continue;
        } else if (status == simplex_certificate::SimplexCertificateStatus::VisibleGapless) {
            if (refinable) {
                result.refine.push_back(simplex_id);
            } else {
                result.terminal_surface.push_back(simplex_id);
                ++result.visible_gapless_terminal;
            }
        } else if (refinable) {
            result.refine.push_back(simplex_id);
        } else {
            result.terminal_surface.push_back(simplex_id);
            ++result.inconclusive_terminal;
        }
    }
    return result;
}

std::set<core::SimplexId> simplex_set(const std::vector<core::SimplexId> &simplex_ids) {
    return std::set<core::SimplexId>(simplex_ids.begin(), simplex_ids.end());
}

std::vector<core::SimplexId> next_frontier(
    const core::Geometry &geometry,
    const std::set<core::SimplexId> &previous_active,
    const std::vector<core::SimplexId> &requested_refinement,
    const std::vector<core::SimplexId> &refined
) {
    const auto refined_set = simplex_set(refined);
    std::set<core::SimplexId> seen;
    std::vector<core::SimplexId> result;

    for (const auto simplex_id : active_simplices(geometry)) {
        if (!previous_active.contains(simplex_id)) {
            seen.insert(simplex_id);
            result.push_back(simplex_id);
        }
    }
    for (const auto simplex_id : requested_refinement) {
        if (!refined_set.contains(simplex_id) && seen.insert(simplex_id).second) {
            result.push_back(simplex_id);
        }
    }
    return result;
}

}  // namespace lineartetrahedron::fermi_surface_detail
