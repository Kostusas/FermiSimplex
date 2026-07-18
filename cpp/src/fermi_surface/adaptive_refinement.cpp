#include "fermi_surface/adaptive_refinement.h"

#include <fermisimplex/spectral_mesh.h>
#include "fermi_surface/simplex_classification.h"
#include "fermi_surface/surface_extraction.h"

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace fermisimplex::fermi_surface_detail {
namespace core = adaptivesimplex::core;
namespace {

class FermiSurfaceRun {
public:
    FermiSurfaceRun(
        SpectralMesh &mesh,
        double mu,
        double min_feature_size,
        std::int64_t max_evaluations,
        double curvature_bound
    ) : mesh_(mesh),
        mu_(mu),
        min_feature_size_(min_feature_size),
        max_evaluations_(max_evaluations),
        curvature_bound_(curvature_bound) {
        const auto active = mesh_.geometry().simplices().active_simplices();
        frontier_.assign(active.begin(), active.end());
        result_.ndim = mesh_.ndim();
    }

    FermiSurfaceResult run() {
        refine_until_terminal();
        result_.coverage_certified =
            result_.completed &&
            result_.stats.terminal_inconclusive_simplices == 0;
        extract_terminal_surface_mesh();
        return std::move(result_);
    }

private:
    std::vector<core::VertexId> missing_frontier_vertices() {
        return mesh_.geometry().missing_vertices(
            std::span<const core::SimplexId>(frontier_.data(), frontier_.size()),
            mesh_.eigensystems(),
            0
        );
    }

    bool fits_evaluation_budget(std::span<const core::VertexId> missing) const {
        if (max_evaluations_ < 0) {
            return true;
        }
        const auto budget = static_cast<std::uint64_t>(max_evaluations_);
        return result_.stats.evaluations <= budget &&
               static_cast<std::uint64_t>(missing.size()) <=
                   budget - result_.stats.evaluations;
    }

    void evaluate_frontier_vertices(std::span<const core::VertexId> missing) {
        for (const auto vertex_id : missing) {
            const auto point = mesh_.geometry()
                                   .vertices()
                                   .dyadic_vertex(vertex_id)
                                   .to_point();
            mesh_.eigensystems().insert(
                vertex_id,
                mesh_.spectrum(
                    std::span<const double>(point.data(), point.size())
                )
            );
        }
        result_.stats.evaluations += static_cast<std::uint64_t>(missing.size());
    }

    SimplexClassification classify_frontier_simplices() const {
        return classify_frontier(
            mesh_,
            frontier_,
            mu_,
            min_feature_size_,
            curvature_bound_
        );
    }

    void accumulate_terminal_counts(const SimplexClassification &classification) {
        result_.stats.terminal_visible_simplices +=
            classification.terminal_visible;
        result_.stats.terminal_inconclusive_simplices +=
            classification.terminal_inconclusive;
        terminal_surface_simplices_.insert(
            terminal_surface_simplices_.end(),
            classification.terminal_surface.begin(),
            classification.terminal_surface.end()
        );
    }

    void refine_requested_simplices(const SimplexClassification &classification) {
        frontier_ = mesh_.geometry().refine_active(
            classification.refine,
            1
        );
    }

    void refine_until_terminal() {
        while (!frontier_.empty()) {
            const auto missing = missing_frontier_vertices();
            if (!fits_evaluation_budget(missing)) {
                result_.completed = false;
                break;
            }

            evaluate_frontier_vertices(
                std::span<const core::VertexId>(missing.data(), missing.size())
            );
            auto classification = classify_frontier_simplices();
            accumulate_terminal_counts(classification);

            if (classification.refine.empty()) {
                frontier_.clear();
                break;
            }
            refine_requested_simplices(classification);
        }
    }

    void extract_terminal_surface_mesh() {
        extract_terminal_surface(
            mesh_,
            std::span<const core::SimplexId>(
                terminal_surface_simplices_.data(),
                terminal_surface_simplices_.size()
            ),
            mu_,
            result_
        );
    }

    SpectralMesh &mesh_;
    double mu_ = 0.0;
    double min_feature_size_ = 0.0;
    std::int64_t max_evaluations_ = -1;
    double curvature_bound_ = 0.0;
    std::vector<core::SimplexId> frontier_;
    std::vector<core::SimplexId> terminal_surface_simplices_;
    FermiSurfaceResult result_;
};

}  // namespace

FermiSurfaceResult run_fermi_surface(
    SpectralMesh &mesh,
    double mu,
    double min_feature_size,
    std::int64_t max_evaluations,
    double curvature_bound
) {
    return FermiSurfaceRun(
        mesh,
        mu,
        min_feature_size,
        max_evaluations,
        curvature_bound
    ).run();
}

}  // namespace fermisimplex::fermi_surface_detail
