#include "fermi_surface/adaptive_refinement.h"
#include "fermi_surface/simplex_classification.h"
#include "fermi_surface/surface_extraction.h"
#include "fermi_surface/vertex_evaluation.h"

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace lineartetrahedron::fermi_surface_detail {
namespace {

class FermiSurfaceRun {
public:
    FermiSurfaceRun(
        std::shared_ptr<const HamiltonianModel> model,
        double mu,
        double min_feature_size,
        std::int64_t max_diagonalizations,
        double margin,
        double tol,
        bool return_states
    ) : model_(std::move(model)),
        mu_(mu),
        min_feature_size_(min_feature_size),
        max_diagonalizations_(max_diagonalizations),
        margin_(margin),
        tol_(tol),
        return_states_(return_states),
        geometry_(make_fermi_geometry(model_->ndim())),
        evaluator_(model_),
        frontier_(active_simplices(geometry_)) {
        result_.ndim = model_->ndim();
        result_.ndof = model_->ndof();
        result_.has_states = return_states_;
        result_.min_feature_size = min_feature_size_;
    }

    FermiSurfaceResult run() {
        refine_until_terminal();
        extract_terminal_surface_mesh();
        return std::move(result_);
    }

private:
    std::vector<core::VertexId> missing_frontier_vertices() {
        return missing_vertices_for(geometry_, cache_, frontier_);
    }

    bool fits_diagonalization_budget(std::span<const core::VertexId> missing) const {
        if (max_diagonalizations_ < 0) {
            return true;
        }
        return fermi_surface_stats_.evaluated_vertices + missing.size() <=
               static_cast<std::uint64_t>(max_diagonalizations_);
    }

    void evaluate_frontier_vertices(std::span<const core::VertexId> missing) {
        evaluate_vertices(geometry_, cache_, evaluator_, missing);
    }

    SimplexClassification classify_frontier_simplices() const {
        return classify_frontier(
            geometry_,
            cache_,
            frontier_,
            mu_,
            min_feature_size_,
            margin_,
            tol_
        );
    }

    void accumulate_terminal_counts(const SimplexClassification &classification) {
        result_.n_cut_simplices += classification.visible_gapless_terminal;
        result_.n_feature_size_simplices += classification.inconclusive_terminal;
        result_.n_unresolved_simplices += classification.unresolved;
        terminal_surface_simplices_.insert(
            terminal_surface_simplices_.end(),
            classification.terminal_surface.begin(),
            classification.terminal_surface.end()
        );
    }

    void refine_requested_simplices(const SimplexClassification &classification) {
        const auto previous_active = simplex_set(active_simplices(geometry_));
        auto refined = classification.refine;
        geometry_.refine_active(refined, 1);
        frontier_ = next_frontier(
            geometry_,
            previous_active,
            classification.refine,
            refined
        );
    }

    void refine_until_terminal() {
        while (!frontier_.empty()) {
            const auto missing = missing_frontier_vertices();
            if (!fits_diagonalization_budget(missing)) {
                result_.converged = false;
                result_.n_unresolved_simplices += static_cast<std::int64_t>(frontier_.size());
                break;
            }

            evaluate_frontier_vertices(
                std::span<const core::VertexId>(missing.data(), missing.size())
            );
            auto classification = classify_frontier_simplices();
            accumulate_terminal_counts(classification);

            if (classification.refine.empty()) {
                frontier_.clear();
                result_.converged = result_.n_unresolved_simplices == 0;
                break;
            }
            refine_requested_simplices(classification);
        }
    }

    void extract_terminal_surface_mesh() {
        extract_terminal_surface(
            *model_,
            geometry_,
            cache_,
            std::span<const core::SimplexId>(
                terminal_surface_simplices_.data(),
                terminal_surface_simplices_.size()
            ),
            mu_,
            tol_,
            return_states_,
            result_
        );
    }

    std::shared_ptr<const HamiltonianModel> model_;
    double mu_ = 0.0;
    double min_feature_size_ = 0.0;
    std::int64_t max_diagonalizations_ = -1;
    double margin_ = 0.0;
    double tol_ = 1e-14;
    bool return_states_ = false;
    core::Geometry geometry_;
    SpectraCache cache_;
    VertexSpectraEvaluator evaluator_;
    std::vector<core::SimplexId> frontier_;
    std::vector<core::SimplexId> terminal_surface_simplices_;
    FermiSurfaceResult result_;
};

}  // namespace

FermiSurfaceResult run_fermi_surface(
    std::shared_ptr<const HamiltonianModel> model,
    double mu,
    double min_feature_size,
    std::int64_t max_diagonalizations,
    double margin,
    double tol,
    bool return_states
) {
    return FermiSurfaceRun(
        std::move(model),
        mu,
        min_feature_size,
        max_diagonalizations,
        margin,
        tol,
        return_states
    ).run();
}

}  // namespace lineartetrahedron::fermi_surface_detail
