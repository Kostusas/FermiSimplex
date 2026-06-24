#include "internal.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <set>
#include <utility>

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

MarkResult classify_frontier(
    const core::Geometry &geometry,
    const SpectraCache &cache,
    const std::vector<core::SimplexId> &frontier,
    double mu,
    double min_feature_size,
    double margin,
    double tol,
    std::optional<simplex_certificate::GapPrecision> gap_precision
) {
    MarkResult result;
    for (const auto simplex_id : frontier) {
        const auto refinable = max_reduced_edge_length(geometry, simplex_id) > min_feature_size;
        const auto certificate = simplex_certificate::certify_simplex_gap(
            mu,
            geometry,
            simplex_id,
            cache,
            margin,
            tol,
            gap_precision
        );
        const auto status = certificate.status;

        if (status == simplex_certificate::SimplexCertificateStatus::CertifiedGapped) {
            continue;
        } else if (status == simplex_certificate::SimplexCertificateStatus::VisibleCut) {
            if (refinable) {
                result.marked.push_back(simplex_id);
            } else {
                result.surface_terminal.push_back(simplex_id);
                ++result.cut;
            }
        } else if (refinable) {
            result.marked.push_back(simplex_id);
        } else {
            result.surface_terminal.push_back(simplex_id);
            ++result.feature_size;
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
    const std::vector<core::SimplexId> &marked,
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
    for (const auto simplex_id : marked) {
        if (!refined_set.contains(simplex_id) && seen.insert(simplex_id).second) {
            result.push_back(simplex_id);
        }
    }
    return result;
}

class FermiSurfaceRun {
public:
    FermiSurfaceRun(
        std::shared_ptr<const HamiltonianModel> model,
        double mu,
        double min_feature_size,
        std::int64_t max_diagonalizations,
        double margin,
        double tol,
        bool return_states,
        std::optional<simplex_certificate::GapPrecision> gap_precision
    ) : model_(std::move(model)),
        mu_(mu),
        min_feature_size_(min_feature_size),
        max_diagonalizations_(max_diagonalizations),
        margin_(margin),
        tol_(tol),
        return_states_(return_states),
        gap_precision_(gap_precision),
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
        extract_terminal_surface();
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

    MarkResult classify_frontier_simplices() const {
        return classify_frontier(
            geometry_,
            cache_,
            frontier_,
            mu_,
            min_feature_size_,
            margin_,
            tol_,
            gap_precision_
        );
    }

    void accumulate_terminal_counts(const MarkResult &marks) {
        result_.n_cut_simplices += marks.cut;
        result_.n_feature_size_simplices += marks.feature_size;
        result_.n_unresolved_simplices += marks.unresolved;
        terminal_surface_simplices_.insert(
            terminal_surface_simplices_.end(),
            marks.surface_terminal.begin(),
            marks.surface_terminal.end()
        );
    }

    void refine_marked_simplices(const MarkResult &marks) {
        const auto previous_active = simplex_set(active_simplices(geometry_));
        auto refined = marks.marked;
        geometry_.refine_active(refined, 1);
        frontier_ = next_frontier(geometry_, previous_active, marks.marked, refined);
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
            auto marks = classify_frontier_simplices();
            accumulate_terminal_counts(marks);

            if (marks.marked.empty()) {
                frontier_.clear();
                result_.converged = result_.n_unresolved_simplices == 0;
                break;
            }
            refine_marked_simplices(marks);
        }
    }

    void extract_terminal_surface() {
        extract_surface(
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
    std::optional<simplex_certificate::GapPrecision> gap_precision_;
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
    bool return_states,
    std::optional<simplex_certificate::GapPrecision> gap_precision
) {
    return FermiSurfaceRun(
        std::move(model),
        mu,
        min_feature_size,
        max_diagonalizations,
        margin,
        tol,
        return_states,
        gap_precision
    ).run();
}

}  // namespace lineartetrahedron::fermi_surface_detail
