#include "core/vertex_spectra.h"

#include "core/linear_algebra.h"

#include <stdexcept>
#include <utility>

namespace lineartetrahedron {

VertexSpectraEvaluator::VertexSpectraEvaluator(
    std::shared_ptr<const HamiltonianModel> model
) : model_(std::move(model)) {
    if (!model_) {
        throw std::runtime_error("VertexSpectraEvaluator: model must not be null");
    }
}

VertexSpectra VertexSpectraEvaluator::evaluate(
    const adaptivesimplex::core::Geometry &geometry,
    adaptivesimplex::core::VertexId vertex_id
) {
    const auto reduced_point = geometry.vertices().dyadic_vertex(vertex_id).to_point();
    return evaluate_reduced_point(std::span<const double>(reduced_point.data(), reduced_point.size()));
}

VertexSpectra VertexSpectraEvaluator::evaluate_reduced_point(
    std::span<const double> reduced_point
) const {
    auto h = model_->evaluate_reduced_point_raw(reduced_point.data());
    VertexSpectra entry;
    diagonalize_hermitian_in_place(
        h,
        entry.eigenvalues,
        model_->ndof(),
        true,
        "VertexSpectraEvaluator"
    );
    entry.eigenvectors = std::move(h);
    return entry;
}

VertexEigenvaluesEvaluator::VertexEigenvaluesEvaluator(
    std::shared_ptr<const HamiltonianModel> model
) : model_(std::move(model)) {
    if (!model_) {
        throw std::runtime_error("VertexEigenvaluesEvaluator: model must not be null");
    }
}

std::vector<double> VertexEigenvaluesEvaluator::evaluate_reduced_point(
    std::span<const double> reduced_point
) const {
    auto h = model_->evaluate_reduced_point_raw(reduced_point.data());
    std::vector<double> eigenvalues;
    diagonalize_hermitian_in_place(
        h,
        eigenvalues,
        model_->ndof(),
        false,
        "VertexEigenvaluesEvaluator"
    );
    return eigenvalues;
}

}  // namespace lineartetrahedron
