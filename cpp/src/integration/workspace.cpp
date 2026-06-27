#include "integration/workspace.h"

#include <stdexcept>
#include <utility>

namespace lineartetrahedron {
namespace core = adaptivesimplex::core;

namespace {

size_t supported_dimension(const std::shared_ptr<const HamiltonianModel> &model) {
    if (!model) {
        throw std::runtime_error("IntegrationWorkspace: model must not be null");
    }
    const size_t ndim = model->ndim();
    if (ndim < 1) {
        throw std::runtime_error("LinearTetrahedron dimension must be positive");
    }
    return ndim;
}

core::Geometry make_root_geometry(const std::shared_ptr<const HamiltonianModel> &model) {
    const size_t ndim = supported_dimension(model);
    return core::root_geometry(ndim, ndim == 1 ? 2U : 1U);
}

}  // namespace

IntegrationWorkspace::IntegrationWorkspace(
    std::shared_ptr<const HamiltonianModel> model,
    double tol
) : model_(std::move(model)),
    geometry_(make_root_geometry(model_)),
    evaluator_(model_),
    tol_(tol) {
    ndim_ = model_->ndim();
    ndof_ = model_->ndof();
}

VertexSpectra IntegrationWorkspace::evaluate_vertex(std::span<const double> reduced_point) const {
    return evaluator_.evaluate_reduced_point(reduced_point);
}

}  // namespace lineartetrahedron
