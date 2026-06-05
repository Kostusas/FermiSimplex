#include "lineartetrahedron/integration_state.h"

#include <stdexcept>
#include <utility>

namespace lineartetrahedron {
namespace core = adaptivesimplex::core;

namespace {

size_t supported_dimension(const std::shared_ptr<TightBindingModel> &model) {
    if (!model) {
        throw std::runtime_error("IntegrationState: model must not be null");
    }
    const size_t ndim = model->ndim();
    if (ndim < 1 || ndim > 3) {
        throw std::runtime_error("LinearTetrahedron supports dimensions 1, 2, and 3");
    }
    return ndim;
}

core::Geometry make_root_geometry(const std::shared_ptr<TightBindingModel> &model) {
    const size_t ndim = supported_dimension(model);
    return core::root_geometry(ndim, ndim == 1 ? 2U : 1U);
}

}  // namespace

IntegrationState::IntegrationState(
    std::shared_ptr<TightBindingModel> model,
    KeyArray keys,
    ComponentIndexArray component_rows,
    ComponentIndexArray component_cols,
    ComponentIndexArray component_key_indices,
    double tol
) : model_(std::move(model)),
    geometry_(make_root_geometry(model_)),
    evaluator_(model_),
    tol_(tol) {
    ndim_ = model_->ndim();
    ndof_ = model_->ndof();
    if (keys.shape(1) != ndim_) {
        throw std::runtime_error("IntegrationState: density key dimension mismatch");
    }
    n_keys_ = keys.shape(0);
    if (n_keys_ == 0) {
        throw std::runtime_error("IntegrationState: keys must be non-empty");
    }
    keys_.assign(keys.data(), keys.data() + n_keys_ * ndim_);

    const size_t count = component_rows.shape(0);
    if (component_cols.shape(0) != count || component_key_indices.shape(0) != count) {
        throw std::runtime_error("IntegrationState: selected component arrays must match");
    }
    components_.reserve(count);
    for (size_t index = 0; index < count; ++index) {
        const auto row = component_rows.data()[index];
        const auto col = component_cols.data()[index];
        const auto key_index = component_key_indices.data()[index];
        if (row < 0 || col < 0 || key_index < 0) {
            throw std::runtime_error("IntegrationState: selected component indices must be non-negative");
        }
        if (static_cast<size_t>(row) >= ndof_ || static_cast<size_t>(col) >= ndof_) {
            throw std::runtime_error("IntegrationState: selected row/column out of bounds");
        }
        if (static_cast<size_t>(key_index) >= n_keys_) {
            throw std::runtime_error("IntegrationState: selected key index out of bounds");
        }
        components_.push_back(DensityComponent{
            static_cast<size_t>(row),
            static_cast<size_t>(col),
            static_cast<size_t>(key_index),
        });
    }
}

}  // namespace lineartetrahedron
