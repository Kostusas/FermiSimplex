#include "lineartetrahedron/integration_state.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
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

std::uint64_t component_pair_key(size_t row, size_t col, size_t ndof) {
    return static_cast<std::uint64_t>(row * ndof + col);
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
    component_pair_groups_.reserve(std::min(count, ndof_ * ndof_));
    std::unordered_map<std::uint64_t, size_t> group_index_by_pair;
    group_index_by_pair.reserve(std::min(count, ndof_ * ndof_));
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
        const auto component_index = components_.size();
        components_.push_back(DensityComponent{
            static_cast<size_t>(row),
            static_cast<size_t>(col),
            static_cast<size_t>(key_index),
        });
        const auto pair_key = component_pair_key(
            static_cast<size_t>(row),
            static_cast<size_t>(col),
            ndof_
        );
        auto group_index = group_index_by_pair.find(pair_key);
        if (group_index == group_index_by_pair.end()) {
            const auto new_group_index = component_pair_groups_.size();
            component_pair_groups_.push_back(DensityComponentPairGroup{
                static_cast<size_t>(row),
                static_cast<size_t>(col),
                {},
            });
            group_index = group_index_by_pair.emplace(pair_key, new_group_index).first;
        }
        component_pair_groups_[group_index->second].contributions.push_back(DensityComponentContribution{
            component_index,
            static_cast<size_t>(key_index),
        });
    }
}

}  // namespace lineartetrahedron
