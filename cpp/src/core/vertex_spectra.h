#pragma once

#include "core/tight_binding.h"

#include <adaptivesimplex/core/geometry.h>

#include <complex>
#include <memory>
#include <span>
#include <vector>

namespace lineartetrahedron {

struct VertexSpectra {
    std::vector<double> eigenvalues;
    std::vector<std::complex<double>> eigenvectors;
};

class VertexSpectraEvaluator {
public:
    explicit VertexSpectraEvaluator(std::shared_ptr<const HamiltonianModel> model);

    size_t ndim() const noexcept { return model_->ndim(); }
    size_t ndof() const noexcept { return model_->ndof(); }

    VertexSpectra evaluate(
        const adaptivesimplex::core::Geometry &geometry,
        adaptivesimplex::core::VertexId vertex_id
    );
    VertexSpectra evaluate_reduced_point(std::span<const double> reduced_point) const;

private:
    std::shared_ptr<const HamiltonianModel> model_;
};

class VertexEigenvaluesEvaluator {
public:
    explicit VertexEigenvaluesEvaluator(std::shared_ptr<const HamiltonianModel> model);

    size_t ndim() const noexcept { return model_->ndim(); }
    size_t ndof() const noexcept { return model_->ndof(); }

    std::vector<double> evaluate_reduced_point(std::span<const double> reduced_point) const;

private:
    std::shared_ptr<const HamiltonianModel> model_;
};

}  // namespace lineartetrahedron
