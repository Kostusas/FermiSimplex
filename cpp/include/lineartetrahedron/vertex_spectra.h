#pragma once

#include "lineartetrahedron/tight_binding.h"

#include <adaptivesimplex/core/geometry.h>

#include <complex>
#include <memory>
#include <vector>

namespace lineartetrahedron {

struct VertexSpectra {
    std::vector<double> eigenvalues;
    std::vector<std::complex<double>> eigenvectors;
};

class VertexSpectraEvaluator {
public:
    explicit VertexSpectraEvaluator(std::shared_ptr<TightBindingModel> model);

    size_t ndim() const noexcept { return model_->ndim(); }
    size_t ndof() const noexcept { return model_->ndof(); }

    VertexSpectra evaluate(
        const adaptivesimplex::core::Geometry &geometry,
        adaptivesimplex::core::VertexId vertex_id
    );

private:
    VertexSpectra diagonalize_reduced_point(const std::vector<double> &reduced_point);

    std::shared_ptr<TightBindingModel> model_;
};

}  // namespace lineartetrahedron
