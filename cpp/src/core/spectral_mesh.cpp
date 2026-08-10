#include <fermisimplex/spectral_mesh.h>

#include "core/simplex_geometry.h"
#include "linalg/blas_lapack.h"

#include <adaptivesimplex/core/root_mesh.h>
#include <adaptivesimplex/cut/simplex_moments.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <utility>

namespace fermisimplex {
namespace core = adaptivesimplex::core;
namespace {

std::shared_ptr<const HamiltonianModel> validate_model(
    std::shared_ptr<const HamiltonianModel> model
) {
    if (!model) {
        throw std::runtime_error("SpectralMesh: model must not be null");
    }
    if (model->ndim() == 0 || model->ndof() == 0) {
        throw std::runtime_error("SpectralMesh: dimensions must be positive");
    }
    return model;
}

double validate_tolerance(double tolerance) {
    if (!std::isfinite(tolerance) || tolerance < 0.0) {
        throw std::runtime_error("SpectralMesh: tolerance must be finite and non-negative");
    }
    return tolerance;
}

}  // namespace

SpectralMesh::SpectralMesh(
    std::shared_ptr<const HamiltonianModel> model,
    double tolerance,
    std::uint32_t root_level
) : model_(validate_model(std::move(model))),
    geometry_(core::root_geometry(model_->ndim(), root_level)),
    tolerance_(validate_tolerance(tolerance)) {}

Eigensystem SpectralMesh::spectrum(std::span<const double> reduced_point) const {
    auto matrix = hamiltonian(reduced_point);
    auto result = Eigensystem{};
    linalg::diagonalize_hermitian_in_place(
        matrix,
        result.eigenvalues,
        model_->ndof(),
        true,
        "SpectralMesh"
    );
    result.eigenvectors = std::move(matrix);
    return result;
}

std::vector<std::complex<double>> SpectralMesh::hamiltonian(
    std::span<const double> reduced_point
) const {
    return model_->evaluate(reduced_point);
}

std::vector<core::VertexId> SpectralMesh::active_vertex_ids() const {
    auto used = std::vector<bool>(geometry_.vertices().size(), false);
    for (const auto simplex_id : geometry_.simplices().active_simplices()) {
        for (const auto vertex_id :
             geometry_.simplices().simplex(simplex_id).vertex_ids) {
            used[vertex_id] = true;
        }
    }

    auto result = std::vector<core::VertexId>{};
    result.reserve(geometry_.n_active_vertices());
    for (std::size_t vertex_id = 0; vertex_id < used.size(); ++vertex_id) {
        if (used[vertex_id]) {
            result.push_back(static_cast<core::VertexId>(vertex_id));
        }
    }
    return result;
}

std::vector<double> SpectralMesh::occupied_weights(double mu) const {
    if (!std::isfinite(mu)) {
        throw std::runtime_error("chemical potential mu must be finite");
    }

    const auto vertex_ids = active_vertex_ids();
    auto compact_index = std::vector<std::size_t>(
        geometry_.vertices().size(), 0
    );
    for (std::size_t index = 0; index < vertex_ids.size(); ++index) {
        compact_index[vertex_ids[index]] = index;
        if (!eigensystems_.contains(vertex_ids[index])) {
            throw std::runtime_error(
                "SpectralMesh: active mesh eigensystems are not fully cached"
            );
        }
    }

    auto result = std::vector<double>(vertex_ids.size() * ndof(), 0.0);
    namespace cut = adaptivesimplex::cut;
    for (const auto simplex_id : geometry_.simplices().active_simplices()) {
        const auto &simplex = geometry_.simplices().simplex(simplex_id);
        for (std::size_t band = 0; band < ndof(); ++band) {
            auto moments = cut::simplex_moments(
                geometry_,
                simplex_id,
                [&](core::VertexId vertex_id) {
                    return eigensystems_.get(vertex_id).eigenvalues[band];
                },
                cut::LevelOptions{
                    .level = mu,
                    .level_tolerance = tolerance_,
                }
            );
            if (moments.kind == cut::SimplexCutKind::on_level) {
                std::fill(
                    moments.barycentric_moments.begin(),
                    moments.barycentric_moments.end(),
                    0.5 * simplex.volume /
                        static_cast<double>(simplex.vertex_ids.size())
                );
            }
            for (std::size_t local = 0;
                 local < simplex.vertex_ids.size();
                 ++local) {
                const auto vertex = compact_index[simplex.vertex_ids[local]];
                result[vertex * ndof() + band] +=
                    moments.barycentric_moments[local];
            }
        }
    }
    return result;
}

double SpectralMesh::linearization_error_bound(
    core::SimplexId simplex_id,
    double curvature_bound
) const {
    if (!std::isfinite(curvature_bound) || curvature_bound < 0.0) {
        throw std::runtime_error(
            "curvature_bound must be finite and non-negative"
        );
    }
    return symmetric_linearization_error_bound(
        curvature_bound,
        simplex_diameter(geometry_, simplex_id)
    );
}

}  // namespace fermisimplex
