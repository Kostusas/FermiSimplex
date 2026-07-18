#include "integration/projected_error.h"

#include "linalg/blas_lapack.h"

#include <algorithm>
#include <complex>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace fermisimplex {
namespace core = adaptivesimplex::core;

namespace {

using Complex = std::complex<double>;

size_t column_major_index(size_t row, size_t column, size_t rows) {
    return row + column * rows;
}

double ambiguous_block_margin(
    const Eigensystem &spectra,
    size_t lower_band,
    size_t upper_band
) {
    const auto ndof = spectra.eigenvalues.size();
    auto lower_margin = std::numeric_limits<double>::infinity();
    auto upper_margin = std::numeric_limits<double>::infinity();
    if (lower_band > 0) {
        lower_margin = spectra.eigenvalues[lower_band] - spectra.eigenvalues[lower_band - 1];
    }
    if (upper_band < ndof) {
        upper_margin = spectra.eigenvalues[upper_band] - spectra.eigenvalues[upper_band - 1];
    }
    return std::min(lower_margin, upper_margin);
}

size_t largest_gap_anchor(
    const core::Geometry &geometry,
    core::SimplexId simplex_id,
    const core::VertexCache<Eigensystem> &cache,
    size_t lower_band,
    size_t upper_band
) {
    const auto &simplex = geometry.simplices().simplex(simplex_id);
    auto best_local_index = size_t{0};
    auto best_margin = -std::numeric_limits<double>::infinity();
    for (size_t local_index = 0; local_index < simplex.vertex_ids.size(); ++local_index) {
        const auto margin =
            ambiguous_block_margin(cache.get(simplex.vertex_ids[local_index]), lower_band, upper_band);
        if (margin > best_margin) {
            best_margin = margin;
            best_local_index = local_index;
        }
    }
    return best_local_index;
}

std::vector<Complex> ambiguous_anchor_vectors(
    const Eigensystem &anchor,
    size_t lower_band,
    size_t upper_band
) {
    const auto ndof = anchor.eigenvalues.size();
    const auto jdim = upper_band - lower_band;
    std::vector<Complex> result(ndof * jdim, Complex{0.0, 0.0});
    for (size_t column = 0; column < jdim; ++column) {
        for (size_t row = 0; row < ndof; ++row) {
            result[column_major_index(row, column, ndof)] =
                anchor.eigenvectors[column_major_index(row, lower_band + column, ndof)];
        }
    }
    return result;
}

std::vector<double> barycentric_sample(size_t nvertices, size_t first, size_t second) {
    std::vector<double> weights(nvertices, 0.0);
    weights[first] = 0.5;
    weights[second] = 0.5;
    return weights;
}

std::vector<double> center_sample(size_t nvertices) {
    return std::vector<double>(nvertices, 1.0 / static_cast<double>(nvertices));
}

std::vector<double> sample_point(
    const core::Geometry &geometry,
    const core::Simplex &simplex,
    std::span<const double> weights
) {
    std::vector<double> point(geometry.ndim(), 0.0);
    for (size_t vertex = 0; vertex < simplex.vertex_ids.size(); ++vertex) {
        const auto vertex_point =
            geometry.vertices().dyadic_vertex(simplex.vertex_ids[vertex]).to_point();
        for (size_t axis = 0; axis < geometry.ndim(); ++axis) {
            point[axis] += weights[vertex] * vertex_point[axis];
        }
    }
    return point;
}

std::vector<Complex> project_matrix(
    std::span<const Complex> matrix,
    std::span<const Complex> vectors,
    size_t ndof,
    size_t jdim
) {
    std::vector<Complex> matrix_times_v(ndof * jdim, Complex{0.0, 0.0});
    linalg::matrix_multiply(
        'N',
        'N',
        ndof,
        jdim,
        ndof,
        Complex{1.0, 0.0},
        matrix.data(),
        ndof,
        vectors.data(),
        ndof,
        Complex{0.0, 0.0},
        matrix_times_v.data(),
        ndof
    );

    std::vector<Complex> result(jdim * jdim, Complex{0.0, 0.0});
    linalg::matrix_multiply(
        'C',
        'N',
        jdim,
        jdim,
        ndof,
        Complex{1.0, 0.0},
        vectors.data(),
        ndof,
        matrix_times_v.data(),
        ndof,
        Complex{0.0, 0.0},
        result.data(),
        jdim
    );
    return result;
}

std::vector<Complex> project_vertex_hamiltonian(
    const Eigensystem &spectra,
    std::span<const Complex> vectors,
    size_t jdim
) {
    const auto ndof = spectra.eigenvalues.size();
    std::vector<Complex> overlap(ndof * jdim, Complex{0.0, 0.0});
    linalg::matrix_multiply(
        'C',
        'N',
        ndof,
        jdim,
        ndof,
        Complex{1.0, 0.0},
        spectra.eigenvectors.data(),
        ndof,
        vectors.data(),
        ndof,
        Complex{0.0, 0.0},
        overlap.data(),
        ndof
    );

    auto weighted_overlap = overlap;
    for (size_t column = 0; column < jdim; ++column) {
        for (size_t band = 0; band < ndof; ++band) {
            weighted_overlap[column_major_index(band, column, ndof)] *=
                spectra.eigenvalues[band];
        }
    }

    std::vector<Complex> result(jdim * jdim, Complex{0.0, 0.0});
    linalg::matrix_multiply(
        'C',
        'N',
        jdim,
        jdim,
        ndof,
        Complex{1.0, 0.0},
        overlap.data(),
        ndof,
        weighted_overlap.data(),
        ndof,
        Complex{0.0, 0.0},
        result.data(),
        jdim
    );
    return result;
}

void add_scaled(
    std::vector<Complex> &target,
    std::span<const Complex> source,
    double scale
) {
    for (size_t index = 0; index < target.size(); ++index) {
        target[index] += scale * source[index];
    }
}

}  // namespace

ProjectedResidualEstimate estimate_projected_residual(
    const SpectralMesh &mesh,
    core::SimplexId simplex_id,
    size_t lower_band,
    size_t upper_band
) {
    const auto &geometry = mesh.geometry();
    const auto &cache = mesh.eigensystems();
    const auto ndof = mesh.ndof();
    if (lower_band > upper_band || upper_band > ndof) {
        throw std::runtime_error("estimate_projected_residual: invalid ambiguous band interval");
    }
    const auto jdim = upper_band - lower_band;
    if (jdim == 0) {
        return {};
    }

    const auto &simplex = geometry.simplices().simplex(simplex_id);
    const auto anchor_index =
        largest_gap_anchor(geometry, simplex_id, cache, lower_band, upper_band);
    const auto vectors =
        ambiguous_anchor_vectors(cache.get(simplex.vertex_ids[anchor_index]), lower_band, upper_band);

    std::vector<std::vector<Complex>> vertex_projections;
    vertex_projections.reserve(simplex.vertex_ids.size());
    for (const auto vertex_id : simplex.vertex_ids) {
        vertex_projections.push_back(project_vertex_hamiltonian(cache.get(vertex_id), vectors, jdim));
    }

    std::vector<std::vector<double>> samples;
    const auto nvertices = simplex.vertex_ids.size();
    samples.reserve(nvertices * (nvertices - 1) / 2 + (nvertices > 2 ? 1 : 0));
    for (size_t left = 0; left < simplex.vertex_ids.size(); ++left) {
        for (size_t right = left + 1; right < simplex.vertex_ids.size(); ++right) {
            samples.push_back(barycentric_sample(simplex.vertex_ids.size(), left, right));
        }
    }
    // In one dimension the barycenter is already the sole edge midpoint.
    if (nvertices > 2) {
        samples.push_back(center_sample(nvertices));
    }

    auto estimate = ProjectedResidualEstimate{};
    for (const auto &weights : samples) {
        const auto point = sample_point(geometry, simplex, weights);
        const auto hamiltonian = mesh.hamiltonian(
            std::span<const double>(point.data(), point.size())
        );
        auto residual_projection = project_matrix(hamiltonian, vectors, ndof, jdim);
        for (size_t vertex = 0; vertex < vertex_projections.size(); ++vertex) {
            add_scaled(residual_projection, vertex_projections[vertex], -weights[vertex]);
        }

        std::vector<double> eigenvalues;
        linalg::diagonalize_hermitian_in_place(
            residual_projection,
            eigenvalues,
            jdim,
            false,
            "estimate_projected_residual"
        );
        estimate.positive_estimate = std::max(
            estimate.positive_estimate,
            std::max(eigenvalues.back(), 0.0)
        );
        estimate.negative_estimate = std::max(
            estimate.negative_estimate,
            std::max(-eigenvalues.front(), 0.0)
        );
    }
    return estimate;
}

}  // namespace fermisimplex
