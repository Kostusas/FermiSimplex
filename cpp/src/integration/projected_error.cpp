#include "integration/projected_error.h"

#include "core/linear_algebra.h"

#include <algorithm>
#include <complex>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace lineartetrahedron {
namespace core = adaptivesimplex::core;

namespace {

using Complex = std::complex<double>;

size_t column_major_index(size_t row, size_t column, size_t rows) {
    return row + column * rows;
}

double ambiguous_block_margin(
    const VertexSpectra &spectra,
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
    const core::VertexCache<VertexSpectra> &cache,
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
    const VertexSpectra &anchor,
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
    for (size_t column = 0; column < jdim; ++column) {
        for (size_t row = 0; row < ndof; ++row) {
            auto value = Complex{0.0, 0.0};
            for (size_t inner = 0; inner < ndof; ++inner) {
                value += matrix[column_major_index(row, inner, ndof)] *
                         vectors[column_major_index(inner, column, ndof)];
            }
            matrix_times_v[column_major_index(row, column, ndof)] = value;
        }
    }

    std::vector<Complex> result(jdim * jdim, Complex{0.0, 0.0});
    for (size_t column = 0; column < jdim; ++column) {
        for (size_t row = 0; row < jdim; ++row) {
            auto value = Complex{0.0, 0.0};
            for (size_t inner = 0; inner < ndof; ++inner) {
                value += std::conj(vectors[column_major_index(inner, row, ndof)]) *
                         matrix_times_v[column_major_index(inner, column, ndof)];
            }
            result[column_major_index(row, column, jdim)] = value;
        }
    }
    return result;
}

std::vector<Complex> project_vertex_hamiltonian(
    const VertexSpectra &spectra,
    std::span<const Complex> vectors,
    size_t jdim
) {
    const auto ndof = spectra.eigenvalues.size();
    std::vector<Complex> overlap(ndof * jdim, Complex{0.0, 0.0});
    for (size_t column = 0; column < jdim; ++column) {
        for (size_t band = 0; band < ndof; ++band) {
            auto value = Complex{0.0, 0.0};
            for (size_t row = 0; row < ndof; ++row) {
                value += std::conj(spectra.eigenvectors[column_major_index(row, band, ndof)]) *
                         vectors[column_major_index(row, column, ndof)];
            }
            overlap[column_major_index(band, column, ndof)] = value;
        }
    }

    std::vector<Complex> result(jdim * jdim, Complex{0.0, 0.0});
    for (size_t column = 0; column < jdim; ++column) {
        for (size_t row = 0; row < jdim; ++row) {
            auto value = Complex{0.0, 0.0};
            for (size_t band = 0; band < ndof; ++band) {
                value += spectra.eigenvalues[band] *
                         std::conj(overlap[column_major_index(band, row, ndof)]) *
                         overlap[column_major_index(band, column, ndof)];
            }
            result[column_major_index(row, column, jdim)] = value;
        }
    }
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

ProjectedErrorEstimate estimate_projected_error(
    const IntegrationWorkspace &workspace,
    const core::Geometry &geometry,
    core::SimplexId simplex_id,
    const core::VertexCache<VertexSpectra> &cache,
    size_t lower_band,
    size_t upper_band
) {
    const auto ndof = workspace.ndof();
    if (lower_band > upper_band || upper_band > ndof) {
        throw std::runtime_error("estimate_projected_error: invalid ambiguous band interval");
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
    samples.reserve(simplex.vertex_ids.size() * (simplex.vertex_ids.size() - 1) / 2 + 1);
    for (size_t left = 0; left < simplex.vertex_ids.size(); ++left) {
        for (size_t right = left + 1; right < simplex.vertex_ids.size(); ++right) {
            samples.push_back(barycentric_sample(simplex.vertex_ids.size(), left, right));
        }
    }
    samples.push_back(center_sample(simplex.vertex_ids.size()));

    auto estimate = ProjectedErrorEstimate{};
    for (const auto &weights : samples) {
        const auto point = sample_point(geometry, simplex, weights);
        const auto hamiltonian =
            workspace.evaluate_hamiltonian(std::span<const double>(point.data(), point.size()));
        auto residual_projection = project_matrix(hamiltonian, vectors, ndof, jdim);
        for (size_t vertex = 0; vertex < vertex_projections.size(); ++vertex) {
            add_scaled(residual_projection, vertex_projections[vertex], -weights[vertex]);
        }

        std::vector<double> eigenvalues;
        diagonalize_hermitian_in_place(
            residual_projection,
            eigenvalues,
            jdim,
            false,
            "estimate_projected_error"
        );
        estimate.rho_up = std::max(estimate.rho_up, std::max(eigenvalues.back(), 0.0));
        estimate.rho_down = std::max(estimate.rho_down, std::max(-eigenvalues.front(), 0.0));
    }
    return estimate;
}

}  // namespace lineartetrahedron
