#include "integration/projected_error.h"

#include "linalg/blas_lapack.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <span>
#include <stdexcept>
#include <vector>

namespace fermisimplex {
namespace core = adaptivesimplex::core;

namespace {

using Complex = std::complex<double>;
using Matrix = std::vector<Complex>;

std::size_t column_major_index(
    std::size_t row,
    std::size_t column,
    std::size_t rows
) {
    return row + column * rows;
}

Matrix multiply(
    char left_operation,
    char right_operation,
    std::size_t rows,
    std::size_t columns,
    std::size_t inner_dimension,
    std::span<const Complex> left,
    std::size_t left_leading_dimension,
    std::span<const Complex> right,
    std::size_t right_leading_dimension
) {
    auto result = Matrix(rows * columns, Complex{0.0, 0.0});
    linalg::matrix_multiply(
        left_operation,
        right_operation,
        rows,
        columns,
        inner_dimension,
        Complex{1.0, 0.0},
        left.data(),
        left_leading_dimension,
        right.data(),
        right_leading_dimension,
        Complex{0.0, 0.0},
        result.data(),
        rows
    );
    return result;
}

Matrix band_vectors(
    const Eigensystem &eigensystem,
    std::size_t lower_band,
    std::size_t upper_band
) {
    const auto ndof = eigensystem.eigenvalues.size();
    const auto band_count = upper_band - lower_band;
    auto result = Matrix(ndof * band_count, Complex{0.0, 0.0});
    for (std::size_t column = 0; column < band_count; ++column) {
        for (std::size_t row = 0; row < ndof; ++row) {
            result[column_major_index(row, column, ndof)] =
                eigensystem.eigenvectors[column_major_index(
                    row,
                    lower_band + column,
                    ndof
                )];
        }
    }
    return result;
}

std::vector<Matrix> selected_vertex_blocks(
    const core::Simplex &simplex,
    const EigensystemCache &cache,
    std::size_t lower_band,
    std::size_t upper_band
) {
    auto blocks = std::vector<Matrix>{};
    blocks.reserve(simplex.vertex_ids.size());
    for (const auto vertex_id : simplex.vertex_ids) {
        blocks.push_back(
            band_vectors(cache.get(vertex_id), lower_band, upper_band)
        );
    }
    return blocks;
}

Matrix principal_angle_midpoint_subspace(
    const Matrix &first,
    const Matrix &second,
    std::size_t ndof,
    std::size_t band_count
) {
    auto overlap = multiply(
        'C',
        'N',
        band_count,
        band_count,
        ndof,
        first,
        ndof,
        second,
        ndof
    );
    auto singular_values = std::vector<double>{};
    auto right_adjoint = Matrix{};
    linalg::singular_value_decompose_in_place(
        overlap,
        singular_values,
        right_adjoint,
        band_count,
        band_count,
        "edge principal angles"
    );

    const auto aligned_first = multiply(
        'N',
        'N',
        ndof,
        band_count,
        band_count,
        first,
        ndof,
        overlap,
        band_count
    );
    const auto aligned_second = multiply(
        'N',
        'C',
        ndof,
        band_count,
        band_count,
        second,
        ndof,
        right_adjoint,
        band_count
    );
    auto result = Matrix(ndof * band_count, Complex{0.0, 0.0});
    for (std::size_t column = 0; column < band_count; ++column) {
        const auto cosine =
            std::clamp(singular_values[column], 0.0, 1.0);
        const auto normalization =
            1.0 / std::sqrt(2.0 + 2.0 * cosine);
        for (std::size_t row = 0; row < ndof; ++row) {
            result[column_major_index(row, column, ndof)] =
                normalization * (
                    aligned_first[column_major_index(
                        row,
                        column,
                        ndof
                    )] +
                    aligned_second[column_major_index(
                        row,
                        column,
                        ndof
                    )]
                );
        }
    }
    return result;
}

std::vector<double> projected_eigenvalues(
    std::span<const Complex> hamiltonian,
    std::span<const Complex> vectors,
    std::size_t ndof,
    std::size_t band_count
) {
    const auto hamiltonian_times_vectors = multiply(
        'N',
        'N',
        ndof,
        band_count,
        ndof,
        hamiltonian,
        ndof,
        vectors,
        ndof
    );
    auto projected = multiply(
        'C',
        'N',
        band_count,
        band_count,
        ndof,
        vectors,
        ndof,
        hamiltonian_times_vectors,
        ndof
    );
    auto eigenvalues = std::vector<double>{};
    linalg::diagonalize_hermitian_in_place(
        projected,
        eigenvalues,
        band_count,
        false,
        "projected sampled Hamiltonian"
    );
    return eigenvalues;
}

std::vector<double> center_weights(std::size_t vertex_count) {
    return std::vector<double>(
        vertex_count,
        1.0 / static_cast<double>(vertex_count)
    );
}

std::vector<double> edge_midpoint_weights(
    std::size_t vertex_count,
    std::size_t first,
    std::size_t second
) {
    auto weights = std::vector<double>(vertex_count, 0.0);
    weights[first] = 0.5;
    weights[second] = 0.5;
    return weights;
}

std::vector<double> sample_point(
    const core::Geometry &geometry,
    const core::Simplex &simplex,
    std::span<const double> weights
) {
    auto point = std::vector<double>(geometry.ndim(), 0.0);
    for (std::size_t vertex = 0; vertex < simplex.vertex_ids.size(); ++vertex) {
        const auto vertex_point =
            geometry.vertices().dyadic_vertex(
                simplex.vertex_ids[vertex]
            ).to_point();
        for (std::size_t axis = 0; axis < geometry.ndim(); ++axis) {
            point[axis] += weights[vertex] * vertex_point[axis];
        }
    }
    return point;
}

void add_projected_probe(
    ProjectedErrorEstimate &estimate,
    const SpectralMesh &mesh,
    const core::Geometry &geometry,
    const core::Simplex &simplex,
    const Matrix &subspace,
    std::span<const double> weights,
    std::size_t lower_band,
    std::size_t upper_band
) {
    const auto ndof = mesh.ndof();
    const auto band_count = upper_band - lower_band;
    const auto point = sample_point(geometry, simplex, weights);
    const auto hamiltonian = mesh.hamiltonian(point);
    const auto values = projected_eigenvalues(
        hamiltonian,
        subspace,
        ndof,
        band_count
    );

    const auto &cache = mesh.eigensystems();
    for (std::size_t offset = 0; offset < band_count; ++offset) {
        auto interpolated = 0.0;
        for (std::size_t vertex = 0;
             vertex < simplex.vertex_ids.size();
             ++vertex) {
            interpolated += weights[vertex] *
                cache.get(simplex.vertex_ids[vertex])
                    .eigenvalues[lower_band + offset];
        }
        const auto difference = values[offset] - interpolated;
        estimate.positive_estimate =
            std::max(estimate.positive_estimate, difference);
        estimate.negative_estimate =
            std::max(estimate.negative_estimate, -difference);
    }
}

void add_exact_probe(
    ProjectedErrorEstimate &estimate,
    const SpectralMesh &mesh,
    const core::Geometry &geometry,
    const core::Simplex &simplex,
    std::span<const double> weights,
    std::size_t lower_band,
    std::size_t upper_band
) {
    const auto point = sample_point(geometry, simplex, weights);
    auto hamiltonian = mesh.hamiltonian(point);
    auto values = std::vector<double>{};
    linalg::diagonalize_hermitian_in_place(
        hamiltonian,
        values,
        mesh.ndof(),
        false,
        "exact sampled Hamiltonian"
    );

    const auto &cache = mesh.eigensystems();
    for (std::size_t offset = 0;
         offset < upper_band - lower_band;
         ++offset) {
        auto interpolated = 0.0;
        for (std::size_t vertex = 0;
             vertex < simplex.vertex_ids.size();
             ++vertex) {
            interpolated += weights[vertex] *
                cache.get(simplex.vertex_ids[vertex])
                    .eigenvalues[lower_band + offset];
        }
        const auto difference =
            values[lower_band + offset] - interpolated;
        estimate.positive_estimate =
            std::max(estimate.positive_estimate, difference);
        estimate.negative_estimate =
            std::max(estimate.negative_estimate, -difference);
    }
}

void merge_estimate(
    ProjectedErrorEstimate &estimate,
    const ProjectedErrorEstimate &addition
) {
    estimate.positive_estimate = std::max(
        estimate.positive_estimate,
        addition.positive_estimate
    );
    estimate.negative_estimate = std::max(
        estimate.negative_estimate,
        addition.negative_estimate
    );
}

ProjectedErrorCache::EdgeKey canonical_edge_key(
    const core::Simplex &simplex,
    std::size_t first,
    std::size_t second,
    std::size_t lower_band,
    std::size_t upper_band
) {
    auto first_id = simplex.vertex_ids[first];
    auto second_id = simplex.vertex_ids[second];
    if (second_id < first_id) {
        std::swap(first_id, second_id);
    }
    return {
        first_id,
        second_id,
        lower_band,
        upper_band,
    };
}

template <class Evaluate>
ProjectedErrorEstimate cached_edge_estimate(
    ProjectedErrorCache *cache,
    const ProjectedErrorCache::EdgeKey &key,
    Evaluate &&evaluate
) {
    if (cache == nullptr) {
        return evaluate();
    }
    const auto found = cache->edge_estimates.find(key);
    if (found != cache->edge_estimates.end()) {
        return found->second;
    }
    const auto estimate = evaluate();
    cache->edge_estimates.emplace(key, estimate);
    return estimate;
}

}  // namespace

ProjectedErrorEstimate estimate_projected_error(
    const SpectralMesh &mesh,
    core::SimplexId simplex_id,
    std::size_t lower_band,
    std::size_t upper_band,
    ProjectedErrorCache *cache
) {
    const auto ndof = mesh.ndof();
    if (lower_band > upper_band || upper_band > ndof) {
        throw std::runtime_error(
            "estimate_projected_error: invalid selected band interval"
        );
    }
    const auto band_count = upper_band - lower_band;
    if (band_count == 0) {
        return {};
    }

    const auto &geometry = mesh.geometry();
    const auto &simplex = geometry.simplices().simplex(simplex_id);
    auto estimate = ProjectedErrorEstimate{};
    const auto vertex_count = simplex.vertex_ids.size();
    const auto use_full_space = band_count == ndof;
    const auto vertex_blocks = use_full_space
        ? std::vector<Matrix>{}
        : selected_vertex_blocks(
              simplex,
              mesh.eigensystems(),
              lower_band,
              upper_band
          );

    if (vertex_count > 2) {
        const auto weights = center_weights(vertex_count);
        add_exact_probe(
            estimate,
            mesh,
            geometry,
            simplex,
            weights,
            lower_band,
            upper_band
        );
    }

    for (std::size_t first = 0; first < vertex_count; ++first) {
        for (std::size_t second = first + 1;
             second < vertex_count;
             ++second) {
            const auto key = canonical_edge_key(
                simplex,
                first,
                second,
                lower_band,
                upper_band
            );
            const auto evaluate_edge = [&] {
                const auto weights =
                    edge_midpoint_weights(vertex_count, first, second);
                auto edge_estimate = ProjectedErrorEstimate{};
                if (use_full_space) {
                    add_exact_probe(
                        edge_estimate,
                        mesh,
                        geometry,
                        simplex,
                        weights,
                        lower_band,
                        upper_band
                    );
                } else {
                    const auto subspace =
                        principal_angle_midpoint_subspace(
                            vertex_blocks[first],
                            vertex_blocks[second],
                            ndof,
                            band_count
                        );
                    add_projected_probe(
                        edge_estimate,
                        mesh,
                        geometry,
                        simplex,
                        subspace,
                        weights,
                        lower_band,
                        upper_band
                    );
                }
                return edge_estimate;
            };
            merge_estimate(
                estimate,
                cached_edge_estimate(cache, key, evaluate_edge)
            );
        }
    }
    return estimate;
}

}  // namespace fermisimplex
