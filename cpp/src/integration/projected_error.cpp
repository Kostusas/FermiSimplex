#include "integration/projected_error.h"

#include "linalg/blas_lapack.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fermisimplex {
namespace core = adaptivesimplex::core;

namespace {

using Complex = std::complex<double>;

size_t column_major_index(size_t row, size_t column, size_t rows) {
    return row + column * rows;
}

std::vector<Complex> band_vectors(
    const Eigensystem &eigensystem,
    size_t lower_band,
    size_t upper_band
) {
    const auto ndof = eigensystem.eigenvalues.size();
    const auto band_count = upper_band - lower_band;
    std::vector<Complex> result(ndof * band_count, Complex{0.0, 0.0});
    for (size_t column = 0; column < band_count; ++column) {
        for (size_t row = 0; row < ndof; ++row) {
            result[column_major_index(row, column, ndof)] =
                eigensystem.eigenvectors[
                    column_major_index(row, lower_band + column, ndof)
                ];
        }
    }
    return result;
}

std::vector<double> barycentric_sample(
    size_t nvertices,
    size_t first,
    size_t second,
    double second_weight
) {
    std::vector<double> weights(nvertices, 0.0);
    weights[first] = 1.0 - second_weight;
    weights[second] = second_weight;
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

void add_scaled(
    std::vector<Complex> &target,
    std::span<const Complex> source,
    double scale
) {
    for (size_t index = 0; index < target.size(); ++index) {
        target[index] += scale * source[index];
    }
}

struct VertexSubspaceInterpolator {
    size_t band_count = 0;
    size_t union_dimension = 0;
    std::vector<Complex> union_vectors;
    std::vector<std::vector<Complex>> vertex_projectors;
};

VertexSubspaceInterpolator build_vertex_subspace_interpolator(
    const core::Simplex &simplex,
    const EigensystemCache &cache,
    size_t ndof,
    size_t lower_band,
    size_t upper_band,
    double tolerance
) {
    const auto band_count = upper_band - lower_band;
    const auto candidate_count = simplex.vertex_ids.size() * band_count;
    if (candidate_count >= ndof) {
        // Resolving and projecting through a union that may already span the
        // full Hilbert space costs more than one eigenvalue-only solve.
        return VertexSubspaceInterpolator{
            .band_count = band_count,
            .union_dimension = ndof,
            .union_vectors = {},
            .vertex_projectors = {},
        };
    }
    auto vertex_blocks = std::vector<std::vector<Complex>>{};
    vertex_blocks.reserve(simplex.vertex_ids.size());
    auto candidates =
        std::vector<Complex>(ndof * candidate_count, Complex{0.0, 0.0});
    for (size_t vertex = 0; vertex < simplex.vertex_ids.size(); ++vertex) {
        auto block = band_vectors(
            cache.get(simplex.vertex_ids[vertex]),
            lower_band,
            upper_band
        );
        for (size_t column = 0; column < band_count; ++column) {
            for (size_t row = 0; row < ndof; ++row) {
                candidates[column_major_index(
                    row,
                    vertex * band_count + column,
                    ndof
                )] = block[column_major_index(row, column, ndof)];
            }
        }
        vertex_blocks.push_back(std::move(block));
    }

    const auto numerical_margin =
        64.0 * std::numeric_limits<double>::epsilon() *
        static_cast<double>(std::max(ndof, candidate_count));
    const auto vector_threshold = std::sqrt(
        std::max(tolerance, numerical_margin) *
        static_cast<double>(simplex.vertex_ids.size())
    );
    auto union_vectors = std::vector<Complex>{};
    union_vectors.reserve(ndof * std::min(ndof, candidate_count));
    for (size_t candidate = 0;
         candidate < candidate_count && union_vectors.size() < ndof * ndof;
         ++candidate) {
        auto vector = std::vector<Complex>(ndof, Complex{0.0, 0.0});
        for (size_t row = 0; row < ndof; ++row) {
            vector[row] =
                candidates[column_major_index(row, candidate, ndof)];
        }

        // Reorthogonalized modified Gram-Schmidt is substantially cheaper than
        // diagonalizing the candidate Gram matrix, whose dimension can exceed
        // the Hamiltonian dimension by the number of simplex vertices.
        for (size_t pass = 0; pass < 2; ++pass) {
            const auto current_dimension = union_vectors.size() / ndof;
            for (size_t column = 0; column < current_dimension; ++column) {
                auto overlap = Complex{0.0, 0.0};
                for (size_t row = 0; row < ndof; ++row) {
                    overlap += std::conj(union_vectors[column_major_index(
                        row,
                        column,
                        ndof
                    )]) * vector[row];
                }
                for (size_t row = 0; row < ndof; ++row) {
                    vector[row] -=
                        union_vectors[column_major_index(
                            row,
                            column,
                            ndof
                        )] * overlap;
                }
            }
        }
        auto squared_norm = 0.0;
        for (const auto value : vector) {
            squared_norm += std::norm(value);
        }
        const auto norm = std::sqrt(squared_norm);
        if (norm <= vector_threshold) {
            continue;
        }
        for (auto &value : vector) {
            union_vectors.push_back(value / norm);
        }
    }
    const auto union_dimension = union_vectors.size() / ndof;
    if (union_dimension < band_count) {
        throw std::runtime_error(
            "vertex subspace union lost part of the ambiguous band block"
        );
    }

    auto vertex_projectors = std::vector<std::vector<Complex>>{};
    if (union_dimension == ndof) {
        return VertexSubspaceInterpolator{
            .band_count = band_count,
            .union_dimension = union_dimension,
            .union_vectors = std::move(union_vectors),
            .vertex_projectors = {},
        };
    }
    vertex_projectors.reserve(vertex_blocks.size());
    for (const auto &block : vertex_blocks) {
        auto coordinates = std::vector<Complex>(
            union_dimension * band_count,
            Complex{0.0, 0.0}
        );
        linalg::matrix_multiply(
            'C',
            'N',
            union_dimension,
            band_count,
            ndof,
            Complex{1.0, 0.0},
            union_vectors.data(),
            ndof,
            block.data(),
            ndof,
            Complex{0.0, 0.0},
            coordinates.data(),
            union_dimension
        );
        auto projector = std::vector<Complex>(
            union_dimension * union_dimension,
            Complex{0.0, 0.0}
        );
        linalg::matrix_multiply(
            'N',
            'C',
            union_dimension,
            union_dimension,
            band_count,
            Complex{1.0, 0.0},
            coordinates.data(),
            union_dimension,
            coordinates.data(),
            union_dimension,
            Complex{0.0, 0.0},
            projector.data(),
            union_dimension
        );
        vertex_projectors.push_back(std::move(projector));
    }

    return VertexSubspaceInterpolator{
        .band_count = band_count,
        .union_dimension = union_dimension,
        .union_vectors = std::move(union_vectors),
        .vertex_projectors = std::move(vertex_projectors),
    };
}

struct InterpolatedProjector {
    std::vector<Complex> matrix;
    double projector_gap = std::numeric_limits<double>::infinity();
};

InterpolatedProjector interpolated_projector(
    const VertexSubspaceInterpolator &interpolator,
    std::span<const double> weights
) {
    if (interpolator.union_dimension == interpolator.band_count) {
        auto identity = std::vector<Complex>(
            interpolator.union_dimension * interpolator.union_dimension,
            Complex{0.0, 0.0}
        );
        for (size_t index = 0; index < interpolator.union_dimension; ++index) {
            identity[column_major_index(
                index,
                index,
                interpolator.union_dimension
            )] = Complex{1.0, 0.0};
        }
        return InterpolatedProjector{.matrix = std::move(identity)};
    }

    auto projector = std::vector<Complex>(
        interpolator.union_dimension * interpolator.union_dimension,
        Complex{0.0, 0.0}
    );
    for (size_t vertex = 0; vertex < weights.size(); ++vertex) {
        add_scaled(
            projector,
            interpolator.vertex_projectors[vertex],
            weights[vertex]
        );
    }
    auto projector_eigenvectors = projector;
    auto projector_eigenvalues = std::vector<double>{};
    linalg::diagonalize_hermitian_in_place(
        projector_eigenvectors,
        projector_eigenvalues,
        interpolator.union_dimension,
        true,
        "interpolated vertex projector"
    );

    const auto first_selected =
        interpolator.union_dimension - interpolator.band_count;
    const auto projector_gap =
        projector_eigenvalues[first_selected] -
        projector_eigenvalues[first_selected - 1];
    return InterpolatedProjector{
        .matrix = std::move(projector),
        .projector_gap = projector_gap,
    };
}

struct ProjectedEigenvalues {
    std::vector<double> values;
    std::vector<double> residual_norms;
};

ProjectedEigenvalues exact_eigenvalues(
    std::vector<Complex> hamiltonian,
    size_t ndof,
    size_t lower_band,
    size_t upper_band
) {
    auto eigenvalues = std::vector<double>{};
    linalg::diagonalize_hermitian_in_place(
        hamiltonian,
        eigenvalues,
        ndof,
        false,
        "exact sampled Hamiltonian"
    );
    return ProjectedEigenvalues{
        .values = std::vector<double>(
            eigenvalues.begin() + static_cast<std::ptrdiff_t>(lower_band),
            eigenvalues.begin() + static_cast<std::ptrdiff_t>(upper_band)
        ),
        .residual_norms =
            std::vector<double>(upper_band - lower_band, 0.0),
    };
}

ProjectedEigenvalues selected_projected_eigenvalues(
    std::span<const Complex> hamiltonian,
    std::vector<Complex> projected_hamiltonian,
    std::span<const Complex> target_projector,
    std::span<const Complex> union_vectors,
    size_t ndof,
    size_t union_dimension,
    size_t projection_lower_band,
    size_t projection_upper_band,
    size_t target_lower_band,
    size_t target_upper_band
) {
    auto eigenvalues = std::vector<double>{};
    linalg::diagonalize_hermitian_in_place(
        projected_hamiltonian,
        eigenvalues,
        union_dimension,
        true,
        "Hamiltonian in vertex subspace union"
    );
    const auto target_band_count = target_upper_band - target_lower_band;
    const auto projection_band_count =
        projection_upper_band - projection_lower_band;
    auto selected = std::vector<size_t>{};
    if (union_dimension == projection_band_count) {
        selected.resize(union_dimension);
        std::iota(selected.begin(), selected.end(), size_t{0});
    } else {
        auto projector_times_vectors = std::vector<Complex>(
            union_dimension * union_dimension,
            Complex{0.0, 0.0}
        );
        linalg::matrix_multiply(
            'N',
            'N',
            union_dimension,
            union_dimension,
            union_dimension,
            Complex{1.0, 0.0},
            target_projector.data(),
            union_dimension,
            projected_hamiltonian.data(),
            union_dimension,
            Complex{0.0, 0.0},
            projector_times_vectors.data(),
            union_dimension
        );

        auto scores = std::vector<double>(union_dimension, 0.0);
        for (size_t column = 0; column < union_dimension; ++column) {
            auto score = Complex{0.0, 0.0};
            for (size_t row = 0; row < union_dimension; ++row) {
                score += std::conj(projected_hamiltonian[column_major_index(
                    row,
                    column,
                    union_dimension
                )]) * projector_times_vectors[column_major_index(
                    row,
                    column,
                    union_dimension
                )];
            }
            scores[column] = std::real(score);
        }

        selected.resize(union_dimension);
        std::iota(selected.begin(), selected.end(), size_t{0});
        std::stable_sort(
            selected.begin(),
            selected.end(),
            [&scores](size_t left, size_t right) {
                return scores[left] > scores[right];
            }
        );
        selected.resize(projection_band_count);
    }
    std::sort(
        selected.begin(),
        selected.end(),
        [&eigenvalues](size_t left, size_t right) {
            return eigenvalues[left] < eigenvalues[right];
        }
    );

    const auto target_offset = target_lower_band - projection_lower_band;
    auto target_vectors = std::vector<Complex>(
        ndof * target_band_count,
        Complex{0.0, 0.0}
    );
    auto union_coordinates = std::vector<Complex>(
        union_dimension * target_band_count,
        Complex{0.0, 0.0}
    );
    auto result = ProjectedEigenvalues{};
    result.values.reserve(target_band_count);
    for (size_t band = 0; band < target_band_count; ++band) {
        const auto selected_index = selected[target_offset + band];
        result.values.push_back(eigenvalues[selected_index]);
        for (size_t row = 0; row < union_dimension; ++row) {
            union_coordinates[column_major_index(
                row,
                band,
                union_dimension
            )] = projected_hamiltonian[column_major_index(
                row,
                selected_index,
                union_dimension
            )];
        }
    }
    linalg::matrix_multiply(
        'N',
        'N',
        ndof,
        target_band_count,
        union_dimension,
        Complex{1.0, 0.0},
        union_vectors.data(),
        ndof,
        union_coordinates.data(),
        union_dimension,
        Complex{0.0, 0.0},
        target_vectors.data(),
        ndof
    );

    auto residuals = std::vector<Complex>(
        ndof * target_band_count,
        Complex{0.0, 0.0}
    );
    linalg::matrix_multiply(
        'N',
        'N',
        ndof,
        target_band_count,
        ndof,
        Complex{1.0, 0.0},
        hamiltonian.data(),
        ndof,
        target_vectors.data(),
        ndof,
        Complex{0.0, 0.0},
        residuals.data(),
        ndof
    );
    result.residual_norms.assign(target_band_count, 0.0);
    for (size_t band = 0; band < target_band_count; ++band) {
        auto squared_norm = 0.0;
        for (size_t row = 0; row < ndof; ++row) {
            residuals[column_major_index(row, band, ndof)] -=
                result.values[band] *
                target_vectors[column_major_index(row, band, ndof)];
            squared_norm += std::norm(
                residuals[column_major_index(row, band, ndof)]
            );
        }
        result.residual_norms[band] = std::sqrt(squared_norm);
    }
    return result;
}

std::vector<double> interpolated_band_energies(
    const core::Simplex &simplex,
    const EigensystemCache &cache,
    std::span<const double> weights,
    size_t lower_band,
    size_t upper_band
) {
    auto result = std::vector<double>(upper_band - lower_band, 0.0);
    for (size_t vertex = 0; vertex < simplex.vertex_ids.size(); ++vertex) {
        const auto &eigenvalues =
            cache.get(simplex.vertex_ids[vertex]).eigenvalues;
        for (size_t band = lower_band; band < upper_band; ++band) {
            result[band - lower_band] +=
                weights[vertex] * eigenvalues[band];
        }
    }
    return result;
}

std::pair<size_t, size_t> guarded_band_interval(
    const core::Simplex &simplex,
    const EigensystemCache &cache,
    size_t ndof,
    size_t lower_band,
    size_t upper_band,
    double tolerance
) {
    auto guarded_lower = lower_band == 0 ? size_t{0} : lower_band - 1;
    auto guarded_upper = std::min(ndof, upper_band + 1);
    const auto boundary_is_degenerate = [&](size_t boundary) {
        for (const auto vertex_id : simplex.vertex_ids) {
            const auto &values = cache.get(vertex_id).eigenvalues;
            const auto scale = std::max({
                1.0,
                std::abs(values[boundary - 1]),
                std::abs(values[boundary]),
            });
            if (
                values[boundary] - values[boundary - 1] <=
                tolerance * scale
            ) {
                return true;
            }
        }
        return false;
    };

    while (
        guarded_lower > 0 &&
        boundary_is_degenerate(guarded_lower)
    ) {
        --guarded_lower;
    }
    while (
        guarded_upper < ndof &&
        boundary_is_degenerate(guarded_upper)
    ) {
        ++guarded_upper;
    }
    return {guarded_lower, guarded_upper};
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
    const auto [projection_lower_band, projection_upper_band] =
        guarded_band_interval(
            simplex,
            cache,
            ndof,
            lower_band,
            upper_band,
            mesh.tolerance()
        );
    const auto interpolator = build_vertex_subspace_interpolator(
        simplex,
        cache,
        ndof,
        projection_lower_band,
        projection_upper_band,
        mesh.tolerance()
    );

    std::vector<std::vector<double>> samples;
    const auto nvertices = simplex.vertex_ids.size();
    samples.reserve(
        nvertices * (nvertices - 1) / 2 +
        (nvertices > 2 ? 1 : 0)
    );
    for (size_t left = 0; left < simplex.vertex_ids.size(); ++left) {
        for (size_t right = left + 1; right < simplex.vertex_ids.size(); ++right) {
            samples.push_back(
                barycentric_sample(nvertices, left, right, 0.5)
            );
        }
    }
    // In one dimension the barycenter is already the sole edge midpoint.
    if (nvertices > 2) {
        samples.push_back(center_sample(nvertices));
    }

    auto estimate = ProjectedResidualEstimate{};
    estimate.union_dimension = interpolator.union_dimension;
    for (const auto &weights : samples) {
        const auto point = sample_point(geometry, simplex, weights);
        auto hamiltonian = mesh.hamiltonian(
            std::span<const double>(point.data(), point.size())
        );
        auto projected_eigenvalues = ProjectedEigenvalues{};
        if (interpolator.union_dimension == ndof) {
            projected_eigenvalues = exact_eigenvalues(
                hamiltonian,
                ndof,
                lower_band,
                upper_band
            );
        } else {
            const auto target_projector =
                interpolated_projector(interpolator, weights);
            estimate.minimum_projector_gap = std::min(
                estimate.minimum_projector_gap,
                target_projector.projector_gap
            );
            const auto projected_hamiltonian =
                project_matrix(
                    hamiltonian,
                    interpolator.union_vectors,
                    ndof,
                    interpolator.union_dimension
                );
            projected_eigenvalues = selected_projected_eigenvalues(
                hamiltonian,
                projected_hamiltonian,
                target_projector.matrix,
                interpolator.union_vectors,
                ndof,
                interpolator.union_dimension,
                projection_lower_band,
                projection_upper_band,
                lower_band,
                upper_band
            );
        }
        const auto interpolated_eigenvalues = interpolated_band_energies(
            simplex,
            cache,
            weights,
            lower_band,
            upper_band
        );
        for (size_t band = 0; band < jdim; ++band) {
            const auto difference =
                projected_eigenvalues.values[band] -
                interpolated_eigenvalues[band];
            if (interpolator.union_dimension == ndof) {
                estimate.positive_estimate =
                    std::max(estimate.positive_estimate, difference);
                estimate.negative_estimate =
                    std::max(estimate.negative_estimate, -difference);
            } else {
                // The guard block makes band association much more stable, but
                // a reduced subspace does not prove the global ordered-band
                // index. Preserve both signs until a full-space solve is used.
                const auto uncertainty =
                    std::abs(difference) +
                    projected_eigenvalues.residual_norms[band];
                estimate.positive_estimate =
                    std::max(estimate.positive_estimate, uncertainty);
                estimate.negative_estimate =
                    std::max(estimate.negative_estimate, uncertainty);
            }
        }
    }
    return estimate;
}

}  // namespace fermisimplex
