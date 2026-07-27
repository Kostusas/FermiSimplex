#include "integration/charge.h"
#include "integration/projected_error.h"
#include "linalg/blas_lapack.h"
#include "test_helpers.h"

#include <fermisimplex/integration.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <exception>
#include <iostream>
#include <memory>
#include <random>
#include <span>
#include <utility>
#include <vector>

namespace {

using namespace fermisimplex;
using namespace fermisimplex::test;
namespace core = adaptivesimplex::core;

using Matrix = std::vector<Complex>;

std::size_t cm(std::size_t row, std::size_t column, std::size_t rows) {
    return row + column * rows;
}

class CountingQuadraticModel final : public HamiltonianModel {
public:
    CountingQuadraticModel(std::size_t ndim, double sign)
        : ndim_(ndim), sign_(sign) {}

    std::size_t ndim() const noexcept override { return ndim_; }
    std::size_t ndof() const noexcept override { return 1; }

    Matrix evaluate(std::span<const double> point) const override {
        ++evaluation_count_;
        auto value = 0.0;
        for (const auto coordinate : point) {
            value += coordinate * coordinate;
        }
        return {Complex{sign_ * value, 0.0}};
    }

    std::size_t evaluation_count() const noexcept {
        return evaluation_count_;
    }

private:
    std::size_t ndim_ = 0;
    double sign_ = 1.0;
    mutable std::size_t evaluation_count_ = 0;
};

class AffineHermitianModel final : public HamiltonianModel {
public:
    AffineHermitianModel(
        std::size_t ndim,
        std::size_t ndof,
        std::uint64_t seed
    ) : ndim_(ndim), ndof_(ndof) {
        auto generator = std::mt19937_64{seed};
        constant_ = random_hermitian(generator, 0.8);
        directions_.reserve(ndim_);
        for (std::size_t axis = 0; axis < ndim_; ++axis) {
            directions_.push_back(random_hermitian(generator, 0.45));
        }
    }

    std::size_t ndim() const noexcept override { return ndim_; }
    std::size_t ndof() const noexcept override { return ndof_; }

    Matrix evaluate(std::span<const double> point) const override {
        auto result = constant_;
        for (std::size_t axis = 0; axis < ndim_; ++axis) {
            for (std::size_t index = 0; index < result.size(); ++index) {
                result[index] += point[axis] * directions_[axis][index];
            }
        }
        return result;
    }

private:
    Matrix random_hermitian(
        std::mt19937_64 &generator,
        double scale
    ) const {
        auto distribution = std::normal_distribution<double>{0.0, scale};
        auto result = Matrix(ndof_ * ndof_, Complex{0.0, 0.0});
        for (std::size_t column = 0; column < ndof_; ++column) {
            result[cm(column, column, ndof_)] =
                Complex{distribution(generator), 0.0};
            for (std::size_t row = column + 1; row < ndof_; ++row) {
                const auto value = Complex{
                    distribution(generator),
                    distribution(generator),
                };
                result[cm(row, column, ndof_)] = value;
                result[cm(column, row, ndof_)] = std::conj(value);
            }
        }
        return result;
    }

    std::size_t ndim_ = 0;
    std::size_t ndof_ = 0;
    Matrix constant_;
    std::vector<Matrix> directions_;
};

void fill_simplex(SpectralMesh &mesh, core::SimplexId simplex_id) {
    fill_vertex_cache(
        mesh.geometry(),
        simplex_id,
        mesh,
        mesh.eigensystems()
    );
}

Matrix multiply(
    char left_operation,
    char right_operation,
    std::size_t rows,
    std::size_t columns,
    std::size_t inner,
    const Matrix &left,
    std::size_t left_rows,
    const Matrix &right,
    std::size_t right_rows
) {
    auto result = Matrix(rows * columns, Complex{0.0, 0.0});
    linalg::matrix_multiply(
        left_operation,
        right_operation,
        rows,
        columns,
        inner,
        Complex{1.0, 0.0},
        left.data(),
        left_rows,
        right.data(),
        right_rows,
        Complex{0.0, 0.0},
        result.data(),
        rows
    );
    return result;
}

Matrix selected_vectors(
    const Eigensystem &entry,
    std::size_t lower,
    std::size_t upper
) {
    const auto ndof = entry.eigenvalues.size();
    const auto count = upper - lower;
    auto result = Matrix(ndof * count, Complex{0.0, 0.0});
    for (std::size_t column = 0; column < count; ++column) {
        for (std::size_t row = 0; row < ndof; ++row) {
            result[cm(row, column, ndof)] =
                entry.eigenvectors[cm(row, lower + column, ndof)];
        }
    }
    return result;
}

Matrix reference_projector_subspace(
    std::span<const Matrix> blocks,
    std::span<const double> weights,
    std::size_t ndof,
    std::size_t band_count
) {
    auto projector = Matrix(ndof * ndof, Complex{0.0, 0.0});
    for (std::size_t vertex = 0; vertex < blocks.size(); ++vertex) {
        const auto contribution = multiply(
            'N',
            'C',
            ndof,
            ndof,
            band_count,
            blocks[vertex],
            ndof,
            blocks[vertex],
            ndof
        );
        for (std::size_t index = 0; index < projector.size(); ++index) {
            projector[index] += weights[vertex] * contribution[index];
        }
    }
    auto eigenvalues = std::vector<double>{};
    linalg::diagonalize_hermitian_in_place(
        projector,
        eigenvalues,
        ndof,
        true,
        "projected-error test reference projector"
    );
    auto result = Matrix(ndof * band_count, Complex{0.0, 0.0});
    for (std::size_t column = 0; column < band_count; ++column) {
        const auto source = ndof - band_count + column;
        for (std::size_t row = 0; row < ndof; ++row) {
            result[cm(row, column, ndof)] =
                projector[cm(row, source, ndof)];
        }
    }
    return result;
}

std::vector<double> reference_projected_values(
    const Matrix &hamiltonian,
    const Matrix &vectors,
    std::size_t ndof,
    std::size_t band_count
) {
    const auto h_times_v = multiply(
        'N', 'N', ndof, band_count, ndof,
        hamiltonian, ndof, vectors, ndof
    );
    auto projected = multiply(
        'C', 'N', band_count, band_count, ndof,
        vectors, ndof, h_times_v, ndof
    );
    auto values = std::vector<double>{};
    linalg::diagonalize_hermitian_in_place(
        projected,
        values,
        band_count,
        false,
        "projected-error test reference values"
    );
    return values;
}

std::vector<double> sample_point(
    const core::Geometry &geometry,
    const core::Simplex &simplex,
    std::span<const double> weights
) {
    auto result = std::vector<double>(geometry.ndim(), 0.0);
    for (std::size_t vertex = 0; vertex < weights.size(); ++vertex) {
        const auto point = geometry.vertices().dyadic_vertex(
            simplex.vertex_ids[vertex]
        ).to_point();
        for (std::size_t axis = 0; axis < result.size(); ++axis) {
            result[axis] += weights[vertex] * point[axis];
        }
    }
    return result;
}

void add_reference_probe(
    ProjectedErrorEstimate &estimate,
    const SpectralMesh &mesh,
    const core::Simplex &simplex,
    std::span<const Matrix> blocks,
    std::span<const double> weights,
    std::size_t lower,
    std::size_t upper
) {
    const auto ndof = mesh.ndof();
    const auto count = upper - lower;
    const auto vectors = reference_projector_subspace(
        blocks, weights, ndof, count
    );
    const auto point = sample_point(mesh.geometry(), simplex, weights);
    const auto values = reference_projected_values(
        mesh.hamiltonian(point), vectors, ndof, count
    );
    for (std::size_t offset = 0; offset < count; ++offset) {
        auto linear = 0.0;
        for (std::size_t vertex = 0; vertex < weights.size(); ++vertex) {
            linear += weights[vertex] * mesh.eigensystems().get(
                simplex.vertex_ids[vertex]
            ).eigenvalues[lower + offset];
        }
        const auto difference = values[offset] - linear;
        estimate.positive_estimate =
            std::max(estimate.positive_estimate, difference);
        estimate.negative_estimate =
            std::max(estimate.negative_estimate, -difference);
    }
}

void add_reference_exact_probe(
    ProjectedErrorEstimate &estimate,
    const SpectralMesh &mesh,
    const core::Simplex &simplex,
    std::span<const double> weights,
    std::size_t lower,
    std::size_t upper
) {
    const auto point = sample_point(mesh.geometry(), simplex, weights);
    auto hamiltonian = mesh.hamiltonian(point);
    auto values = std::vector<double>{};
    linalg::diagonalize_hermitian_in_place(
        hamiltonian,
        values,
        mesh.ndof(),
        false,
        "projected-error test exact center"
    );
    for (std::size_t offset = 0; offset < upper - lower; ++offset) {
        auto linear = 0.0;
        for (std::size_t vertex = 0; vertex < weights.size(); ++vertex) {
            linear += weights[vertex] * mesh.eigensystems().get(
                simplex.vertex_ids[vertex]
            ).eigenvalues[lower + offset];
        }
        const auto difference = values[lower + offset] - linear;
        estimate.positive_estimate =
            std::max(estimate.positive_estimate, difference);
        estimate.negative_estimate =
            std::max(estimate.negative_estimate, -difference);
    }
}

ProjectedErrorEstimate reference_estimate(
    const SpectralMesh &mesh,
    core::SimplexId simplex_id,
    std::size_t lower,
    std::size_t upper
) {
    const auto &simplex = mesh.geometry().simplices().simplex(simplex_id);
    const auto vertex_count = simplex.vertex_ids.size();
    auto blocks = std::vector<Matrix>{};
    blocks.reserve(vertex_count);
    for (const auto vertex_id : simplex.vertex_ids) {
        blocks.push_back(selected_vectors(
            mesh.eigensystems().get(vertex_id), lower, upper
        ));
    }

    auto result = ProjectedErrorEstimate{};
    if (vertex_count > 2) {
        const auto weights = std::vector<double>(
            vertex_count,
            1.0 / static_cast<double>(vertex_count)
        );
        add_reference_exact_probe(
            result, mesh, simplex, weights, lower, upper
        );
    }
    for (std::size_t first = 0; first < vertex_count; ++first) {
        for (std::size_t second = first + 1;
             second < vertex_count;
             ++second) {
            auto weights = std::vector<double>(vertex_count, 0.0);
            weights[first] = 0.5;
            weights[second] = 0.5;
            add_reference_probe(
                result, mesh, simplex, blocks, weights, lower, upper
            );
        }
    }
    return result;
}

void test_empty_and_invalid_intervals() {
    const auto model = std::make_shared<AffineHermitianModel>(3, 8, 11);
    auto mesh = SpectralMesh(model, kTol, 0);
    const auto simplex_id = first_active_simplex(mesh.geometry());
    fill_simplex(mesh, simplex_id);

    const auto empty = estimate_projected_error(mesh, simplex_id, 3, 3);
    expect_near(empty.negative_estimate, 0.0, 0.0, "empty negative estimate");
    expect_near(empty.positive_estimate, 0.0, 0.0, "empty positive estimate");
    expect_runtime_error(
        [&] { estimate_projected_error(mesh, simplex_id, 5, 4); },
        "invalid selected band interval",
        "reversed selected interval"
    );
    expect_runtime_error(
        [&] { estimate_projected_error(mesh, simplex_id, 0, 9); },
        "invalid selected band interval",
        "out-of-range selected interval"
    );
}

void test_scalar_quadratic_signs() {
    auto convex_model = std::make_shared<CountingQuadraticModel>(1, 1.0);
    auto convex = SpectralMesh(convex_model, kTol, 0);
    const auto convex_simplex = first_active_simplex(convex.geometry());
    fill_simplex(convex, convex_simplex);
    const auto convex_error = estimate_projected_error(
        convex, convex_simplex, 0, 1
    );
    expect_near(convex_error.positive_estimate, 0.0, kTol, "convex positive error");
    expect_near(convex_error.negative_estimate, 0.25, kTol, "convex negative error");

    auto concave_model = std::make_shared<CountingQuadraticModel>(1, -1.0);
    auto concave = SpectralMesh(concave_model, kTol, 0);
    const auto concave_simplex = first_active_simplex(concave.geometry());
    fill_simplex(concave, concave_simplex);
    const auto concave_error = estimate_projected_error(
        concave, concave_simplex, 0, 1
    );
    expect_near(concave_error.negative_estimate, 0.0, kTol, "concave negative error");
    expect_near(concave_error.positive_estimate, 0.25, kTol, "concave positive error");
}

void test_distinct_probe_counts() {
    for (const auto [ndim, expected] : std::array{
             std::pair{std::size_t{1}, std::size_t{1}},
             std::pair{std::size_t{2}, std::size_t{4}},
             std::pair{std::size_t{3}, std::size_t{7}},
         }) {
        const auto model =
            std::make_shared<CountingQuadraticModel>(ndim, 1.0);
        auto mesh = SpectralMesh(model, kTol, 0);
        const auto simplex_id = first_active_simplex(mesh.geometry());
        fill_simplex(mesh, simplex_id);
        const auto before = model->evaluation_count();
        static_cast<void>(estimate_projected_error(
            mesh, simplex_id, 0, 1
        ));
        expect_eq(
            model->evaluation_count() - before,
            expected,
            "distinct projected-error probe count"
        );
    }
}

void rotate_selected_block(
    SpectralMesh &mesh,
    core::SimplexId simplex_id,
    std::size_t lower
) {
    const auto &simplex = mesh.geometry().simplices().simplex(simplex_id);
    for (std::size_t vertex = 0; vertex < simplex.vertex_ids.size(); ++vertex) {
        const auto vertex_id = simplex.vertex_ids[vertex];
        auto entry = mesh.eigensystems().get(vertex_id);
        const auto angle = 0.31 + 0.27 * static_cast<double>(vertex);
        const auto cosine = std::cos(angle);
        const auto sine = std::sin(angle);
        for (std::size_t row = 0; row < mesh.ndof(); ++row) {
            const auto first =
                std::polar(1.0, 0.19 + 0.13 * static_cast<double>(vertex)) *
                entry.eigenvectors[cm(row, lower, mesh.ndof())];
            const auto second =
                std::polar(1.0, -0.23 - 0.17 * static_cast<double>(vertex)) *
                entry.eigenvectors[
                    cm(row, lower + 1, mesh.ndof())
                ];
            entry.eigenvectors[cm(row, lower, mesh.ndof())] =
                cosine * first + sine * second;
            entry.eigenvectors[cm(row, lower + 1, mesh.ndof())] =
                -sine * first + cosine * second;
        }
        mesh.eigensystems().insert(vertex_id, std::move(entry));
    }
}

void test_multiband_gauge_invariance() {
    auto mesh = SpectralMesh(
        std::make_shared<AffineHermitianModel>(3, 8, 37),
        kTol,
        0
    );
    const auto simplex_id = first_active_simplex(mesh.geometry());
    fill_simplex(mesh, simplex_id);
    const auto original = estimate_projected_error(mesh, simplex_id, 3, 5);
    rotate_selected_block(mesh, simplex_id, 3);
    const auto rotated = estimate_projected_error(mesh, simplex_id, 3, 5);
    expect_near(
        rotated.negative_estimate,
        original.negative_estimate,
        1e-10,
        "multiband negative gauge invariance"
    );
    expect_near(
        rotated.positive_estimate,
        original.positive_estimate,
        1e-10,
        "multiband positive gauge invariance"
    );
}

void test_randomized_reference_equivalence() {
    constexpr auto widths = std::array<std::size_t, 5>{1, 2, 3, 4, 8};
    for (std::uint64_t seed = 0; seed < 12; ++seed) {
        auto mesh = SpectralMesh(
            std::make_shared<AffineHermitianModel>(3, 8, 1000 + seed),
            kTol,
            0
        );
        const auto simplex_id = first_active_simplex(mesh.geometry());
        fill_simplex(mesh, simplex_id);
        for (const auto width : widths) {
            const auto lower = (mesh.ndof() - width) / 2;
            const auto upper = lower + width;
            const auto actual = estimate_projected_error(
                mesh, simplex_id, lower, upper
            );
            const auto reference = reference_estimate(
                mesh, simplex_id, lower, upper
            );
            expect_near(
                actual.negative_estimate,
                reference.negative_estimate,
                1e-9,
                "randomized negative exact-center/projected-edge equivalence"
            );
            expect_near(
                actual.positive_estimate,
                reference.positive_estimate,
                1e-9,
                "randomized positive exact-center/projected-edge equivalence"
            );
        }
    }
}

void test_shared_edge_cache_equivalence() {
    auto mesh = SpectralMesh(
        std::make_shared<AffineHermitianModel>(2, 8, 73),
        kTol,
        1
    );
    const auto active = mesh.geometry().simplices().active_simplices();
    const auto simplex_ids = std::vector<core::SimplexId>(
        active.begin(), active.end()
    );
    for (const auto simplex_id : simplex_ids) {
        fill_simplex(mesh, simplex_id);
    }

    for (const auto width : std::array<std::size_t, 2>{2, 8}) {
        const auto lower = (mesh.ndof() - width) / 2;
        const auto upper = lower + width;
        auto cache = ProjectedErrorCache{};
        auto edge_incidences = std::size_t{0};
        for (const auto simplex_id : simplex_ids) {
            const auto uncached = estimate_projected_error(
                mesh, simplex_id, lower, upper
            );
            const auto cached = estimate_projected_error(
                mesh, simplex_id, lower, upper, &cache
            );
            expect_near(
                cached.negative_estimate,
                uncached.negative_estimate,
                1e-12,
                "shared-edge cache negative equivalence"
            );
            expect_near(
                cached.positive_estimate,
                uncached.positive_estimate,
                1e-12,
                "shared-edge cache positive equivalence"
            );
            const auto vertex_count = mesh.geometry().simplices()
                .simplex(simplex_id).vertex_ids.size();
            edge_incidences += vertex_count * (vertex_count - 1) / 2;
        }
        expect(
            cache.edge_estimates.size() < edge_incidences,
            "shared-edge cache should deduplicate neighboring simplices"
        );
    }
}

void test_charge_uses_projected_samples() {
    const auto model = std::make_shared<CountingQuadraticModel>(1, 1.0);
    auto mesh = SpectralMesh(model, kTol, 0);
    const auto simplex_id = first_active_simplex(mesh.geometry());
    fill_simplex(mesh, simplex_id);
    const auto before = model->evaluation_count();
    const auto contribution = integration_detail::charge_on_simplex(
        0.25,
        mesh,
        mesh.geometry(),
        simplex_id,
        0.0
    );
    expect(
        contribution.projected_error > 0.0,
        "charge should include the projected sample estimate"
    );
    expect(
        model->evaluation_count() > before,
        "charge should evaluate projected-error probes"
    );
}

}  // namespace

int main() {
    try {
        test_empty_and_invalid_intervals();
        test_scalar_quadratic_signs();
        test_distinct_probe_counts();
        test_multiband_gauge_invariance();
        test_randomized_reference_equivalence();
        test_shared_edge_cache_equivalence();
        test_charge_uses_projected_samples();
    } catch (const std::exception &error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
    return 0;
}
