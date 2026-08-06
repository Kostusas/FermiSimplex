#include <fermisimplex/hamiltonian.h>
#include <fermisimplex/fermi_surface.h>
#include <fermisimplex/integration.h>
#include <fermisimplex/spectral_mesh.h>

#include "fermi_surface/simplex_classification.h"
#include "linalg/blas_lapack.h"

#include <adaptivesimplex/core/types.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numbers>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#ifndef FERMISIMPLEX_LAPACK_ZHEEVD
#define FERMISIMPLEX_LAPACK_ZHEEVD zheevd_
#endif

#ifndef FERMISIMPLEX_BENCHMARK_GIT_COMMIT
#define FERMISIMPLEX_BENCHMARK_GIT_COMMIT "unknown"
#endif

#ifndef FERMISIMPLEX_BENCHMARK_COMPILER
#define FERMISIMPLEX_BENCHMARK_COMPILER "unknown"
#endif

#ifndef FERMISIMPLEX_BENCHMARK_GIT_DIRTY
#define FERMISIMPLEX_BENCHMARK_GIT_DIRTY "unknown"
#endif

#ifndef FERMISIMPLEX_BENCHMARK_BUILD_TYPE
#define FERMISIMPLEX_BENCHMARK_BUILD_TYPE "unknown"
#endif

#ifndef FERMISIMPLEX_BENCHMARK_SYSTEM
#define FERMISIMPLEX_BENCHMARK_SYSTEM "unknown"
#endif

#ifndef FERMISIMPLEX_BENCHMARK_LAPACK
#define FERMISIMPLEX_BENCHMARK_LAPACK "unknown"
#endif

extern "C" {
void FERMISIMPLEX_LAPACK_ZHEEVD(
    const char *jobz,
    const char *uplo,
    const int *n,
    std::complex<double> *a,
    const int *lda,
    double *w,
    std::complex<double> *work,
    const int *lwork,
    double *rwork,
    const int *lrwork,
    int *iwork,
    const int *liwork,
    int *info
);
}

namespace {

using Clock = std::chrono::steady_clock;
using Complex = std::complex<double>;
using Matrix = std::vector<Complex>;
namespace core = adaptivesimplex::core;
namespace fsdetail = fermisimplex::fermi_surface_detail;

volatile double benchmark_sink = 0.0;

struct Config {
    std::string preset = "ci";
    std::string only;
    std::size_t samples = 7;
    std::vector<std::size_t> matrix_sizes{2, 4, 8, 16, 32, 60};
    std::vector<std::size_t> tight_binding_terms{1, 5, 17};
    std::filesystem::path output;
};

struct Summary {
    double median_ns_per_operation = 0.0;
    double minimum_ns_per_operation = 0.0;
    double maximum_ns_per_operation = 0.0;
    double median_absolute_deviation_ns = 0.0;
};

struct Result {
    std::string name;
    std::string category;
    std::string unit;
    std::size_t ndim = 0;
    std::size_t ndof = 0;
    std::size_t hopping_terms = 0;
    std::size_t target_bands = 0;
    std::uint32_t root_level = 0;
    std::size_t vertices = 0;
    std::size_t simplices = 0;
    std::size_t simplex_visits = 0;
    std::size_t refinements = 0;
    double target_error = 0.0;
    double min_feature_size = 0.0;
    double curvature_bound = 0.0;
    std::uint32_t preview_depth = 0;
    std::uint32_t error_depth = 0;
    fermisimplex::ChargeErrorStats charge_error_stats;
    double stopping_error = 0.0;
    std::size_t visible_gapless_simplices = 0;
    std::size_t inconclusive_simplices = 0;
    bool target_reached = false;
    std::size_t operations_per_sample = 0;
    std::size_t samples = 0;
    Summary timing;
    double lapack_equivalents_per_operation = 0.0;
    double median_ns_per_vertex = 0.0;
    double median_ns_per_simplex_visit = 0.0;
    double total_lapack_equivalents_per_vertex = 0.0;
    double total_lapack_equivalents_per_simplex_visit = 0.0;
};

double elapsed_ns(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::nano>(end - start).count();
}

double median(std::vector<double> values) {
    if (values.empty()) {
        throw std::runtime_error("cannot summarize an empty benchmark");
    }
    std::sort(values.begin(), values.end());
    const auto middle = values.size() / 2;
    if (values.size() % 2 == 0) {
        return 0.5 * (values[middle - 1] + values[middle]);
    }
    return values[middle];
}

Summary summarize(std::vector<double> ns_per_operation) {
    const auto center = median(ns_per_operation);
    auto deviations = ns_per_operation;
    for (auto &value : deviations) {
        value = std::abs(value - center);
    }
    return Summary{
        .median_ns_per_operation = center,
        .minimum_ns_per_operation = *std::min_element(
            ns_per_operation.begin(), ns_per_operation.end()
        ),
        .maximum_ns_per_operation = *std::max_element(
            ns_per_operation.begin(), ns_per_operation.end()
        ),
        .median_absolute_deviation_ns = median(std::move(deviations)),
    };
}

template <class Sample>
Summary measure(
    std::size_t samples,
    std::size_t operations_per_sample,
    Sample &&sample
) {
    if (samples == 0 || operations_per_sample == 0) {
        throw std::runtime_error("benchmark samples and operations must be positive");
    }
    auto timings = std::vector<double>{};
    timings.reserve(samples);
    for (std::size_t index = 0; index < samples; ++index) {
        const auto total_ns = sample(index);
        timings.push_back(total_ns / static_cast<double>(operations_per_sample));
    }
    return summarize(std::move(timings));
}

Result make_result(
    std::string name,
    std::string category,
    std::string unit,
    std::size_t ndim,
    std::size_t ndof,
    std::size_t hopping_terms,
    std::uint32_t root_level,
    std::size_t vertices,
    std::size_t simplices,
    std::size_t operations_per_sample,
    std::size_t samples,
    Summary timing,
    double lapack_ns
) {
    return Result{
        .name = std::move(name),
        .category = std::move(category),
        .unit = std::move(unit),
        .ndim = ndim,
        .ndof = ndof,
        .hopping_terms = hopping_terms,
        .root_level = root_level,
        .vertices = vertices,
        .simplices = simplices,
        .operations_per_sample = operations_per_sample,
        .samples = samples,
        .timing = timing,
        .lapack_equivalents_per_operation =
            lapack_ns > 0.0 ? timing.median_ns_per_operation / lapack_ns : 0.0,
    };
}

Result make_total_result(
    std::string name,
    std::size_t ndim,
    std::size_t ndof,
    std::uint32_t root_level,
    std::size_t vertices,
    std::size_t simplices,
    std::size_t simplex_visits,
    std::size_t refinements,
    std::size_t samples,
    Summary timing,
    double lapack_ns
) {
    auto result = make_result(
        std::move(name), "end_to_end", "run", ndim, ndof, 0, root_level,
        vertices, simplices, 1, samples, timing, lapack_ns
    );
    result.simplex_visits = simplex_visits;
    result.refinements = refinements;
    if (vertices != 0) {
        result.median_ns_per_vertex =
            timing.median_ns_per_operation / static_cast<double>(vertices);
        result.total_lapack_equivalents_per_vertex =
            result.median_ns_per_vertex / lapack_ns;
    }
    if (simplex_visits != 0) {
        result.median_ns_per_simplex_visit =
            timing.median_ns_per_operation / static_cast<double>(simplex_visits);
        result.total_lapack_equivalents_per_simplex_visit =
            result.median_ns_per_simplex_visit / lapack_ns;
    }
    return result;
}

std::size_t column_major_index(
    std::size_t row,
    std::size_t column,
    std::size_t size
) {
    return column * size + row;
}

Matrix dense_hermitian(std::size_t size) {
    auto matrix = Matrix(size * size, Complex{0.0, 0.0});
    const auto center = 0.5 * static_cast<double>(size - 1);
    const auto scale = 0.035 / std::sqrt(static_cast<double>(size));
    for (std::size_t column = 0; column < size; ++column) {
        matrix[column_major_index(column, column, size)] =
            Complex{0.4 * (static_cast<double>(column) - center), 0.0};
        for (std::size_t row = column + 1; row < size; ++row) {
            const auto phase = static_cast<double>((row + 1) * (column + 3));
            const auto value = scale * Complex{
                std::sin(phase),
                std::cos(0.7 * phase),
            };
            matrix[column_major_index(row, column, size)] = value;
            matrix[column_major_index(column, row, size)] = std::conj(value);
        }
    }
    return matrix;
}

Matrix adjoint(const Matrix &matrix, std::size_t size) {
    auto result = Matrix(matrix.size());
    for (std::size_t column = 0; column < size; ++column) {
        for (std::size_t row = 0; row < size; ++row) {
            result[column_major_index(row, column, size)] = std::conj(
                matrix[column_major_index(column, row, size)]
            );
        }
    }
    return result;
}

class FixedDenseModel final : public fermisimplex::HamiltonianModel {
public:
    FixedDenseModel(std::size_t ndim, Matrix matrix)
        : ndim_(ndim),
          ndof_(static_cast<std::size_t>(std::sqrt(matrix.size()))),
          matrix_(std::move(matrix)) {}

    std::size_t ndim() const noexcept override { return ndim_; }
    std::size_t ndof() const noexcept override { return ndof_; }

    Matrix evaluate(std::span<const double>) const override {
        return matrix_;
    }

private:
    std::size_t ndim_ = 0;
    std::size_t ndof_ = 0;
    Matrix matrix_;
};

class CrossingDenseModel final : public fermisimplex::HamiltonianModel {
public:
    explicit CrossingDenseModel(std::size_t ndof)
        : ndof_(ndof), matrix_(ndof * ndof, Complex{0.0, 0.0}) {
        for (std::size_t band = 1; band < ndof_; ++band) {
            matrix_[column_major_index(band, band, ndof_)] =
                Complex{3.0 + 0.1 * static_cast<double>(band), 0.0};
        }
        const auto scale = 0.01 / std::sqrt(static_cast<double>(ndof_));
        for (std::size_t column = 0; column < ndof_; ++column) {
            for (std::size_t row = column + 1; row < ndof_; ++row) {
                const auto phase = static_cast<double>((row + 2) * (column + 1));
                const auto value = scale * Complex{
                    std::sin(phase),
                    std::cos(0.6 * phase),
                };
                matrix_[column_major_index(row, column, ndof_)] = value;
                matrix_[column_major_index(column, row, ndof_)] = std::conj(value);
            }
        }
    }

    std::size_t ndim() const noexcept override { return 2; }
    std::size_t ndof() const noexcept override { return ndof_; }

    Matrix evaluate(std::span<const double> point) const override {
        auto result = matrix_;
        result[0] = Complex{
            std::cos(2.0 * std::numbers::pi_v<double> * point[0]) +
            std::cos(2.0 * std::numbers::pi_v<double> * point[1]) - 0.25,
            0.0,
        };
        return result;
    }

private:
    std::size_t ndof_ = 0;
    Matrix matrix_;
};

class ActiveClusterModel final : public fermisimplex::HamiltonianModel {
public:
    ActiveClusterModel(std::size_t ndof, std::size_t active)
        : ndof_(ndof), active_(active) {
        if (active_ == 0 || active_ > ndof_) {
            throw std::runtime_error("invalid active-cluster width");
        }
    }

    std::size_t ndim() const noexcept override { return 2; }
    std::size_t ndof() const noexcept override { return ndof_; }

    Matrix evaluate(std::span<const double> point) const override {
        auto result = Matrix(ndof_ * ndof_, Complex{0.0, 0.0});
        const auto occupied = (ndof_ - active_) / 2;
        const auto empty_begin = occupied + active_;
        for (std::size_t band = 0; band < occupied; ++band) {
            result[column_major_index(band, band, ndof_)] =
                Complex{-3.0 - 0.01 * static_cast<double>(band), 0.0};
        }
        const auto dispersion =
            std::cos(2.0 * std::numbers::pi_v<double> * point[0]) +
            std::cos(2.0 * std::numbers::pi_v<double> * point[1]) -
            0.25;
        const auto center = 0.5 * static_cast<double>(active_ - 1);
        for (std::size_t band = 0; band < active_; ++band) {
            result[column_major_index(
                occupied + band, occupied + band, ndof_
            )] = Complex{
                dispersion + 0.04 * (static_cast<double>(band) - center),
                0.0,
            };
        }
        for (std::size_t band = empty_begin; band < ndof_; ++band) {
            result[column_major_index(band, band, ndof_)] =
                Complex{3.0 + 0.01 * static_cast<double>(band), 0.0};
        }
        return result;
    }

private:
    std::size_t ndof_ = 0;
    std::size_t active_ = 0;
};

std::shared_ptr<const fermisimplex::HamiltonianModel> fixed_model(
    std::size_t ndim,
    std::size_t ndof
) {
    return std::make_shared<FixedDenseModel>(ndim, dense_hermitian(ndof));
}

std::shared_ptr<const fermisimplex::HamiltonianModel> crossing_model(
    std::size_t ndof
) {
    return std::make_shared<CrossingDenseModel>(ndof);
}

std::shared_ptr<const fermisimplex::HamiltonianModel> tight_binding_model(
    std::size_t ndof,
    std::size_t term_count
) {
    if (term_count == 0 || term_count % 2 == 0) {
        throw std::runtime_error("tight-binding term count must be positive and odd");
    }

    auto terms = std::vector<fermisimplex::HoppingTerm>{};
    terms.reserve(term_count);
    terms.push_back(fermisimplex::HoppingTerm{
        .lattice_vector = {0},
        .matrix = dense_hermitian(ndof),
    });

    const auto pairs = (term_count - 1) / 2;
    for (std::size_t pair = 1; pair <= pairs; ++pair) {
        auto hopping = Matrix(ndof * ndof);
        const auto scale = 0.025 /
            (static_cast<double>(pair) * std::sqrt(static_cast<double>(ndof)));
        for (std::size_t column = 0; column < ndof; ++column) {
            for (std::size_t row = 0; row < ndof; ++row) {
                const auto phase = static_cast<double>(
                    (pair + 1) * (row + 2) * (column + 1)
                );
                hopping[column_major_index(row, column, ndof)] = scale * Complex{
                    std::sin(phase),
                    std::cos(0.5 * phase),
                };
            }
        }
        terms.push_back(fermisimplex::HoppingTerm{
            .lattice_vector = {static_cast<std::int64_t>(pair)},
            .matrix = hopping,
        });
        terms.push_back(fermisimplex::HoppingTerm{
            .lattice_vector = {-static_cast<std::int64_t>(pair)},
            .matrix = adjoint(hopping, ndof),
        });
    }
    return std::make_shared<fermisimplex::TightBindingModel>(std::move(terms));
}

int lapack_int(std::size_t value) {
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("benchmark matrix is too large for LP64 LAPACK");
    }
    return static_cast<int>(value);
}

int workspace_size(double value) {
    if (!std::isfinite(value) || value > static_cast<double>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("invalid LAPACK workspace query result");
    }
    return std::max(1, static_cast<int>(value));
}

class ReusedWorkspaceZheevd {
public:
    explicit ReusedWorkspaceZheevd(std::size_t size)
        : size_(size),
          n_(lapack_int(size)),
          lda_(std::max(1, n_)),
          eigenvalues_(size) {
        auto matrix = Matrix(size * size);
        auto lwork = -1;
        auto lrwork = -1;
        auto liwork = -1;
        auto work_query = Complex{0.0, 0.0};
        auto rwork_query = 0.0;
        auto iwork_query = 0;
        auto info = 0;
        FERMISIMPLEX_LAPACK_ZHEEVD(
            &jobz_, &uplo_, &n_, matrix.data(), &lda_, eigenvalues_.data(),
            &work_query, &lwork, &rwork_query, &lrwork, &iwork_query, &liwork,
            &info
        );
        if (info != 0) {
            throw std::runtime_error("benchmark zheevd workspace query failed");
        }
        lwork_ = workspace_size(std::real(work_query));
        lrwork_ = workspace_size(rwork_query);
        liwork_ = workspace_size(static_cast<double>(iwork_query));
        work_.resize(static_cast<std::size_t>(lwork_));
        rwork_.resize(static_cast<std::size_t>(lrwork_));
        iwork_.resize(static_cast<std::size_t>(liwork_));
    }

    double solve(Matrix &matrix) {
        if (matrix.size() != size_ * size_) {
            throw std::runtime_error("benchmark zheevd matrix shape mismatch");
        }
        auto info = 0;
        FERMISIMPLEX_LAPACK_ZHEEVD(
            &jobz_, &uplo_, &n_, matrix.data(), &lda_, eigenvalues_.data(),
            work_.data(), &lwork_, rwork_.data(), &lrwork_, iwork_.data(),
            &liwork_, &info
        );
        if (info != 0) {
            throw std::runtime_error("benchmark zheevd failed");
        }
        return eigenvalues_[size_ / 2];
    }

private:
    std::size_t size_ = 0;
    char jobz_ = 'V';
    char uplo_ = 'L';
    int n_ = 0;
    int lda_ = 0;
    int lwork_ = 0;
    int lrwork_ = 0;
    int liwork_ = 0;
    std::vector<double> eigenvalues_;
    std::vector<Complex> work_;
    std::vector<double> rwork_;
    std::vector<int> iwork_;
};

std::size_t matrix_iterations(std::size_t ndof, const Config &config) {
    std::size_t iterations = 0;
    if (ndof <= 4) {
        iterations = 50000;
    } else if (ndof <= 8) {
        iterations = 10000;
    } else if (ndof <= 16) {
        iterations = 3000;
    } else if (ndof <= 32) {
        iterations = 600;
    } else if (ndof <= 64) {
        iterations = 150;
    } else {
        iterations = 30;
    }
    if (config.preset == "quick") {
        iterations = std::max<std::size_t>(10, iterations / 4);
    }
    return iterations;
}

std::size_t warmup_diagonalizations(const Config &config) {
    return config.preset == "quick" ? 1000 : 8000;
}

void warm_up(const Config &config) {
    constexpr auto size = std::size_t{32};
    const auto input = dense_hermitian(size);
    auto matrix = input;
    ReusedWorkspaceZheevd solver(size);
    auto checksum = 0.0;
    for (std::size_t index = 0; index < warmup_diagonalizations(config); ++index) {
        matrix = input;
        checksum += solver.solve(matrix);
    }
    benchmark_sink = checksum;
}

std::vector<core::SimplexId> active_simplices(
    const fermisimplex::SpectralMesh &mesh
) {
    const auto active = mesh.geometry().simplices().active_simplices();
    return {active.begin(), active.end()};
}

std::vector<core::VertexId> missing_active_vertices(
    fermisimplex::SpectralMesh &mesh,
    const std::vector<core::SimplexId> &simplices
) {
    return mesh.geometry().missing_vertices(
        std::span<const core::SimplexId>(simplices.data(), simplices.size()),
        mesh.eigensystems(),
        0
    );
}

double fill_vertices(
    fermisimplex::SpectralMesh &mesh,
    std::span<const core::VertexId> vertices
) {
    auto checksum = 0.0;
    for (const auto vertex_id : vertices) {
        const auto point = mesh.geometry().vertices().dyadic_vertex(vertex_id).to_point();
        auto spectrum = mesh.spectrum(
            std::span<const double>(point.data(), point.size())
        );
        checksum += spectrum.eigenvalues[spectrum.eigenvalues.size() / 2];
        mesh.eigensystems().insert(vertex_id, std::move(spectrum));
    }
    return checksum;
}

std::size_t lapack_batch_size(std::size_t ndof, std::size_t iterations) {
    constexpr auto target_bytes = std::size_t{256 * 1024};
    const auto matrix_bytes = std::max<std::size_t>(
        1,
        ndof * ndof * sizeof(Complex)
    );
    return std::max<std::size_t>(
        1,
        std::min(iterations, target_bytes / matrix_bytes)
    );
}

template <class Solve>
double time_prepared_lapack_calls(
    const Matrix &input,
    std::size_t ndof,
    std::size_t iterations,
    Solve &&solve
) {
    const auto batch_size = lapack_batch_size(ndof, iterations);
    auto completed = std::size_t{0};
    auto total_ns = 0.0;
    auto checksum = 0.0;
    while (completed < iterations) {
        const auto count = std::min(batch_size, iterations - completed);
        auto matrices = std::vector<Matrix>(count, input);
        const auto started = Clock::now();
        for (auto &matrix : matrices) {
            checksum += solve(matrix);
        }
        const auto finished = Clock::now();
        total_ns += elapsed_ns(started, finished);
        completed += count;
    }
    benchmark_sink = checksum;
    return total_ns;
}

double benchmark_reused_lapack(
    std::size_t ndof,
    const Config &config,
    std::vector<Result> &results
) {
    const auto iterations = matrix_iterations(ndof, config);
    const auto input = dense_hermitian(ndof);
    ReusedWorkspaceZheevd solver(ndof);
    auto warm_matrix = input;
    for (std::size_t warmup = 0; warmup < 8; ++warmup) {
        warm_matrix = input;
        benchmark_sink = solver.solve(warm_matrix);
    }
    const auto timing = measure(config.samples, iterations, [&](std::size_t) {
        return time_prepared_lapack_calls(
            input,
            ndof,
            iterations,
            [&solver](Matrix &matrix) { return solver.solve(matrix); }
        );
    });
    const auto baseline = timing.median_ns_per_operation;
    results.push_back(make_result(
        "lapack_reused_workspace", "vertex", "diagonalization", 0, ndof, 0,
        0, 0, 0, iterations, config.samples, timing, baseline
    ));
    return baseline;
}

double benchmark_vertex_pipeline(
    std::size_t ndof,
    const Config &config,
    double reused_lapack_ns,
    std::vector<Result> &results
) {
    const auto iterations = matrix_iterations(ndof, config);
    const auto evaluation_iterations = std::max<std::size_t>(
        iterations,
        config.preset == "quick" ? 250 : 2000
    );
    const auto input = dense_hermitian(ndof);

    auto warm_matrix = input;
    auto warm_eigenvalues = std::vector<double>{};
    for (std::size_t warmup = 0; warmup < 8; ++warmup) {
        warm_matrix = input;
        fermisimplex::linalg::diagonalize_hermitian_in_place(
            warm_matrix, warm_eigenvalues, ndof, true, "benchmark warmup"
        );
    }

    const auto wrapper = measure(config.samples, iterations, [&](std::size_t) {
        return time_prepared_lapack_calls(
            input,
            ndof,
            iterations,
            [ndof](Matrix &matrix) {
                auto eigenvalues = std::vector<double>{};
                fermisimplex::linalg::diagonalize_hermitian_in_place(
                    matrix, eigenvalues, ndof, true, "benchmark"
                );
                return eigenvalues[ndof / 2];
            }
        );
    });
    const auto lapack_ns = std::min(
        reused_lapack_ns,
        wrapper.median_ns_per_operation
    );
    auto &reused_result = results.back();
    reused_result.lapack_equivalents_per_operation =
        reused_lapack_ns / lapack_ns;
    const auto reference = reused_lapack_ns <= wrapper.median_ns_per_operation
        ? reused_result.timing
        : wrapper;
    results.push_back(make_result(
        "lapack_current_wrapper", "vertex", "diagonalization", 0, ndof, 0,
        0, 0, 0, iterations, config.samples, wrapper, lapack_ns
    ));
    results.push_back(make_result(
        "lapack_reference_best", "vertex", "diagonalization", 0, ndof, 0,
        0, 0, 0, iterations, config.samples, reference, lapack_ns
    ));

    const auto model = crossing_model(ndof);
    auto point = std::vector<double>{0.25, 0.75};
    const auto model_evaluation = measure(
        config.samples, evaluation_iterations, [&](std::size_t sample) {
            auto checksum = 0.0;
            point[0] = 0.1 * static_cast<double>(sample + 1);
            const auto started = Clock::now();
            for (std::size_t index = 0; index < evaluation_iterations; ++index) {
                const auto matrix = model->evaluate(point);
                checksum += std::real(matrix[index % matrix.size()]);
            }
            const auto finished = Clock::now();
            benchmark_sink = checksum;
            return elapsed_ns(started, finished);
        }
    );
    results.push_back(make_result(
        "dense_model_evaluate", "vertex", "evaluation", 2, ndof, 0,
        0, 0, 0, evaluation_iterations, config.samples, model_evaluation,
        lapack_ns
    ));

    auto mesh = fermisimplex::SpectralMesh(model);
    const auto validated = measure(
        config.samples, evaluation_iterations, [&](std::size_t sample) {
            auto checksum = 0.0;
            point[1] = 0.1 * static_cast<double>(sample + 1);
            const auto started = Clock::now();
            for (std::size_t index = 0; index < evaluation_iterations; ++index) {
                const auto matrix = mesh.hamiltonian(point);
                checksum += std::real(matrix[index % matrix.size()]);
            }
            const auto finished = Clock::now();
            benchmark_sink = checksum;
            return elapsed_ns(started, finished);
        }
    );
    results.push_back(make_result(
        "hamiltonian_validate", "vertex", "evaluation", 2, ndof, 0,
        0, 0, 0, evaluation_iterations, config.samples, validated, lapack_ns
    ));

    const auto spectrum = measure(
        config.samples, iterations, [&](std::size_t sample) {
            auto checksum = 0.0;
            point[0] = 0.07 * static_cast<double>(sample + 1);
            const auto started = Clock::now();
            for (std::size_t index = 0; index < iterations; ++index) {
                const auto eigensystem = mesh.spectrum(point);
                checksum += eigensystem.eigenvalues[ndof / 2];
            }
            const auto finished = Clock::now();
            benchmark_sink = checksum;
            return elapsed_ns(started, finished);
        }
    );
    results.push_back(make_result(
        "spectrum_dense_model", "vertex", "spectrum", 2, ndof, 0,
        0, 0, 0, iterations, config.samples, spectrum, lapack_ns
    ));

    const auto cached = measure(
        config.samples, iterations, [&](std::size_t sample) {
            mesh.eigensystems().clear();
            auto checksum = 0.0;
            point[1] = 0.09 * static_cast<double>(sample + 1);
            const auto started = Clock::now();
            for (std::size_t index = 0; index < iterations; ++index) {
                auto eigensystem = mesh.spectrum(point);
                checksum += eigensystem.eigenvalues[ndof / 2];
                mesh.eigensystems().insert(
                    static_cast<core::VertexId>(index),
                    std::move(eigensystem)
                );
            }
            const auto finished = Clock::now();
            benchmark_sink = checksum;
            return elapsed_ns(started, finished);
        }
    );
    results.push_back(make_result(
        "spectrum_and_cache_insert", "vertex", "vertex", 2, ndof, 0,
        0, iterations, 0, iterations, config.samples, cached, lapack_ns
    ));
    return lapack_ns;
}

void benchmark_tight_binding(
    std::size_t ndof,
    std::size_t hopping_terms,
    const Config &config,
    double lapack_ns,
    std::vector<Result> &results
) {
    const auto base_iterations = matrix_iterations(ndof, config);
    const auto spectrum_iterations = std::max<std::size_t>(
        8,
        base_iterations / std::max<std::size_t>(1, hopping_terms / 4)
    );
    const auto evaluation_iterations = std::max<std::size_t>(
        base_iterations,
        config.preset == "quick" ? 250 : 1000
    );
    const auto model = tight_binding_model(ndof, hopping_terms);
    auto point = std::vector<double>{0.271828};

    const auto evaluation = measure(
        config.samples, evaluation_iterations, [&](std::size_t sample) {
            auto checksum = 0.0;
            const auto started = Clock::now();
            for (std::size_t index = 0; index < evaluation_iterations; ++index) {
                point[0] = std::fmod(
                    0.137 * static_cast<double>(index + sample + 1),
                    1.0
                );
                const auto matrix = model->evaluate(point);
                checksum += std::real(matrix[index % matrix.size()]);
            }
            const auto finished = Clock::now();
            benchmark_sink = checksum;
            return elapsed_ns(started, finished);
        }
    );
    results.push_back(make_result(
        "tight_binding_evaluate", "vertex", "evaluation", 1, ndof,
        hopping_terms, 0, 0, 0, evaluation_iterations, config.samples,
        evaluation, lapack_ns
    ));

    auto mesh = fermisimplex::SpectralMesh(model);
    const auto spectrum = measure(
        config.samples, spectrum_iterations, [&](std::size_t sample) {
            auto checksum = 0.0;
            const auto started = Clock::now();
            for (std::size_t index = 0; index < spectrum_iterations; ++index) {
                point[0] = std::fmod(
                    0.173 * static_cast<double>(index + sample + 1),
                    1.0
                );
                const auto eigensystem = mesh.spectrum(point);
                checksum += eigensystem.eigenvalues[ndof / 2];
            }
            const auto finished = Clock::now();
            benchmark_sink = checksum;
            return elapsed_ns(started, finished);
        }
    );
    results.push_back(make_result(
        "spectrum_tight_binding", "vertex", "spectrum", 1, ndof,
        hopping_terms, 0, 0, 0, spectrum_iterations, config.samples,
        spectrum, lapack_ns
    ));
}

void require_stable_count(
    std::size_t &stored,
    std::size_t observed,
    std::string_view name
) {
    if (stored == 0) {
        stored = observed;
    } else if (stored != observed) {
        throw std::runtime_error(
            std::string(name) + " changed between benchmark samples"
        );
    }
}

void benchmark_charge_total(
    std::size_t ndof,
    const Config &config,
    double lapack_ns,
    std::vector<Result> &results
) {
    constexpr auto ndim = std::size_t{2};
    constexpr auto root_level = std::uint32_t{2};
    constexpr auto target_error = 0.02;
    auto vertices = std::size_t{0};
    auto simplices = std::size_t{0};
    auto simplex_visits = std::size_t{0};
    auto visible_gapless_simplices = std::size_t{0};
    auto inconclusive_simplices = std::size_t{0};
    auto stopping_error = 0.0;
    auto target_reached = false;
    auto charge_error_stats = fermisimplex::ChargeErrorStats{};
    const auto timing = measure(config.samples, 1, [&](std::size_t) {
        auto mesh = fermisimplex::SpectralMesh(
            crossing_model(ndof), 1e-14, root_level
        );
        const auto started = Clock::now();
        const auto charge = fermisimplex::estimate_charge_on_current_mesh(
            mesh,
            0.0,
            target_error,
            1
        );
        const auto finished = Clock::now();
        require_stable_count(
            vertices,
            static_cast<std::size_t>(charge.stats.evaluations),
            "charge vertex count"
        );
        require_stable_count(
            simplices,
            static_cast<std::size_t>(charge.stats.active_simplices),
            "charge active-simplex count"
        );
        require_stable_count(
            simplex_visits,
            static_cast<std::size_t>(charge.stats.simplex_visits),
            "charge simplex-visit count"
        );
        require_stable_count(
            visible_gapless_simplices,
            static_cast<std::size_t>(charge.visible_gapless_simplices),
            "charge visible-gapless count"
        );
        require_stable_count(
            inconclusive_simplices,
            static_cast<std::size_t>(charge.inconclusive_simplices),
            "charge inconclusive count"
        );
        stopping_error = charge.stopping_error;
        charge_error_stats = charge.error_stats;
        target_reached = charge.stats.target_reached;
        benchmark_sink = charge.value + charge.stopping_error;
        return elapsed_ns(started, finished);
    });
    auto result = make_total_result(
        "charge_current_mesh_total",
        ndim,
        ndof,
        root_level,
        vertices,
        simplices,
        simplex_visits,
        0,
        config.samples,
        timing,
        lapack_ns
    );
    result.target_error = target_error;
    result.preview_depth = 0;
    result.stopping_error = stopping_error;
    result.error_depth = 1;
    result.charge_error_stats = charge_error_stats;
    result.visible_gapless_simplices = visible_gapless_simplices;
    result.inconclusive_simplices = inconclusive_simplices;
    result.target_reached = target_reached;
    results.push_back(std::move(result));
}

void benchmark_charge_estimator_case(
    std::size_t ndof,
    std::size_t active,
    std::uint32_t error_depth,
    const Config &config,
    double lapack_ns,
    std::vector<Result> &results
) {
    constexpr auto ndim = std::size_t{2};
    constexpr auto root_level = std::uint32_t{2};
    constexpr auto target_error = 1.0e9;

    auto vertices = std::size_t{0};
    auto simplices = std::size_t{0};
    auto simplex_visits = std::size_t{0};
    auto stopping_error = 0.0;
    auto target_reached = false;
    auto error_stats = fermisimplex::ChargeErrorStats{};
    const auto timing = measure(config.samples, 1, [&](std::size_t) {
        auto mesh = fermisimplex::SpectralMesh(
            std::make_shared<ActiveClusterModel>(ndof, active),
            1e-14,
            root_level
        );
        const auto started = Clock::now();
        const auto charge = fermisimplex::estimate_charge_on_current_mesh(
            mesh,
            0.0,
            target_error,
            error_depth
        );
        const auto finished = Clock::now();

        vertices = static_cast<std::size_t>(charge.stats.evaluations);
        simplices =
            static_cast<std::size_t>(charge.stats.active_simplices);
        simplex_visits =
            static_cast<std::size_t>(charge.stats.simplex_visits);
        stopping_error = charge.stopping_error;
        target_reached = charge.stats.target_reached;
        error_stats = charge.error_stats;
        benchmark_sink = charge.value + charge.stopping_error;
        return elapsed_ns(started, finished);
    });

    auto result = make_total_result(
        "charge_estimator_n" + std::to_string(ndof) +
            "_q" + std::to_string(active) +
            "_d" + std::to_string(error_depth),
        ndim,
        ndof,
        root_level,
        vertices,
        simplices,
        simplex_visits,
        0,
        config.samples,
        timing,
        lapack_ns
    );
    result.category = "charge_estimator";
    result.target_bands = active;
    result.target_error = target_error;
    result.error_depth = error_depth;
    result.stopping_error = stopping_error;
    result.target_reached = target_reached;
    result.charge_error_stats = error_stats;
    results.push_back(std::move(result));
}

void benchmark_fermi_surface_total(
    std::size_t ndof,
    const Config &config,
    double lapack_ns,
    std::vector<Result> &results
) {
    constexpr auto ndim = std::size_t{2};
    constexpr auto root_level = std::uint32_t{1};
    constexpr auto min_feature_size = 0.125;
    constexpr auto curvature_bound =
        4.0 * std::numbers::pi_v<double> * std::numbers::pi_v<double>;
    auto vertices = std::size_t{0};
    auto simplices = std::size_t{0};
    auto simplex_visits = std::size_t{0};
    auto refinements = std::size_t{0};
    const auto timing = measure(config.samples, 1, [&](std::size_t) {
        auto mesh = fermisimplex::SpectralMesh(
            crossing_model(ndof), 1e-14, root_level
        );
        const auto started = Clock::now();
        const auto surface = fermisimplex::fermi_surface(
            mesh,
            0.0,
            min_feature_size,
            -1,
            curvature_bound
        );
        const auto finished = Clock::now();
        require_stable_count(
            vertices,
            static_cast<std::size_t>(surface.stats.evaluations),
            "Fermi-surface vertex count"
        );
        require_stable_count(
            simplices,
            static_cast<std::size_t>(mesh.active_simplices()),
            "Fermi-surface active-simplex count"
        );
        require_stable_count(
            simplex_visits,
            static_cast<std::size_t>(surface.stats.simplex_visits),
            "Fermi-surface simplex-visit count"
        );
        require_stable_count(
            refinements,
            static_cast<std::size_t>(surface.stats.refinements),
            "Fermi-surface refinement count"
        );
        benchmark_sink = static_cast<double>(
            surface.points.size() + surface.cells.size()
        );
        return elapsed_ns(started, finished);
    });
    auto result = make_total_result(
        "fermi_surface_total",
        ndim,
        ndof,
        root_level,
        vertices,
        simplices,
        simplex_visits,
        refinements,
        config.samples,
        timing,
        lapack_ns
    );
    result.min_feature_size = min_feature_size;
    result.curvature_bound = curvature_bound;
    results.push_back(std::move(result));
}

void benchmark_adaptive_charge_total(
    std::size_t ndof,
    const Config &config,
    double lapack_ns,
    std::vector<Result> &results
) {
    constexpr auto ndim = std::size_t{2};
    constexpr auto root_level = std::uint32_t{1};
    const auto options = adaptivesimplex::adaptive::Options{
        .target_error = 0.02,
        .max_refinements = -1,
        .preview_depth = 0,
        .min_refinement_batch_size = 1,
        .max_refinement_batch_size = 100,
    };
    auto vertices = std::size_t{0};
    auto simplices = std::size_t{0};
    auto simplex_visits = std::size_t{0};
    auto refinements = std::size_t{0};
    auto visible_gapless_simplices = std::size_t{0};
    auto inconclusive_simplices = std::size_t{0};
    auto stopping_error = 0.0;
    auto target_reached = false;
    auto charge_error_stats = fermisimplex::ChargeErrorStats{};
    const auto timing = measure(config.samples, 1, [&](std::size_t) {
        auto mesh = fermisimplex::SpectralMesh(
            crossing_model(ndof), 1e-14, root_level
        );
        const auto started = Clock::now();
        const auto charge = fermisimplex::integrate_charge(
            mesh,
            0.0,
            options,
            1
        );
        const auto finished = Clock::now();
        require_stable_count(
            vertices,
            static_cast<std::size_t>(charge.stats.evaluations),
            "adaptive-charge vertex count"
        );
        require_stable_count(
            simplices,
            static_cast<std::size_t>(charge.stats.active_simplices),
            "adaptive-charge active-simplex count"
        );
        require_stable_count(
            simplex_visits,
            static_cast<std::size_t>(charge.stats.simplex_visits),
            "adaptive-charge simplex-visit count"
        );
        require_stable_count(
            refinements,
            static_cast<std::size_t>(charge.stats.refinements),
            "adaptive-charge refinement count"
        );
        require_stable_count(
            visible_gapless_simplices,
            static_cast<std::size_t>(charge.visible_gapless_simplices),
            "adaptive-charge visible-gapless count"
        );
        require_stable_count(
            inconclusive_simplices,
            static_cast<std::size_t>(charge.inconclusive_simplices),
            "adaptive-charge inconclusive count"
        );
        if (!charge.stats.target_reached ||
            charge.stopping_error > options.target_error) {
            throw std::runtime_error(
                "adaptive-charge benchmark did not reach its error target"
            );
        }
        stopping_error = charge.stopping_error;
        charge_error_stats = charge.error_stats;
        target_reached = charge.stats.target_reached;
        benchmark_sink = charge.value + charge.stopping_error;
        return elapsed_ns(started, finished);
    });
    auto result = make_total_result(
        "charge_adaptive_total",
        ndim,
        ndof,
        root_level,
        vertices,
        simplices,
        simplex_visits,
        refinements,
        config.samples,
        timing,
        lapack_ns
    );
    result.target_error = options.target_error;
    result.preview_depth = options.preview_depth;
    result.stopping_error = stopping_error;
    result.error_depth = 1;
    result.charge_error_stats = charge_error_stats;
    result.visible_gapless_simplices = visible_gapless_simplices;
    result.inconclusive_simplices = inconclusive_simplices;
    result.target_reached = target_reached;
    results.push_back(std::move(result));
}

void benchmark_root_mesh_case(
    std::size_t ndim,
    std::uint32_t root_level,
    std::size_t ndof,
    const Config &config,
    double lapack_ns,
    std::vector<Result> &results
) {
    auto probe = fermisimplex::SpectralMesh(
        fixed_model(ndim, ndof), 1e-14, root_level
    );
    const auto probe_simplices = active_simplices(probe);
    const auto probe_vertices = missing_active_vertices(probe, probe_simplices);
    const auto vertex_count = probe_vertices.size();
    const auto simplex_count = probe_simplices.size();

    const auto vertex_timing = measure(
        config.samples, vertex_count, [&](std::size_t) {
            auto mesh = fermisimplex::SpectralMesh(
                fixed_model(ndim, ndof), 1e-14, root_level
            );
            const auto simplices = active_simplices(mesh);
            const auto vertices = missing_active_vertices(mesh, simplices);
            const auto started = Clock::now();
            const auto checksum = fill_vertices(mesh, vertices);
            const auto finished = Clock::now();
            benchmark_sink = checksum;
            return elapsed_ns(started, finished);
        }
    );
    results.push_back(make_result(
        "root_mesh_vertex_fill", "scaling", "vertex", ndim, ndof, 0,
        root_level, vertex_count, simplex_count, vertex_count,
        config.samples, vertex_timing, lapack_ns
    ));

    const auto passes = std::max<std::size_t>(
        1,
        std::min<std::size_t>(16, 512 / simplex_count)
    );
    const auto classification_operations = passes * simplex_count;
    const auto classification_timing = measure(
        config.samples, classification_operations, [&](std::size_t) {
            auto mesh = fermisimplex::SpectralMesh(
                fixed_model(ndim, ndof), 1e-14, root_level
            );
            const auto simplices = active_simplices(mesh);
            const auto vertices = missing_active_vertices(mesh, simplices);
            benchmark_sink = fill_vertices(mesh, vertices);
            auto checksum = 0.0;
            const auto started = Clock::now();
            for (std::size_t pass = 0; pass < passes; ++pass) {
                const auto classified = fsdetail::classify_frontier(
                    mesh, simplices, 0.0, 2.0, 0.0
                );
                checksum += static_cast<double>(
                    classified.refine.size() + classified.terminal_surface.size()
                );
            }
            const auto finished = Clock::now();
            benchmark_sink = checksum;
            return elapsed_ns(started, finished);
        }
    );
    results.push_back(make_result(
        "root_mesh_classification", "scaling", "simplex", ndim, ndof, 0,
        root_level, vertex_count, simplex_count, classification_operations,
        config.samples, classification_timing, lapack_ns
    ));

    const auto total_timing = measure(
        config.samples, 1, [&](std::size_t) {
            auto mesh = fermisimplex::SpectralMesh(
                fixed_model(ndim, ndof), 1e-14, root_level
            );
            const auto simplices = active_simplices(mesh);
            const auto vertices = missing_active_vertices(mesh, simplices);
            const auto started = Clock::now();
            auto checksum = fill_vertices(mesh, vertices);
            const auto classified = fsdetail::classify_frontier(
                mesh, simplices, 0.0, 2.0, 0.0
            );
            checksum += static_cast<double>(
                classified.refine.size() + classified.terminal_surface.size()
            );
            const auto finished = Clock::now();
            benchmark_sink = checksum;
            return elapsed_ns(started, finished);
        }
    );
    results.push_back(make_result(
        "root_mesh_evaluate_and_classify", "scaling", "mesh", ndim, ndof, 0,
        root_level, vertex_count, simplex_count, 1, config.samples,
        total_timing, lapack_ns
    ));
}

std::string environment_value(const char *name) {
    const auto *value = std::getenv(name);
    return value == nullptr ? "unset" : value;
}

std::string json_escape(std::string_view value) {
    auto result = std::string{};
    result.reserve(value.size());
    for (const auto character : value) {
        switch (character) {
        case '\\': result += "\\\\"; break;
        case '"': result += "\\\""; break;
        case '\n': result += "\\n"; break;
        case '\r': result += "\\r"; break;
        case '\t': result += "\\t"; break;
        default: result += character; break;
        }
    }
    return result;
}

void write_result(std::ostream &output, const Result &result, bool trailing_comma) {
    output
        << "    {\n"
        << "      \"name\": \"" << json_escape(result.name) << "\",\n"
        << "      \"category\": \"" << json_escape(result.category) << "\",\n"
        << "      \"unit\": \"" << json_escape(result.unit) << "\",\n"
        << "      \"ndim\": " << result.ndim << ",\n"
        << "      \"ndof\": " << result.ndof << ",\n"
        << "      \"hopping_terms\": " << result.hopping_terms << ",\n"
        << "      \"target_bands\": " << result.target_bands << ",\n"
        << "      \"root_level\": " << result.root_level << ",\n"
        << "      \"vertices\": " << result.vertices << ",\n"
        << "      \"simplices\": " << result.simplices << ",\n"
        << "      \"simplex_visits\": " << result.simplex_visits << ",\n"
        << "      \"refinements\": " << result.refinements << ",\n"
        << "      \"target_error\": " << result.target_error << ",\n"
        << "      \"min_feature_size\": " << result.min_feature_size << ",\n"
        << "      \"curvature_bound\": " << result.curvature_bound << ",\n"
        << "      \"preview_depth\": " << result.preview_depth << ",\n"
        << "      \"error_depth\": " << result.error_depth << ",\n"
        << "      \"charge_root_simplices\": "
        << result.charge_error_stats.root_simplices << ",\n"
        << "      \"charge_hamiltonian_evaluations\": "
        << result.charge_error_stats.hamiltonian_evaluations << ",\n"
        << "      \"charge_full_eigensystems\": "
        << result.charge_error_stats.full_eigensystems << ",\n"
        << "      \"charge_reduced_eigensystems\": "
        << result.charge_error_stats.reduced_eigensystems << ",\n"
        << "      \"charge_norm_eigensystems\": "
        << result.charge_error_stats.norm_eigensystems << ",\n"
        << "      \"charge_safe_block_solves\": "
        << result.charge_error_stats.safe_block_solves << ",\n"
        << "      \"charge_schur_reductions\": "
        << result.charge_error_stats.schur_reductions << ",\n"
        << "      \"charge_micro_simplices\": "
        << result.charge_error_stats.micro_simplices << ",\n"
        << "      \"charge_terminal_simplices\": "
        << result.charge_error_stats.terminal_simplices << ",\n"
        << "      \"charge_conservative_fallbacks\": "
        << result.charge_error_stats.conservative_fallbacks << ",\n"
        << "      \"charge_singular_schur_failures\": "
        << result.charge_error_stats.singular_schur_failures << ",\n"
        << "      \"charge_initial_active_dimension_sum\": "
        << result.charge_error_stats.initial_active_dimension_sum << ",\n"
        << "      \"charge_terminal_active_dimension_sum\": "
        << result.charge_error_stats.terminal_active_dimension_sum << ",\n"
        << "      \"charge_minimum_active_dimension\": "
        << result.charge_error_stats.minimum_active_dimension << ",\n"
        << "      \"stopping_error\": " << result.stopping_error << ",\n"
        << "      \"visible_gapless_simplices\": "
        << result.visible_gapless_simplices << ",\n"
        << "      \"inconclusive_simplices\": "
        << result.inconclusive_simplices << ",\n"
        << "      \"target_reached\": "
        << (result.target_reached ? "true" : "false") << ",\n"
        << "      \"operations_per_sample\": "
        << result.operations_per_sample << ",\n"
        << "      \"samples\": " << result.samples << ",\n"
        << "      \"median_ns_per_operation\": "
        << result.timing.median_ns_per_operation << ",\n"
        << "      \"minimum_ns_per_operation\": "
        << result.timing.minimum_ns_per_operation << ",\n"
        << "      \"maximum_ns_per_operation\": "
        << result.timing.maximum_ns_per_operation << ",\n"
        << "      \"median_absolute_deviation_ns\": "
        << result.timing.median_absolute_deviation_ns << ",\n"
        << "      \"lapack_equivalents_per_operation\": "
        << result.lapack_equivalents_per_operation << ",\n"
        << "      \"median_ns_per_vertex\": "
        << result.median_ns_per_vertex << ",\n"
        << "      \"median_ns_per_simplex_visit\": "
        << result.median_ns_per_simplex_visit << ",\n"
        << "      \"total_lapack_equivalents_per_vertex\": "
        << result.total_lapack_equivalents_per_vertex << ",\n"
        << "      \"total_lapack_equivalents_per_simplex_visit\": "
        << result.total_lapack_equivalents_per_simplex_visit << "\n"
        << "    }" << (trailing_comma ? "," : "") << "\n";
}

std::string render_json(const Config &config, const std::vector<Result> &results) {
    auto output = std::ostringstream{};
    output << std::setprecision(12);
    output
        << "{\n"
        << "  \"schema_version\": 5,\n"
        << "  \"metadata\": {\n"
        << "    \"git_commit\": \""
        << json_escape(FERMISIMPLEX_BENCHMARK_GIT_COMMIT) << "\",\n"
        << "    \"git_dirty\": \""
        << json_escape(FERMISIMPLEX_BENCHMARK_GIT_DIRTY) << "\",\n"
        << "    \"compiler\": \""
        << json_escape(FERMISIMPLEX_BENCHMARK_COMPILER) << "\",\n"
        << "    \"build_type\": \""
        << json_escape(FERMISIMPLEX_BENCHMARK_BUILD_TYPE) << "\",\n"
        << "    \"system\": \""
        << json_escape(FERMISIMPLEX_BENCHMARK_SYSTEM) << "\",\n"
        << "    \"lapack\": \""
        << json_escape(FERMISIMPLEX_BENCHMARK_LAPACK) << "\",\n"
        << "    \"hardware_concurrency\": "
        << std::thread::hardware_concurrency() << ",\n"
        << "    \"openblas_num_threads\": \""
        << json_escape(environment_value("OPENBLAS_NUM_THREADS")) << "\",\n"
        << "    \"omp_num_threads\": \""
        << json_escape(environment_value("OMP_NUM_THREADS")) << "\",\n"
        << "    \"mkl_num_threads\": \""
        << json_escape(environment_value("MKL_NUM_THREADS")) << "\",\n"
        << "    \"veclib_maximum_threads\": \""
        << json_escape(environment_value("VECLIB_MAXIMUM_THREADS")) << "\"\n"
        << "  },\n"
        << "  \"config\": {\n"
        << "    \"preset\": \"" << json_escape(config.preset) << "\",\n"
        << "    \"samples\": " << config.samples << ",\n"
        << "    \"warmup_diagonalizations\": "
        << warmup_diagonalizations(config) << "\n"
        << "  },\n"
        << "  \"benchmarks\": [\n";
    for (std::size_t index = 0; index < results.size(); ++index) {
        write_result(output, results[index], index + 1 != results.size());
    }
    output
        << "  ],\n"
        << "  \"checksum\": " << benchmark_sink << "\n"
        << "}\n";
    return output.str();
}

std::string render_summary(const Config &config, const std::vector<Result> &results) {
    auto output = std::ostringstream{};
    output << std::fixed << std::setprecision(2);
    const auto workload_label = [](const Result &result) {
        return result.name == "charge_current_mesh_total"
            ? std::string_view{"charge (current mesh)"}
            : result.name == "charge_adaptive_total"
                ? std::string_view{"charge (adaptive)"}
                : std::string_view{"Fermi surface"};
    };
    const auto measurement = [&results](
        std::string_view name,
        std::size_t ndof
    ) -> const Result & {
        const auto found = std::find_if(
            results.begin(), results.end(), [&](const Result &result) {
                return result.name == name && result.ndof == ndof;
            }
        );
        if (found == results.end()) {
            throw std::runtime_error("missing measurement in summary");
        }
        return *found;
    };
    const auto lapack_us = [&](std::size_t ndof) {
        return measurement("lapack_reference_best", ndof)
            .timing.median_ns_per_operation / 1e3;
    };

    output
        << "FermiSimplex C++ performance (" << config.preset << ")\n"
        << "Primary metric: total eigensolve-equivalents per new vertex.\n"
        << "One eigensolve is the fastest measured LAPACK eigenvalue + "
           "eigenvector call.\n\n"
        << std::left << std::setw(27) << "workload"
        << std::right << std::setw(7) << "bands"
        << std::setw(12) << "LAPACK us"
        << std::setw(11) << "total ms"
        << std::setw(10) << "vertices"
        << std::setw(12) << "visits/vtx"
        << std::setw(12) << "us/vertex"
        << std::setw(15) << "solves/vertex"
        << '\n';
    for (const auto &result : results) {
        if (result.category != "end_to_end") {
            continue;
        }
        const auto visits_per_vertex =
            static_cast<double>(result.simplex_visits) /
            static_cast<double>(result.vertices);
        output
            << std::left << std::setw(27) << workload_label(result)
            << std::right << std::setw(7) << result.ndof
            << std::setw(12) << lapack_us(result.ndof)
            << std::setw(11) << result.timing.median_ns_per_operation / 1e6
            << std::setw(10) << result.vertices
            << std::setw(12) << visits_per_vertex
            << std::setw(12) << result.median_ns_per_vertex / 1e3
            << std::setw(15) << result.total_lapack_equivalents_per_vertex
            << '\n';
    }
    const auto has_charge_estimator = std::any_of(
        results.begin(),
        results.end(),
        [](const Result &result) {
            return result.category == "charge_estimator" ||
                result.name == "charge_current_mesh_total" ||
                result.name == "charge_adaptive_total";
        }
    );
    if (has_charge_estimator) {
        output
            << "\nCharge estimator diagnostics\n"
            << std::left << std::setw(31) << "workload"
            << std::right << std::setw(7) << "bands"
            << std::setw(7) << "root q"
            << std::setw(7) << "depth"
            << std::setw(11) << "total ms"
            << std::setw(11) << "eig/root"
            << std::setw(10) << "H/root"
            << std::setw(11) << "red/root"
            << std::setw(12) << "solve/root"
            << std::setw(10) << "micro"
            << std::setw(10) << "fallback"
            << '\n';
        for (const auto &result : results) {
            if (
                result.category != "charge_estimator" &&
                result.name != "charge_current_mesh_total" &&
                result.name != "charge_adaptive_total"
            ) {
                continue;
            }
            const auto roots = std::max<std::int64_t>(
                1, result.charge_error_stats.root_simplices
            );
            const auto mean_active = static_cast<double>(
                result.charge_error_stats.initial_active_dimension_sum
            ) / static_cast<double>(roots);
            output
                << std::left << std::setw(31)
                << (
                    result.category == "charge_estimator"
                        ? std::string_view{result.name}
                        : workload_label(result)
                )
                << std::right << std::setw(7) << result.ndof
                << std::setw(7) << mean_active
                << std::setw(7) << result.error_depth
                << std::setw(11)
                << result.timing.median_ns_per_operation / 1e6
                << std::setw(11)
                << result.lapack_equivalents_per_operation /
                    static_cast<double>(roots)
                << std::setw(10)
                << static_cast<double>(
                    result.charge_error_stats.hamiltonian_evaluations
                ) / static_cast<double>(roots)
                << std::setw(11)
                << static_cast<double>(
                    result.charge_error_stats.reduced_eigensystems
                ) / static_cast<double>(roots)
                << std::setw(12)
                << static_cast<double>(
                    result.charge_error_stats.safe_block_solves
                ) / static_cast<double>(roots)
                << std::setw(10)
                << result.charge_error_stats.micro_simplices
                << std::setw(10)
                << result.charge_error_stats.conservative_fallbacks
                << '\n';
        }
    }


    output
        << "\nPer-vertex pipeline scaling (LAPACK eigensystems)\n"
        << std::left << std::setw(9) << "bands"
        << std::right << std::setw(12) << "LAPACK us"
        << std::setw(13) << "model eval"
        << std::setw(13) << "Hamiltonian"
        << std::setw(13) << "eigensystem"
        << std::setw(15) << "cached vertex"
        << '\n';
    for (const auto ndof : config.matrix_sizes) {
        const auto &model = measurement("dense_model_evaluate", ndof);
        const auto &validation = measurement("hamiltonian_validate", ndof);
        const auto &spectrum = measurement("spectrum_dense_model", ndof);
        const auto &cached = measurement("spectrum_and_cache_insert", ndof);
        output
            << std::left << std::setw(9) << ndof
            << std::right << std::setw(12) << lapack_us(ndof)
            << std::setw(13)
            << model.lapack_equivalents_per_operation
            << std::setw(13)
            << validation.lapack_equivalents_per_operation
            << std::setw(13)
            << spectrum.lapack_equivalents_per_operation
            << std::setw(15)
            << cached.lapack_equivalents_per_operation
            << '\n';
    }
    output
        << "Stages are cumulative after model eval: Hamiltonian adds validation, "
           "eigensystem adds diagonalization,\n"
        << "and cached vertex adds eigensystem-cache insertion.\n";

    const auto diagnostic_ndof = *std::max_element(
        config.matrix_sizes.begin(), config.matrix_sizes.end()
    );
    output
        << "\nSecondary simplex diagnostic (" << diagnostic_ndof << " bands)\n"
        << std::left << std::setw(27) << "workload"
        << std::right << std::setw(10) << "visits"
        << std::setw(10) << "refine"
        << std::setw(12) << "visits/vtx"
        << std::setw(12) << "us/visit"
        << std::setw(15) << "solves/visit"
        << '\n';
    for (const auto &result : results) {
        if (result.category != "end_to_end" ||
            result.ndof != diagnostic_ndof) {
            continue;
        }
        output
            << std::left << std::setw(27) << workload_label(result)
            << std::right << std::setw(10) << result.simplex_visits
            << std::setw(10) << result.refinements
            << std::setw(12)
            << static_cast<double>(result.simplex_visits) /
                static_cast<double>(result.vertices)
            << std::setw(12) << result.median_ns_per_simplex_visit / 1e3
            << std::setw(15)
            << result.total_lapack_equivalents_per_simplex_visit
            << '\n';
    }

    output << "\nMachine-readable results: " << config.output.string() << '\n';
    return output.str();
}

Config preset(std::string name) {
    if (name == "quick") {
        return Config{
            .preset = "quick",
            .samples = 3,
            .matrix_sizes = {2, 8, 32},
            .tight_binding_terms = {1, 5},
        };
    }
    if (name == "ci") {
        return Config{};
    }
    if (name == "full") {
        return Config{
            .preset = "full",
            .samples = 7,
            .matrix_sizes = {2, 4, 8, 16, 32, 60, 128},
            .tight_binding_terms = {1, 5, 17, 33},
        };
    }
    throw std::runtime_error("unknown benchmark preset: " + name);
}

Config parse_arguments(int argc, char **argv) {
    auto preset_name = std::string{"ci"};
    auto output = std::filesystem::path{};
    auto only = std::string{};
    auto samples_override = std::size_t{0};
    for (int index = 1; index < argc; ++index) {
        const auto argument = std::string_view{argv[index]};
        if (argument == "--preset") {
            if (++index >= argc) {
                throw std::runtime_error("--preset requires a value");
            }
            preset_name = argv[index];
        } else if (argument == "--output") {
            if (++index >= argc) {
                throw std::runtime_error("--output requires a path");
            }
            output = argv[index];
        } else if (argument == "--only") {
            if (++index >= argc) {
                throw std::runtime_error("--only requires a value");
            }
            only = argv[index];
            if (only != "charge-estimator" &&
                only != "charge-scaling") {
                throw std::runtime_error("unknown --only value: " + only);
            }
        } else if (argument == "--samples") {
            if (++index >= argc) {
                throw std::runtime_error("--samples requires a value");
            }
            samples_override = static_cast<std::size_t>(std::stoull(argv[index]));
            if (samples_override == 0) {
                throw std::runtime_error("--samples must be positive");
            }
        } else if (argument == "--help" || argument == "-h") {
            std::cout
                << "Usage: fermisimplex_performance_benchmark "
                   "[--preset quick|ci|full] "
                   "[--only charge-estimator|charge-scaling] "
                   "[--samples N] [--output PATH]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(argument));
        }
    }
    auto config = preset(std::move(preset_name));
    config.output = std::move(output);
    config.only = std::move(only);
    if (config.only == "charge-scaling") {
        config.matrix_sizes = {60, 96, 128, 192, 256};
    } else if (config.only == "charge-estimator") {
        if (config.preset == "quick") {
            config.matrix_sizes = {16, 32};
        } else if (config.preset == "full") {
            config.matrix_sizes = {16, 64, 128};
        } else {
            config.matrix_sizes = {16, 64};
        }
    }
    if (samples_override != 0) {
        config.samples = samples_override;
    }
    return config;
}

std::vector<std::pair<std::size_t, std::uint32_t>> scaling_cases(
    const Config &config
) {
    if (config.preset == "quick") {
        return {{2, 2}, {2, 4}};
    }
    return {
        {1, 4}, {1, 6}, {1, 8},
        {2, 2}, {2, 3}, {2, 4},
        {3, 1}, {3, 2}, {3, 3},
    };
}

std::vector<Result> run_benchmarks(const Config &config) {
    warm_up(config);
    auto results = std::vector<Result>{};
    auto lapack_baselines =
        std::vector<std::pair<std::size_t, double>>{};

    for (const auto ndof : config.matrix_sizes) {
        const auto reused_lapack_ns =
            benchmark_reused_lapack(ndof, config, results);
        const auto lapack_ns = benchmark_vertex_pipeline(
            ndof, config, reused_lapack_ns, results
        );
        lapack_baselines.emplace_back(ndof, lapack_ns);

        if (config.only == "charge-estimator") {
            benchmark_charge_estimator_case(
                ndof, 1, 0, config, lapack_ns, results
            );
            benchmark_charge_estimator_case(
                ndof, 1, 1, config, lapack_ns, results
            );
            if (ndof >= 8) {
                benchmark_charge_estimator_case(
                    ndof,
                    std::min<std::size_t>(4, ndof / 4),
                    1,
                    config,
                    lapack_ns,
                    results
                );
            }
            if (ndof >= 16) {
                benchmark_charge_estimator_case(
                    ndof, ndof / 2, 1, config, lapack_ns, results
                );
            }
            if (
                ndof == config.matrix_sizes.back() &&
                config.preset != "quick"
            ) {
                benchmark_charge_estimator_case(
                    ndof,
                    std::min<std::size_t>(4, ndof / 4),
                    2,
                    config,
                    lapack_ns,
                    results
                );
            }
            continue;
        }

        for (const auto terms : config.tight_binding_terms) {
            benchmark_tight_binding(
                ndof, terms, config, lapack_ns, results
            );
        }
        benchmark_charge_total(ndof, config, lapack_ns, results);
        benchmark_adaptive_charge_total(
            ndof, config, lapack_ns, results
        );
        if (config.only.empty()) {
            benchmark_fermi_surface_total(
                ndof, config, lapack_ns, results
            );
        }
    }

    if (
        config.only == "charge-estimator" ||
        config.only == "charge-scaling"
    ) {
        return results;
    }

    constexpr auto scaling_ndof = std::size_t{8};
    const auto scaling_baseline = std::find_if(
        lapack_baselines.begin(),
        lapack_baselines.end(),
        [](const auto &entry) {
            return entry.first == scaling_ndof;
        }
    );
    if (scaling_baseline == lapack_baselines.end()) {
        throw std::runtime_error(
            "missing LAPACK baseline for scaling benchmark"
        );
    }
    for (const auto [ndim, root_level] : scaling_cases(config)) {
        benchmark_root_mesh_case(
            ndim,
            root_level,
            scaling_ndof,
            config,
            scaling_baseline->second,
            results
        );
    }
    return results;
}

}  // namespace

int main(int argc, char **argv) {
    try {
        const auto config = parse_arguments(argc, argv);
        const auto results = run_benchmarks(config);
        const auto rendered = render_json(config, results);
        if (!config.output.empty()) {
            const auto parent = config.output.parent_path();
            if (!parent.empty()) {
                std::filesystem::create_directories(parent);
            }
            auto file = std::ofstream{config.output};
            if (!file) {
                throw std::runtime_error(
                    "could not open benchmark output: " + config.output.string()
                );
            }
            file << rendered;
            std::cout << render_summary(config, results);
        } else {
            std::cout << rendered;
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "fermisimplex performance benchmark: " << error.what() << '\n';
        return 1;
    }
}
