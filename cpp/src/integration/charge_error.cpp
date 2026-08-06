#include "integration/charge_error.h"

#include "linalg/blas_lapack.h"

#include <adaptivesimplex/core/dyadic_vertex.h>
#include <adaptivesimplex/core/simplex_table.h>
#include <adaptivesimplex/core/vertex_table.h>
#include <adaptivesimplex/cut/simplex_moments.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fermisimplex::integration_detail {
namespace cert = certification;
namespace core = adaptivesimplex::core;
namespace cut = adaptivesimplex::cut;

namespace {

using Complex = std::complex<double>;
using Matrix = std::vector<Complex>;
using Point = core::DyadicVertex;

struct SingularSafeBlock {};

struct ChargeInterval {
    double lower = 0.0;
    double upper = 0.0;

    ChargeInterval &operator+=(const ChargeInterval &other) noexcept {
        lower += other.lower;
        upper += other.upper;
        return *this;
    }
};

using OccupationRange = cert::OccupationBounds;

std::size_t matrix_index(
    std::size_t row,
    std::size_t column,
    std::size_t rows
) {
    return row + column * rows;
}

bool finite(Complex value) {
    return std::isfinite(value.real()) && std::isfinite(value.imag());
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

void make_hermitian(Matrix &matrix, std::size_t size) {
    for (std::size_t column = 0; column < size; ++column) {
        matrix[matrix_index(column, column, size)] =
            Complex{matrix[matrix_index(column, column, size)].real(), 0.0};
        for (std::size_t row = column + 1; row < size; ++row) {
            const auto lower = matrix[matrix_index(row, column, size)];
            const auto upper = matrix[matrix_index(column, row, size)];
            const auto value = 0.5 * (lower + std::conj(upper));
            matrix[matrix_index(row, column, size)] = value;
            matrix[matrix_index(column, row, size)] = std::conj(value);
        }
    }
}

Eigensystem diagonalize(
    Matrix matrix,
    std::size_t size,
    bool compute_vectors,
    const char *context
) {
    auto result = Eigensystem{};
    linalg::diagonalize_hermitian_in_place(
        matrix,
        result.eigenvalues,
        size,
        compute_vectors,
        context
    );
    if (compute_vectors) {
        result.eigenvectors = std::move(matrix);
    }
    return result;
}

Matrix selected_columns(
    const Eigensystem &eigensystem,
    std::span<const std::size_t> columns
) {
    const auto size = eigensystem.eigenvalues.size();
    auto result = Matrix(size * columns.size());
    for (std::size_t output = 0; output < columns.size(); ++output) {
        for (std::size_t row = 0; row < size; ++row) {
            result[matrix_index(row, output, size)] =
                eigensystem.eigenvectors[
                    matrix_index(row, columns[output], size)
                ];
        }
    }
    return result;
}

struct SchurLayer {
    std::size_t parent_dimension = 0;
    std::size_t active_dimension = 0;
    Matrix active_basis;
    Matrix safe_basis;
};

Matrix apply_layer(
    std::span<const Complex> matrix,
    const SchurLayer &layer,
    ChargeErrorStats &stats
) {
    const auto size = layer.parent_dimension;
    const auto active = layer.active_dimension;
    const auto safe = size - active;

    const auto h_active = multiply(
        'N', 'N', size, active, size,
        matrix, size, layer.active_basis, size
    );
    auto result = multiply(
        'C', 'N', active, active, size,
        layer.active_basis, size, h_active, size
    );
    const auto coupling = multiply(
        'C', 'N', safe, active, size,
        layer.safe_basis, size, h_active, size
    );
    const auto h_safe = multiply(
        'N', 'N', size, safe, size,
        matrix, size, layer.safe_basis, size
    );
    auto safe_block = multiply(
        'C', 'N', safe, safe, size,
        layer.safe_basis, size, h_safe, size
    );
    make_hermitian(safe_block, safe);

    auto solved_coupling = coupling;
    ++stats.safe_block_solves;
    if (!linalg::solve_linear_system_in_place(
            safe_block,
            solved_coupling,
            safe,
            active,
            "charge-error Schur complement"
        )) {
        throw SingularSafeBlock{};
    }
    if (!std::all_of(
            solved_coupling.begin(),
            solved_coupling.end(),
            finite
        )) {
        throw SingularSafeBlock{};
    }

    const auto correction = multiply(
        'C', 'N', active, active, safe,
        coupling, safe, solved_coupling, safe
    );
    for (std::size_t index = 0; index < result.size(); ++index) {
        result[index] -= correction[index];
        if (!finite(result[index])) {
            throw SingularSafeBlock{};
        }
    }
    make_hermitian(result, active);
    return result;
}

class SharedHamiltonians {
public:
    SharedHamiltonians(
        SpectralMesh &mesh,
        double mu,
        ChargeErrorStats &stats
    ) : mesh_(mesh), mu_(mu), stats_(stats) {}

    void remember_spectrum(const Point &point, const Eigensystem &spectrum) {
        auto shifted = spectrum;
        for (auto &value : shifted.eigenvalues) {
            value -= mu_;
        }
        spectra_.insert_or_assign(point, std::move(shifted));
    }

    const Matrix &matrix(const Point &point) {
        const auto found = matrices_.find(point);
        if (found != matrices_.end()) {
            return found->second;
        }

        auto value = mesh_.hamiltonian(point.to_point());
        for (std::size_t index = 0; index < mesh_.ndof(); ++index) {
            value[matrix_index(index, index, mesh_.ndof())] -= mu_;
        }
        ++stats_.hamiltonian_evaluations;
        return matrices_.emplace(point, std::move(value)).first->second;
    }

    const Eigensystem &spectrum(const Point &point) {
        const auto found = spectra_.find(point);
        if (found != spectra_.end()) {
            return found->second;
        }

        auto result = diagonalize(
            matrix(point),
            mesh_.ndof(),
            true,
            "charge-error full Hamiltonian"
        );
        ++stats_.full_eigensystems;
        return spectra_.emplace(point, std::move(result)).first->second;
    }

    SpectralMesh &mesh() const noexcept {
        return mesh_;
    }

private:
    SpectralMesh &mesh_;
    double mu_ = 0.0;
    ChargeErrorStats &stats_;
    std::unordered_map<Point, Matrix, Point::Hash> matrices_;
    std::unordered_map<Point, Eigensystem, Point::Hash> spectra_;
};

class EffectiveModel {
public:
    EffectiveModel(
        SharedHamiltonians &hamiltonians,
        ChargeErrorStats &stats
    ) : hamiltonians_(hamiltonians),
        stats_(stats),
        dimension_(hamiltonians.mesh().ndof()) {}

    EffectiveModel(
        std::shared_ptr<const EffectiveModel> parent,
        SchurLayer layer,
        ChargeErrorStats &stats
    ) : hamiltonians_(parent->hamiltonians_),
        stats_(stats),
        parent_(std::move(parent)),
        layer_(std::move(layer)),
        dimension_(layer_->active_dimension) {}

    std::size_t dimension() const noexcept {
        return dimension_;
    }

    const Matrix &matrix(const Point &point) const {
        if (!parent_) {
            return hamiltonians_.matrix(point);
        }
        const auto found = matrices_.find(point);
        if (found != matrices_.end()) {
            return found->second;
        }
        auto value = apply_layer(parent_->matrix(point), *layer_, stats_);
        return matrices_.emplace(point, std::move(value)).first->second;
    }

    const Eigensystem &spectrum(const Point &point) const {
        if (!parent_) {
            return hamiltonians_.spectrum(point);
        }
        const auto found = spectra_.find(point);
        if (found != spectra_.end()) {
            return found->second;
        }
        auto result = diagonalize(
            matrix(point),
            dimension_,
            true,
            "charge-error reduced Hamiltonian"
        );
        ++stats_.reduced_eigensystems;
        return spectra_.emplace(point, std::move(result)).first->second;
    }

private:
    SharedHamiltonians &hamiltonians_;
    ChargeErrorStats &stats_;
    std::shared_ptr<const EffectiveModel> parent_;
    std::optional<SchurLayer> layer_;
    std::size_t dimension_ = 0;
    mutable std::unordered_map<Point, Matrix, Point::Hash> matrices_;
    mutable std::unordered_map<Point, Eigensystem, Point::Hash> spectra_;
};

struct SimplexState {
    std::vector<Point> points;
    std::vector<const Eigensystem *> spectra;
};

SimplexState simplex_state(
    const core::Geometry &geometry,
    core::SimplexId simplex_id,
    const EffectiveModel &model
) {
    const auto &simplex = geometry.simplices().simplex(simplex_id);
    auto result = SimplexState{};
    result.points.reserve(simplex.vertex_ids.size());
    result.spectra.reserve(simplex.vertex_ids.size());
    for (const auto vertex_id : simplex.vertex_ids) {
        result.points.push_back(
            geometry.vertices().dyadic_vertex(vertex_id)
        );
        result.spectra.push_back(&model.spectrum(result.points.back()));
    }
    return result;
}

cert::SimplexCertificate certify(
    const SimplexState &state,
    double radius,
    double tolerance
) {
    auto eigenvalues = std::vector<std::span<const double>>{};
    auto eigenvectors =
        std::vector<std::span<const std::complex<double>>>{};
    eigenvalues.reserve(state.spectra.size());
    eigenvectors.reserve(state.spectra.size());
    for (const auto *spectrum : state.spectra) {
        eigenvalues.emplace_back(spectrum->eigenvalues);
        eigenvectors.emplace_back(spectrum->eigenvectors);
    }
    return cert::certify_simplex(
        eigenvalues,
        eigenvectors,
        0.0,
        radius,
        tolerance
    );
}

std::size_t best_anchor(
    const SimplexState &state,
    cert::OccupationBounds bounds
) {
    auto best = std::size_t{0};
    auto best_score = -1.0;
    for (std::size_t vertex = 0; vertex < state.spectra.size(); ++vertex) {
        const auto &values = state.spectra[vertex]->eigenvalues;
        auto score = std::numeric_limits<double>::infinity();
        for (std::size_t band = 0; band < values.size(); ++band) {
            if (band < bounds.lower || band >= bounds.upper) {
                score = std::min(score, std::abs(values[band]));
            }
        }
        if (score > best_score) {
            best = vertex;
            best_score = score;
        }
    }
    return best;
}

SchurLayer make_layer(
    const Eigensystem &anchor,
    cert::OccupationBounds bounds
) {
    const auto size = anchor.eigenvalues.size();
    auto active_columns = std::vector<std::size_t>{};
    auto safe_columns = std::vector<std::size_t>{};
    const auto active = bounds.upper - bounds.lower;
    active_columns.reserve(active);
    safe_columns.reserve(size - active);
    for (std::size_t band = 0; band < size; ++band) {
        if (bounds.lower <= band && band < bounds.upper) {
            active_columns.push_back(band);
        } else {
            safe_columns.push_back(band);
        }
    }
    return SchurLayer{
        .parent_dimension = size,
        .active_dimension = active_columns.size(),
        .active_basis = selected_columns(anchor, active_columns),
        .safe_basis = selected_columns(anchor, safe_columns),
    };
}

struct MicroMesh {
    core::Geometry geometry;
    std::vector<core::SimplexId> simplex_ids;
};

MicroMesh subdivide(
    const core::Geometry &source_geometry,
    core::SimplexId source_id,
    std::uint32_t binary_depth
) {
    const auto &source =
        source_geometry.simplices().simplex(source_id);
    auto vertices = core::VertexTable(source_geometry.ndim());
    auto vertex_ids = std::vector<core::VertexId>{};
    vertex_ids.reserve(source.vertex_ids.size());
    for (const auto source_vertex : source.vertex_ids) {
        vertex_ids.push_back(vertices.get_or_add(
            source_geometry.vertices().dyadic_vertex(source_vertex)
        ));
    }

    auto simplices = core::SimplexTable{};
    const auto root_id =
        simplices.add(std::move(vertex_ids), source.volume);
    simplices.replace_active_simplices({root_id});
    auto geometry = core::Geometry(
        source_geometry.ndim(),
        std::move(vertices),
        std::move(simplices)
    );
    auto simplex_ids = geometry.preview_active(root_id, binary_depth);
    return {std::move(geometry), std::move(simplex_ids)};
}

double occupied_volume(
    double simplex_volume,
    std::span<const double> energies,
    double level,
    double tolerance
) {
    const auto moments = cut::simplex_moments(
        simplex_volume,
        energies,
        cut::LevelOptions{
            .level = level,
            .level_tolerance = tolerance,
        }
    );
    return moments.kind == cut::SimplexCutKind::on_level
        ? 0.5 * simplex_volume
        : moments.volume;
}

double defect_norm(
    Matrix matrix,
    std::size_t size,
    bool exact,
    ChargeErrorStats &stats
) {
    make_hermitian(matrix, size);
    if (!exact) {
        auto squared_norm = 0.0;
        for (const auto value : matrix) {
            squared_norm += std::norm(value);
        }
        return std::sqrt(squared_norm);
    }
    if (std::all_of(matrix.begin(), matrix.end(), [](Complex value) {
            return value == Complex{0.0, 0.0};
        })) {
        return 0.0;
    }

    auto values = std::vector<double>{};
    linalg::diagonalize_hermitian_in_place(
        matrix,
        values,
        size,
        false,
        "charge-error midpoint defect"
    );
    ++stats.norm_eigensystems;
    auto result = 0.0;
    for (const auto value : values) {
        result = std::max(result, std::abs(value));
    }
    return result;
}

}  // namespace

struct ChargeErrorEstimator::Impl {
    SpectralMesh &mesh;
    double mu = 0.0;
    std::uint32_t depth = 0;
    ChargeErrorStats &stats;

    Impl(
        SpectralMesh &mesh_,
        double mu_,
        std::uint32_t depth_,
        ChargeErrorStats &stats_
    ) : mesh(mesh_),
        mu(mu_),
        depth(depth_),
        stats(stats_) {}

    ChargeInterval fallback(
        OccupationRange range,
        double volume
    ) {
        ++stats.conservative_fallbacks;
        return {
            static_cast<double>(range.lower) * volume,
            static_cast<double>(range.upper) * volume,
        };
    }

    std::optional<std::pair<
        std::shared_ptr<EffectiveModel>,
        SimplexState
    >> reduce(
        const std::shared_ptr<EffectiveModel> &model,
        const core::Geometry &geometry,
        core::SimplexId simplex_id,
        const SimplexState &state,
        cert::OccupationBounds bounds
    ) {
        const auto active = bounds.upper - bounds.lower;
        if (active == 0 || active == model->dimension()) {
            return std::nullopt;
        }

        const auto anchor = best_anchor(state, bounds);
        auto candidate = std::make_shared<EffectiveModel>(
            model,
            make_layer(*state.spectra[anchor], bounds),
            stats
        );
        try {
            auto candidate_state =
                simplex_state(geometry, simplex_id, *candidate);
            ++stats.schur_reductions;
            if (stats.minimum_active_dimension == 0) {
                stats.minimum_active_dimension = active;
            } else {
                stats.minimum_active_dimension =
                    std::min(stats.minimum_active_dimension, active);
            }
            return std::pair{
                std::move(candidate),
                std::move(candidate_state),
            };
        } catch (const SingularSafeBlock &) {
            ++stats.singular_schur_failures;
            return std::nullopt;
        }
    }

    void record_terminal(std::size_t active) {
        ++stats.terminal_simplices;
        stats.terminal_active_dimension_sum +=
            static_cast<std::int64_t>(active);
    }

    ChargeInterval terminal(
        const std::shared_ptr<EffectiveModel> &model,
        const core::Geometry &geometry,
        core::SimplexId simplex_id,
        const SimplexState &state,
        std::size_t fixed_occupied,
        bool include_band_defect
    ) {
        const auto &simplex =
            geometry.simplices().simplex(simplex_id);
        const auto vertex_count = state.points.size();
        const auto size = model->dimension();
        const auto sample_defects = [&](bool include_bands) {
            auto maximum_matrix_defect = 0.0;
            auto maximum_band_defect = 0.0;
            for (std::size_t first = 0; first < vertex_count; ++first) {
                const auto &first_matrix =
                    model->matrix(state.points[first]);
                for (std::size_t second = first + 1;
                     second < vertex_count;
                     ++second) {
                    const auto midpoint = Point::midpoint(
                        state.points[first], state.points[second]
                    );
                    const auto &midpoint_matrix = model->matrix(midpoint);
                    const auto &second_matrix =
                        model->matrix(state.points[second]);
                    auto defect = Matrix(size * size);
                    for (std::size_t index = 0;
                         index < defect.size();
                         ++index) {
                        defect[index] = midpoint_matrix[index] -
                            0.5 * (first_matrix[index] + second_matrix[index]);
                    }
                    maximum_matrix_defect = std::max(
                        maximum_matrix_defect,
                        defect_norm(
                            std::move(defect),
                            size,
                            include_bands,
                            stats
                        )
                    );

                    if (include_bands) {
                        const auto &midpoint_spectrum =
                            model->spectrum(midpoint);
                        const auto &first_values =
                            state.spectra[first]->eigenvalues;
                        const auto &second_values =
                            state.spectra[second]->eigenvalues;
                        for (std::size_t band = 0; band < size; ++band) {
                            maximum_band_defect = std::max(
                                maximum_band_defect,
                                std::abs(
                                    midpoint_spectrum.eigenvalues[band] -
                                    0.5 * (
                                        first_values[band] +
                                        second_values[band]
                                    )
                                )
                            );
                        }
                    }
                }
            }
            return std::pair{
                maximum_matrix_defect,
                maximum_band_defect,
            };
        };
        const auto make_beta = [&](const auto &defects) {
            const auto dimension = static_cast<double>(mesh.ndim());
            return 2.0 * dimension / (dimension + 1.0) *
                   std::max(defects.first, defects.second);
        };

        auto beta = make_beta(sample_defects(include_band_defect));
        auto certificate = certify(state, beta, mesh.tolerance());
        auto bounds = certificate.occupation_bounds;
        auto active = bounds.upper - bounds.lower;
        if (!include_band_defect && active != 0) {
            beta = make_beta(sample_defects(true));
            certificate = certify(state, beta, mesh.tolerance());
            bounds = certificate.occupation_bounds;
            active = bounds.upper - bounds.lower;
        }
        record_terminal(active);

        const auto fixed =
            fixed_occupied + bounds.lower;
        if (active == 0) {
            const auto charge =
                static_cast<double>(fixed) * simplex.volume;
            return {charge, charge};
        }

        auto lower_charge =
            static_cast<double>(fixed) * simplex.volume;
        auto upper_charge = lower_charge;
        auto energies = std::vector<double>(vertex_count);
        for (std::size_t band = bounds.lower;
             band < bounds.upper;
             ++band) {
            for (std::size_t vertex = 0;
                 vertex < vertex_count;
                 ++vertex) {
                energies[vertex] =
                    state.spectra[vertex]->eigenvalues[band];
            }
            lower_charge += occupied_volume(
                simplex.volume,
                energies,
                -beta,
                mesh.tolerance()
            );
            upper_charge += occupied_volume(
                simplex.volume,
                energies,
                beta,
                mesh.tolerance()
            );
        }
        return {lower_charge, upper_charge};
    }

    ChargeInterval visit(
        const std::shared_ptr<EffectiveModel> &model,
        const core::Geometry &geometry,
        core::SimplexId simplex_id,
        std::size_t fixed_occupied,
        OccupationRange fallback_range,
        std::uint32_t logical_depth,
        std::optional<cert::SimplexCertificate> known_certificate = std::nullopt
    ) {
        ++stats.micro_simplices;
        const auto volume =
            geometry.simplices().simplex(simplex_id).volume;

        SimplexState state;
        try {
            state = simplex_state(geometry, simplex_id, *model);
        } catch (const SingularSafeBlock &) {
            ++stats.singular_schur_failures;
            record_terminal(fallback_range.upper - fallback_range.lower);
            return fallback(fallback_range, volume);
        }

        const auto certificate = known_certificate.has_value()
            ? *known_certificate
            : certify(state, 0.0, mesh.tolerance());
        const auto bounds = certificate.occupation_bounds;
        const auto active = bounds.upper - bounds.lower;

        auto current_model = model;
        auto current_state = std::move(state);
        auto current_fixed = fixed_occupied;
        const auto certified_fallback = OccupationRange{
            .lower = std::max(
                fallback_range.lower,
                fixed_occupied + bounds.lower
            ),
            .upper = std::min(
                fallback_range.upper,
                fixed_occupied + bounds.upper
            ),
        };
        auto current_fallback = fallback_range;
        if (certified_fallback.lower <= certified_fallback.upper) {
            current_fallback = certified_fallback;
        }
        if (const auto reduction = reduce(
                model,
                geometry,
                simplex_id,
                current_state,
                bounds
            )) {
            current_model = reduction->first;
            current_state = reduction->second;
            current_fixed += bounds.lower;
        }

        if (active == 0 || logical_depth == depth) {
            try {
                return terminal(
                    current_model,
                    geometry,
                    simplex_id,
                    current_state,
                    current_fixed,
                    active != 0
                );
            } catch (const SingularSafeBlock &) {
                ++stats.singular_schur_failures;
                record_terminal(
                    current_fallback.upper - current_fallback.lower
                );
                return fallback(current_fallback, volume);
            }
        }

        auto result = ChargeInterval{};
        auto children = subdivide(
            geometry,
            simplex_id,
            static_cast<std::uint32_t>(mesh.ndim())
        );
        for (const auto child : children.simplex_ids) {
            result += visit(
                current_model,
                children.geometry,
                child,
                current_fixed,
                current_fallback,
                logical_depth + 1
            );
        }
        return result;
    }

    double estimate(
        const core::Geometry &source_geometry,
        core::SimplexId source_id,
        double linear_charge,
        cert::SimplexCertificate root_certificate
    ) {
        ++stats.root_simplices;
        const auto &source =
            source_geometry.simplices().simplex(source_id);
        const auto root_bounds = root_certificate.occupation_bounds;
        const auto active =
            root_bounds.upper - root_bounds.lower;
        stats.initial_active_dimension_sum +=
            static_cast<std::int64_t>(active);

        const auto half_dimension = mesh.ndof() / 2 + mesh.ndof() % 2;
        if (active > 2 && active >= half_dimension) {
            ++stats.micro_simplices;
            record_terminal(active);
            const auto interval = fallback(
                {root_bounds.lower, root_bounds.upper},
                source.volume
            );
            return std::max(
                std::abs(linear_charge - interval.lower),
                std::abs(interval.upper - linear_charge)
            );
        }

        auto hamiltonians = SharedHamiltonians(mesh, mu, stats);
        for (const auto source_vertex_id : source.vertex_ids) {
            const auto &point =
                source_geometry.vertices().dyadic_vertex(source_vertex_id);
            hamiltonians.remember_spectrum(
                point,
                mesh.eigensystems().get(source_vertex_id)
            );
        }
        auto root = subdivide(source_geometry, source_id, 0);
        auto model =
            std::make_shared<EffectiveModel>(hamiltonians, stats);
        const auto interval = visit(
            model,
            root.geometry,
            root.simplex_ids.front(),
            0,
            {root_bounds.lower, root_bounds.upper},
            0,
            root_certificate
        );
        return std::max(
            std::abs(linear_charge - interval.lower),
            std::abs(interval.upper - linear_charge)
        );
    }
};

ChargeErrorEstimator::ChargeErrorEstimator(
    SpectralMesh &mesh,
    double mu,
    std::uint32_t depth,
    ChargeErrorStats &stats
) : impl_(std::make_unique<Impl>(mesh, mu, depth, stats)) {}

ChargeErrorEstimator::~ChargeErrorEstimator() = default;

double ChargeErrorEstimator::estimate(
    const core::Geometry &geometry,
    core::SimplexId simplex_id,
    double linear_charge,
    cert::SimplexCertificate root_certificate
) {
    return impl_->estimate(
        geometry,
        simplex_id,
        linear_charge,
        std::move(root_certificate)
    );
}

}  // namespace fermisimplex::integration_detail
