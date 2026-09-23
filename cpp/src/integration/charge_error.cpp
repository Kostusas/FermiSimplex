#include "integration/charge_error.h"

#include "integration/charge_error/cached_model.h"
#include "integration/charge_error/cut_disagreement.h"
#include "integration/charge_error/diagnostics.h"
#include "integration/charge_profile.h"

#include "certification/mesh_certificate.h"
#include "linalg/blas_lapack.h"

#include <adaptivesimplex/core/dyadic_vertex.h>
#include <adaptivesimplex/core/simplex_table.h>
#include <adaptivesimplex/core/vertex_table.h>
#include <adaptivesimplex/cut/simplex_moments.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fermisimplex::integration_detail {
namespace cert = certification;
namespace core = adaptivesimplex::core;
namespace cut = adaptivesimplex::cut;

namespace {

using charge_error_detail::ProfileTimer;
using Complex = std::complex<double>;
using Matrix = std::vector<Complex>;
using Point = core::DyadicVertex;
using Clock = std::chrono::steady_clock;

double elapsed_seconds(Clock::time_point started) {
    return std::chrono::duration<double>(Clock::now() - started).count();
}

struct ChargeInterval {
    double lower = 0.0;
    double upper = 0.0;
    double density_cut_error = 0.0;

    ChargeInterval &operator+=(const ChargeInterval &other) noexcept {
        lower += other.lower;
        upper += other.upper;
        density_cut_error += other.density_cut_error;
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

bool covers_radius(
    const cert::SimplexCertificate &certificate,
    double radius
) {
    return cert::occupation_bounds_valid_at(certificate, -radius) &&
           cert::occupation_bounds_valid_at(certificate, radius);
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

struct SimplexState {
    std::vector<Point> points;
    std::vector<const CachedSpectrum *> spectra;
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

// The reported charge cut is affine in the original simplex barycentric
// coordinates. Retain it while the temporary recursion changes both geometry
// and active-space basis.
class RootBandCuts {
public:
    RootBandCuts(
        const SimplexState &root,
        cert::OccupationBounds active_bounds
    )
        : origin_(root.points.front().to_point()),
          dimension_(origin_.size()),
          active_bounds_(active_bounds) {
        // Fixed roots terminate without child subdivision, so their
        // unshifted child cut is exactly the reported root cut.
        if (active_bounds_.lower == active_bounds_.upper) return;
        auto augmented = std::vector<std::vector<double>>(
            dimension_, std::vector<double>(2 * dimension_, 0.0)
        );
        for (std::size_t column = 0; column < dimension_; ++column) {
            const auto point = root.points[column + 1].to_point();
            for (std::size_t row = 0; row < dimension_; ++row) {
                augmented[row][column] = point[row] - origin_[row];
            }
            augmented[column][dimension_ + column] = 1.0;
        }
        for (std::size_t column = 0; column < dimension_; ++column) {
            auto pivot = column;
            for (std::size_t row = column + 1; row < dimension_; ++row) {
                if (std::abs(augmented[row][column]) >
                    std::abs(augmented[pivot][column])) {
                    pivot = row;
                }
            }
            std::swap(augmented[pivot], augmented[column]);
            const auto scale = augmented[column][column];
            for (auto &value : augmented[column]) {
                value /= scale;
            }
            for (std::size_t row = 0; row < dimension_; ++row) {
                if (row == column) continue;
                const auto factor = augmented[row][column];
                for (std::size_t j = 0; j < 2 * dimension_; ++j) {
                    augmented[row][j] -= factor * augmented[column][j];
                }
            }
        }
        inverse_.resize(dimension_ * dimension_);
        for (std::size_t row = 0; row < dimension_; ++row) {
            for (std::size_t column = 0; column < dimension_; ++column) {
                inverse_[row * dimension_ + column] =
                    augmented[row][dimension_ + column];
            }
        }
        for (std::size_t i = 0; i < root.points.size(); ++i) {
            root_spectra_.push_back(root.spectra[i]);
            auto weights = std::vector<double>(root.points.size(), 0.0);
            weights[i] = 1.0;
            cache_.emplace(root.points[i], std::move(weights));
        }
    }

    const std::vector<double> &weights(const Point &point) {
        if (const auto found = cache_.find(point); found != cache_.end()) {
            return found->second;
        }
        const auto coordinates = point.to_point();
        auto weights = std::vector<double>(dimension_ + 1, 0.0);
        weights[0] = 1.0;
        for (std::size_t row = 0; row < dimension_; ++row) {
            for (std::size_t column = 0; column < dimension_; ++column) {
                weights[row + 1] += inverse_[row * dimension_ + column] *
                    (coordinates[column] - origin_[column]);
            }
            weights[0] -= weights[row + 1];
        }
        return cache_.emplace(point, std::move(weights)).first->second;
    }

    double energy(std::span<const double> weights, std::size_t rank) const {
        auto result = 0.0;
        for (std::size_t i = 0; i < weights.size(); ++i) {
            result += weights[i] * root_spectra_[i]->eigenvalues[rank];
        }
        return result;
    }

    cert::OccupationBounds active_bounds() const noexcept {
        return active_bounds_;
    }

private:
    std::vector<double> origin_;
    std::size_t dimension_;
    cert::OccupationBounds active_bounds_;
    std::vector<double> inverse_;
    std::vector<const CachedSpectrum *> root_spectra_;
    std::unordered_map<Point, std::vector<double>, Point::Hash> cache_;
};

cert::SimplexCertificate certify(
    const SimplexState &state,
    double radius,
    double tolerance,
    ChargeProfile *profile
) {
    auto eigenvalues = std::vector<std::span<const double>>{};
    auto eigenvectors =
        std::vector<std::span<const std::complex<double>>>{};
    eigenvalues.reserve(state.spectra.size());
    eigenvectors.reserve(state.spectra.size());
    for (const auto *spectrum : state.spectra) {
        eigenvalues.emplace_back(spectrum->eigenvalues);
        eigenvectors.emplace_back(*spectrum->eigenvectors);
    }
    const auto preparation_started = profile == nullptr
        ? Clock::time_point{}
        : Clock::now();
    const auto prepared = cert::prepare_simplex_certificate(
        eigenvalues,
        eigenvectors,
        0.0,
        tolerance
    );
    const auto preparation_seconds = profile == nullptr
        ? 0.0
        : elapsed_seconds(preparation_started);
    const auto application_started = profile == nullptr
        ? Clock::time_point{}
        : Clock::now();
    auto result = prepared.certify(radius);
    if (profile != nullptr) {
        const auto application_seconds = elapsed_seconds(application_started);
        const auto total_seconds = preparation_seconds + application_seconds;
        profile->error_certification_seconds += total_seconds;
        profile->error_certification_prepare_seconds += preparation_seconds;
        profile->error_certification_apply_seconds += application_seconds;
        ++profile->error_certification_calls;
        ++profile->error_certification_direct_calls;
        profile->record_certification(
            state.spectra.front()->eigenvalues.size(),
            result.occupation_bounds.upper - result.occupation_bounds.lower,
            radius,
            total_seconds
        );
    }
    return result;
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

struct MicroMesh {
    core::Geometry geometry;
    std::vector<core::SimplexId> simplex_ids;
};

MicroMesh subdivide(
    const core::Geometry &source_geometry,
    core::SimplexId source_id,
    std::uint32_t binary_depth,
    ChargeProfile *profile
) {
    const auto started = profile == nullptr
        ? Clock::time_point{}
        : Clock::now();
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
    auto result = MicroMesh{std::move(geometry), std::move(simplex_ids)};
    if (profile != nullptr) {
        profile->error_subdivision_seconds += elapsed_seconds(started);
        ++profile->error_subdivision_calls;
    }
    return result;
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
    ChargeErrorStats &stats,
    ChargeProfile *profile
) {
    const auto overhead_started = profile == nullptr
        ? Clock::time_point{}
        : Clock::now();
    make_hermitian(matrix, size);
    if (!exact) {
        auto squared_norm = 0.0;
        for (const auto value : matrix) {
            squared_norm += std::norm(value);
        }
        if (profile != nullptr) {
            profile->error_defect_norm_overhead_seconds +=
                elapsed_seconds(overhead_started);
        }
        return std::sqrt(squared_norm);
    }
    if (std::all_of(matrix.begin(), matrix.end(), [](Complex value) {
            return value == Complex{0.0, 0.0};
        })) {
        if (profile != nullptr) {
            profile->error_defect_norm_overhead_seconds +=
                elapsed_seconds(overhead_started);
        }
        return 0.0;
    }
    if (size == 1) {
        if (profile != nullptr) {
            profile->error_defect_norm_overhead_seconds +=
                elapsed_seconds(overhead_started);
        }
        return std::abs(matrix.front().real());
    }
    if (size == 2) {
        const auto first = matrix[matrix_index(0, 0, size)].real();
        const auto second = matrix[matrix_index(1, 1, size)].real();
        const auto off_diagonal = matrix[matrix_index(1, 0, size)];
        const auto center = 0.5 * (first + second);
        const auto half_difference = 0.5 * (first - second);
        const auto radius = std::sqrt(
            half_difference * half_difference + std::norm(off_diagonal)
        );
        const auto result = std::max(
            std::abs(center - radius),
            std::abs(center + radius)
        );
        if (profile != nullptr) {
            profile->error_defect_norm_overhead_seconds +=
                elapsed_seconds(overhead_started);
        }
        return result;
    }

    if (profile != nullptr) {
        profile->error_defect_norm_overhead_seconds +=
            elapsed_seconds(overhead_started);
    }
    auto values = std::vector<double>{};
    const auto started = profile == nullptr
        ? Clock::time_point{}
        : Clock::now();
    linalg::diagonalize_hermitian_in_place(
        matrix,
        values,
        size,
        false,
        "charge-error midpoint defect"
    );
    if (profile != nullptr) {
        const auto seconds = elapsed_seconds(started);
        profile->error_norm_eigensystem_seconds += seconds;
        profile->record_norm(size, seconds);
    }
    ++stats.norm_eigensystems;
    auto result = 0.0;
    const auto postprocess_started = profile == nullptr
        ? Clock::time_point{}
        : Clock::now();
    for (const auto value : values) {
        result = std::max(result, std::abs(value));
    }
    if (profile != nullptr) {
        profile->error_defect_norm_overhead_seconds +=
            elapsed_seconds(postprocess_started);
    }
    return result;
}

}  // namespace

struct ChargeErrorEstimator::Impl {
    SpectralMesh &mesh;
    std::uint32_t depth = 0;
    ChargeErrorStats &stats;
    ChargeProfile *profile = nullptr;
    ModelBackend backend;

    Impl(
        SpectralMesh &mesh_,
        double mu_,
        std::uint32_t depth_,
        ChargeErrorStats &stats_,
        ChargeProfile *profile_
    ) : mesh(mesh_),
        depth(depth_),
        stats(stats_),
        profile(profile_),
        backend(mesh_, mu_, stats_, profile_) {}

    struct ActiveSpaceReduction {
        std::unique_ptr<EffectiveModel> model;
        SimplexState state;
    };

    ChargeInterval conservative_interval(
        OccupationRange range,
        double volume,
        bool reduction_failed = false
    ) {
        ++stats.conservative_fallbacks;
        return {
            static_cast<double>(range.lower) * volume,
            static_cast<double>(range.upper) * volume,
            static_cast<double>(
                reduction_failed ? mesh.ndof() : range.upper - range.lower
            ) * volume,
        };
    }

    std::optional<ActiveSpaceReduction> try_reduce_active_space(
        const EffectiveModel &model,
        const core::Geometry &geometry,
        core::SimplexId simplex_id,
        const SimplexState &state,
        cert::OccupationBounds bounds
    ) {
        const auto active = bounds.upper - bounds.lower;
        if (active == 0 || active == model.dimension()) {
            return std::nullopt;
        }

        const auto anchor = best_anchor(state, bounds);
        try {
            auto candidate = make_active_space_model(
                model,
                *state.spectra[anchor],
                bounds.lower,
                bounds.upper
            );
            auto candidate_state =
                simplex_state(geometry, simplex_id, *candidate);
            ++stats.schur_reductions;
            if (stats.minimum_active_dimension == 0) {
                stats.minimum_active_dimension = active;
            } else {
                stats.minimum_active_dimension =
                    std::min(stats.minimum_active_dimension, active);
            }
            return ActiveSpaceReduction{
                .model = std::move(candidate),
                .state = std::move(candidate_state),
            };
        } catch (const SchurFailure &) {
            ++stats.schur_failures;
            return std::nullopt;
        }
    }

    void record_terminal(std::size_t active) {
        ++stats.terminal_simplices;
        stats.terminal_active_dimension_sum +=
            static_cast<std::int64_t>(active);
    }

    struct MidpointDefects {
        double matrix = 0.0;
        double band = 0.0;
    };

    MidpointDefects measure_midpoint_defects(
        const EffectiveModel &model,
        const SimplexState &state,
        bool include_band_defect
    ) {
        const auto vertex_count = state.points.size();
        const auto size = model.dimension();
        auto result = MidpointDefects{};
        const auto use_hopping_gram =
            !include_band_defect && model.supports_frobenius_defect();

        for (std::size_t first = 0; first < vertex_count; ++first) {
            for (std::size_t second = first + 1;
                 second < vertex_count;
                 ++second) {
                const auto midpoint = Point::midpoint(
                    state.points[first], state.points[second]
                );
                if (use_hopping_gram) {
                    auto matrix_defect = 0.0;
                    {
                        const ProfileTimer timer(
                            profile,
                            &ChargeProfile::error_defect_norm_overhead_seconds
                        );
                        matrix_defect = model.frobenius_defect(
                            state.points[first],
                            state.points[second],
                            midpoint
                        );
                    }
                    if (profile != nullptr) {
                        ++profile->error_defect_samples;
                    }
                    result.matrix = std::max(
                        result.matrix, matrix_defect
                    );
                    continue;
                }

                const auto &first_matrix =
                    model.matrix(state.points[first]);
                const auto &midpoint_matrix = model.matrix(midpoint);
                const auto &second_matrix =
                    model.matrix(state.points[second]);
                auto defect = Matrix{};
                {
                    const ProfileTimer timer(
                        profile,
                        &ChargeProfile::error_defect_assembly_seconds
                    );
                    defect.resize(size * size);
                    for (std::size_t index = 0;
                         index < defect.size();
                         ++index) {
                        defect[index] = midpoint_matrix[index] -
                            0.5 * (
                                first_matrix[index] +
                                second_matrix[index]
                            );
                    }
                }
                if (profile != nullptr) {
                    ++profile->error_defect_samples;
                }

                const auto matrix_defect = defect_norm(
                    std::move(defect),
                    size,
                    include_band_defect,
                    stats,
                    profile
                );
                result.matrix = std::max(
                    result.matrix, matrix_defect
                );

                if (!include_band_defect) {
                    continue;
                }
                if (size == 1) {
                    result.band = std::max(
                        result.band, matrix_defect
                    );
                    continue;
                }

                const auto &midpoint_values =
                    model.eigenvalues(midpoint);
                const auto &first_values =
                    state.spectra[first]->eigenvalues;
                const auto &second_values =
                    state.spectra[second]->eigenvalues;
                {
                    const ProfileTimer timer(
                        profile,
                        &ChargeProfile::error_band_comparison_seconds
                    );
                    for (std::size_t band = 0; band < size; ++band) {
                        result.band = std::max(
                            result.band,
                            std::abs(
                                midpoint_values[band] -
                                0.5 * (
                                    first_values[band] +
                                    second_values[band]
                                )
                            )
                        );
                    }
                }
                if (profile != nullptr) {
                    ++profile->error_band_comparisons;
                }
            }
        }
        return result;
    }

    double defect_radius(const MidpointDefects &defects) const {
        const auto dimension = static_cast<double>(mesh.ndim());
        return 2.0 * dimension / (dimension + 1.0) *
               std::max(defects.matrix, defects.band);
    }

    cert::SimplexCertificate certify_terminal_radius(
        const SimplexState &state,
        std::size_t model_dimension,
        double radius,
        const cert::SimplexCertificate *reusable_certificate,
        const cert::PreparedSimplexCertificate *reusable_preparation
    ) {
        if (
            reusable_certificate != nullptr &&
            covers_radius(*reusable_certificate, radius)
        ) {
            if (profile != nullptr) {
                ++profile->error_certification_reused_calls;
            }
            return *reusable_certificate;
        }
        if (reusable_preparation != nullptr) {
            const auto started = profile == nullptr
                ? Clock::time_point{}
                : Clock::now();
            const auto occupation_bounds =
                reusable_preparation->occupation_bounds(radius);
            auto result = cert::SimplexCertificate{
                .status = cert::SimplexCertificateStatus::Inconclusive,
                .occupation_bounds = occupation_bounds,
            };
            if (profile != nullptr) {
                const auto seconds = elapsed_seconds(started);
                profile->error_certification_seconds += seconds;
                profile->error_certification_apply_seconds += seconds;
                ++profile->error_certification_calls;
                ++profile->error_certification_prepared_calls;
                profile->record_certification(
                    model_dimension,
                    result.occupation_bounds.upper -
                        result.occupation_bounds.lower,
                    radius,
                    seconds
                );
            }
            return result;
        }
        return certify(state, radius, mesh.tolerance(), profile);
    }

    std::optional<ChargeInterval> try_terminal_reduction(
        const EffectiveModel &model,
        const core::Geometry &geometry,
        core::SimplexId simplex_id,
        const SimplexState &state,
        std::size_t fixed_occupied,
        cert::OccupationBounds bounds,
        RootBandCuts &root_cut
    ) {
        const auto active = bounds.upper - bounds.lower;
        if (active == 0 || active == model.dimension()) {
            return std::nullopt;
        }
        if (const auto reduction = try_reduce_active_space(
                model,
                geometry,
                simplex_id,
                state,
                bounds
            )) {
            return estimate_terminal_interval(
                *reduction->model,
                geometry,
                simplex_id,
                reduction->state,
                fixed_occupied + bounds.lower,
                true,
                nullptr,
                nullptr,
                root_cut
            );
        }
        return std::nullopt;
    }

    ChargeInterval integrate_charge_interval(
        const core::Simplex &simplex,
        const SimplexState &state,
        std::size_t fixed_occupied,
        cert::OccupationBounds bounds,
        double radius
    ) {
        const auto fixed = fixed_occupied + bounds.lower;
        const auto active = bounds.upper - bounds.lower;
        if (active == 0) {
            const auto charge =
                static_cast<double>(fixed) * simplex.volume;
            return {charge, charge};
        }

        auto lower_charge =
            static_cast<double>(fixed) * simplex.volume;
        auto upper_charge = lower_charge;
        auto energies = std::vector<double>(state.points.size());
        for (std::size_t band = bounds.lower;
             band < bounds.upper;
             ++band) {
            for (std::size_t vertex = 0;
                 vertex < state.points.size();
                 ++vertex) {
                energies[vertex] =
                    state.spectra[vertex]->eigenvalues[band];
            }
            {
                const ProfileTimer timer(
                    profile,
                    &ChargeProfile::error_occupied_volume_seconds
                );
                lower_charge += occupied_volume(
                    simplex.volume,
                    energies,
                    -radius,
                    mesh.tolerance()
                );
                upper_charge += occupied_volume(
                    simplex.volume,
                    energies,
                    radius,
                    mesh.tolerance()
                );
            }
            if (profile != nullptr) {
                profile->error_occupied_volume_calls += 2;
            }
        }
        return {lower_charge, upper_charge};
    }

    double cut_disagreement_on_simplex(
        const core::Simplex &simplex,
        const SimplexState &state,
        std::size_t fixed_occupied,
        const EffectiveModel &model,
        cert::OccupationBounds child_bounds,
        RootBandCuts &root_cut
    ) {
        if (root_cut.active_bounds().lower ==
            root_cut.active_bounds().upper) return 0.0;
        const auto root_bounds = root_cut.active_bounds();
        const auto lower = std::min(
            root_bounds.lower, fixed_occupied + child_bounds.lower
        );
        const auto upper = std::max(
            root_bounds.upper, fixed_occupied + child_bounds.upper
        );
        if (lower == upper) return 0.0;
        auto original_vertices = std::vector<const std::vector<double> *>{};
        original_vertices.reserve(state.points.size());
        for (const auto &point : state.points) {
            original_vertices.push_back(&root_cut.weights(point));
        }
        auto original = std::vector<double>(state.points.size());
        auto child = std::vector<double>(state.points.size());
        auto disagreement = 0.0;
        // Compare the union of the root and child uncertain ranks. A root
        // with fixed occupation can still contain a child Fermi crossing.
        for (std::size_t rank = lower; rank < upper; ++rank) {
            for (std::size_t vertex = 0; vertex < state.points.size(); ++vertex) {
                original[vertex] = root_cut.energy(
                    *original_vertices[vertex], rank
                );
            }
            if (rank < fixed_occupied ||
                rank >= fixed_occupied + model.dimension()) {
                const auto original_volume = charge_error_detail::occupied_volume(
                    simplex.volume,
                    original,
                    mesh.tolerance(),
                    charge_error_detail::classify_cut(original, mesh.tolerance())
                );
                disagreement += rank < fixed_occupied
                    ? simplex.volume - original_volume : original_volume;
                continue;
            }
            for (std::size_t vertex = 0; vertex < state.points.size(); ++vertex) {
                child[vertex] = state.spectra[vertex]->eigenvalues[
                    rank - fixed_occupied
                ];
            }
            disagreement += charge_error_detail::cut_disagreement(
                simplex.volume, original, child, mesh.tolerance()
            );
        }
        return disagreement;
    }

    ChargeInterval estimate_terminal_interval(
        const EffectiveModel &model,
        const core::Geometry &geometry,
        core::SimplexId simplex_id,
        const SimplexState &state,
        std::size_t fixed_occupied,
        bool include_band_defect,
        const cert::SimplexCertificate *reusable_certificate,
        const cert::PreparedSimplexCertificate *reusable_preparation,
        RootBandCuts &root_cut
    ) {
        auto radius = defect_radius(
            measure_midpoint_defects(
                model, state, include_band_defect
            )
        );
        auto certificate = certify_terminal_radius(
            state,
            model.dimension(),
            radius,
            reusable_certificate,
            reusable_preparation
        );
        auto bounds = certificate.occupation_bounds;
        auto active = bounds.upper - bounds.lower;

        if (!include_band_defect && active != 0) {
            if (auto reduced = try_terminal_reduction(
                    model,
                    geometry,
                    simplex_id,
                    state,
                    fixed_occupied,
                    bounds,
                    root_cut
                )) {
                return *reduced;
            }

            radius = defect_radius(
                measure_midpoint_defects(model, state, true)
            );
            certificate = certify_terminal_radius(
                state,
                model.dimension(),
                radius,
                reusable_certificate,
                reusable_preparation
            );
            bounds = certificate.occupation_bounds;
            active = bounds.upper - bounds.lower;
        }

        record_terminal(active);
        auto result = integrate_charge_interval(
            geometry.simplices().simplex(simplex_id),
            state,
            fixed_occupied,
            bounds,
            radius
        );
        result.density_cut_error = result.upper - result.lower +
            cut_disagreement_on_simplex(
                geometry.simplices().simplex(simplex_id),
                state, fixed_occupied, model, bounds, root_cut
            );
        return result;
    }

    ChargeInterval estimate_micro_simplex(
        const EffectiveModel &model,
        const core::Geometry &geometry,
        core::SimplexId simplex_id,
        std::size_t fixed_occupied,
        OccupationRange fallback_range,
        std::uint32_t logical_depth,
        RootBandCuts &root_cut,
        std::optional<cert::SimplexCertificate> known_certificate = std::nullopt,
        const cert::PreparedSimplexCertificate *known_preparation = nullptr
    ) {
        ++stats.micro_simplices;
        if (profile != nullptr) {
            ChargeProfile::increment_at(
                profile->error_visit_depth_counts, logical_depth
            );
        }
        const auto volume =
            geometry.simplices().simplex(simplex_id).volume;

        SimplexState state;
        try {
            state = simplex_state(geometry, simplex_id, model);
        } catch (const SchurFailure &) {
            ++stats.schur_failures;
            record_terminal(fallback_range.upper - fallback_range.lower);
            return conservative_interval(fallback_range, volume, true);
        }

        const auto certificate = known_certificate.has_value()
            ? *known_certificate
            : certify(state, 0.0, mesh.tolerance(), profile);
        const auto bounds = certificate.occupation_bounds;
        const auto active = bounds.upper - bounds.lower;

        const auto *current_model = &model;
        auto reduced_model = std::unique_ptr<EffectiveModel>{};
        auto current_state = std::move(state);
        const cert::SimplexCertificate *current_certificate = &certificate;
        const auto *current_preparation = known_preparation;
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
        if (auto reduction = try_reduce_active_space(
                model,
                geometry,
                simplex_id,
                current_state,
                bounds
            )) {
            current_state = std::move(reduction->state);
            reduced_model = std::move(reduction->model);
            current_model = reduced_model.get();
            current_fixed += bounds.lower;
            current_certificate = nullptr;
            current_preparation = nullptr;
        }

        if (active == 0 || logical_depth == depth) {
            if (profile != nullptr) {
                ChargeProfile::increment_at(
                    profile->error_terminal_dimension_counts,
                    current_model->dimension()
                );
                ChargeProfile::increment_at(
                    profile->error_terminal_depth_counts, logical_depth
                );
            }
            try {
                return estimate_terminal_interval(
                    *current_model,
                    geometry,
                    simplex_id,
                    current_state,
                    current_fixed,
                    active != 0,
                    current_certificate,
                    current_preparation,
                    root_cut
                );
            } catch (const SchurFailure &) {
                ++stats.schur_failures;
                record_terminal(
                    current_fallback.upper - current_fallback.lower
                );
                return conservative_interval(current_fallback, volume, true);
            }
        }

        auto result = ChargeInterval{};
        if (profile != nullptr) {
            ChargeProfile::increment_at(
                profile->error_subdivision_depth_counts, logical_depth
            );
        }
        auto children = subdivide(
            geometry,
            simplex_id,
            static_cast<std::uint32_t>(mesh.ndim()),
            profile
        );
        for (const auto child : children.simplex_ids) {
            result += estimate_micro_simplex(
                *current_model,
                children.geometry,
                child,
                current_fixed,
                current_fallback,
                logical_depth + 1,
                root_cut
            );
        }
        return result;
    }

    ChargeErrorEstimator::Estimate estimate(
        const core::Geometry &source_geometry,
        core::SimplexId source_id,
        double linear_charge,
        cert::SimplexCertificate root_certificate,
        const cert::PreparedSimplexCertificate &root_preparation
    ) {
        auto workspace = RootErrorWorkspace(backend);
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
            const auto interval = conservative_interval(
                {root_bounds.lower, root_bounds.upper},
                source.volume
            );
            return {
                std::max(
                    std::abs(linear_charge - interval.lower),
                    std::abs(interval.upper - linear_charge)
                ),
                interval.density_cut_error,
            };
        }

        for (const auto source_vertex_id : source.vertex_ids) {
            const auto &point =
                source_geometry.vertices().dyadic_vertex(source_vertex_id);
            workspace.remember_spectrum(
                point,
                mesh.eigensystems().get(source_vertex_id)
            );
        }
        auto root = subdivide(source_geometry, source_id, 0, profile);
        auto model = EffectiveModel(workspace);
        const auto root_state = simplex_state(
            root.geometry, root.simplex_ids.front(), model
        );
        auto root_cut = RootBandCuts(root_state, root_bounds);
        const auto interval = estimate_micro_simplex(
            model,
            root.geometry,
            root.simplex_ids.front(),
            0,
            {root_bounds.lower, root_bounds.upper},
            0,
            root_cut,
            root_certificate,
            &root_preparation
        );
        return {
            std::max(
                std::abs(linear_charge - interval.lower),
                std::abs(interval.upper - linear_charge)
            ),
            interval.density_cut_error,
        };
    }
};

ChargeErrorEstimator::ChargeErrorEstimator(
    SpectralMesh &mesh,
    double mu,
    std::uint32_t depth,
    ChargeErrorStats &stats,
    ChargeProfile *profile
) : impl_(std::make_unique<Impl>(
        mesh, mu, depth, stats, profile
    )) {}

ChargeErrorEstimator::~ChargeErrorEstimator() = default;

ChargeErrorEstimator::Estimate ChargeErrorEstimator::estimate(
    const core::Geometry &geometry,
    core::SimplexId simplex_id,
    double linear_charge,
    cert::SimplexCertificate root_certificate,
    const cert::PreparedSimplexCertificate &root_preparation
) {
    return impl_->estimate(
        geometry,
        simplex_id,
        linear_charge,
        std::move(root_certificate),
        root_preparation
    );
}

}  // namespace fermisimplex::integration_detail
