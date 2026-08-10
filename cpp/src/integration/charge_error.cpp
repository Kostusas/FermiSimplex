#include "integration/charge_error.h"

#include "integration/charge_error/cached_model.h"
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
        double volume
    ) {
        ++stats.conservative_fallbacks;
        return {
            static_cast<double>(range.lower) * volume,
            static_cast<double>(range.upper) * volume,
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
        cert::OccupationBounds bounds
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
                nullptr
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

    ChargeInterval estimate_terminal_interval(
        const EffectiveModel &model,
        const core::Geometry &geometry,
        core::SimplexId simplex_id,
        const SimplexState &state,
        std::size_t fixed_occupied,
        bool include_band_defect,
        const cert::SimplexCertificate *reusable_certificate,
        const cert::PreparedSimplexCertificate *reusable_preparation
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
                    bounds
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
        return integrate_charge_interval(
            geometry.simplices().simplex(simplex_id),
            state,
            fixed_occupied,
            bounds,
            radius
        );
    }

    ChargeInterval estimate_micro_simplex(
        const EffectiveModel &model,
        const core::Geometry &geometry,
        core::SimplexId simplex_id,
        std::size_t fixed_occupied,
        OccupationRange fallback_range,
        std::uint32_t logical_depth,
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
            return conservative_interval(fallback_range, volume);
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
                    current_preparation
                );
            } catch (const SchurFailure &) {
                ++stats.schur_failures;
                record_terminal(
                    current_fallback.upper - current_fallback.lower
                );
                return conservative_interval(current_fallback, volume);
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
                logical_depth + 1
            );
        }
        return result;
    }

    double estimate(
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
            return std::max(
                std::abs(linear_charge - interval.lower),
                std::abs(interval.upper - linear_charge)
            );
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
        const auto interval = estimate_micro_simplex(
            model,
            root.geometry,
            root.simplex_ids.front(),
            0,
            {root_bounds.lower, root_bounds.upper},
            0,
            root_certificate,
            &root_preparation
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
    ChargeErrorStats &stats,
    ChargeProfile *profile
) : impl_(std::make_unique<Impl>(
        mesh, mu, depth, stats, profile
    )) {}

ChargeErrorEstimator::~ChargeErrorEstimator() = default;

double ChargeErrorEstimator::estimate(
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
