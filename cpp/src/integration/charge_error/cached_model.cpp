#include "integration/charge_error/cached_model.h"

#include "integration/charge_error/diagnostics.h"
#include "integration/charge_profile.h"

#include "core/tight_binding_access.h"
#include "linalg/blas_lapack.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>
#include <span>
#include <utility>
#include <vector>

namespace fermisimplex::integration_detail {
namespace {

using Complex = std::complex<double>;
using Matrix = CachedMatrix;
using Point = CachedPoint;
using Clock = std::chrono::steady_clock;
using charge_error_detail::ProfileTimer;

std::size_t matrix_index(
    std::size_t row,
    std::size_t column,
    std::size_t rows
) {
    return row + column * rows;
}

double elapsed_seconds(Clock::time_point started) {
    return std::chrono::duration<double>(Clock::now() - started).count();
}

CachedSpectrum diagonalize(
    Matrix matrix,
    std::size_t size,
    bool compute_vectors,
    const char *context
) {
    auto result = CachedSpectrum{};
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

Complex conjugate_dot(
    std::span<const Complex> left,
    std::span<const Complex> right
) {
    auto result = Complex{0.0, 0.0};
    for (std::size_t index = 0; index < left.size(); ++index) {
        result += std::conj(left[index]) * right[index];
    }
    return result;
}

}  // namespace

SchurEigensystemView schur_eigensystem_view(
    const CachedSpectrum &spectrum
) {
    if (!spectrum.eigenvectors.has_value()) {
        throw SchurFailure{};
    }
    return {
        .eigenvalues = spectrum.eigenvalues,
        .eigenvectors = *spectrum.eigenvectors,
    };
}

PointCache::PointCache(ChargeProfile *profile) : profile_(profile) {}

PointCache::~PointCache() {
    if (profile_ == nullptr) {
        return;
    }
    profile_->error_cache_live_records -= points_.size();
    profile_->error_cache_live_payload_bytes -= payload_bytes_;
}

const CachedMatrix *PointCache::find_hamiltonian(
    const CachedPoint &point
) const {
    const auto found = points_.find(point);
    if (found == points_.end() || !found->second.hamiltonian.has_value()) {
        return nullptr;
    }
    return &*found->second.hamiltonian;
}

const CachedSpectrum *PointCache::find_spectrum(
    const CachedPoint &point
) const {
    const auto found = points_.find(point);
    if (found == points_.end() || !found->second.spectrum.has_value()) {
        return nullptr;
    }
    return &*found->second.spectrum;
}

const std::vector<double> *PointCache::find_eigenvalues(
    const CachedPoint &point
) const {
    const auto *spectrum = find_spectrum(point);
    return spectrum == nullptr ? nullptr : &spectrum->eigenvalues;
}

const CachedMatrix &PointCache::store_hamiltonian(
    const CachedPoint &point,
    CachedMatrix value
) {
    auto &data = ensure(point);
    const auto before = payload_bytes(data);
    data.hamiltonian = std::move(value);
    update_payload(before, data);
    return *data.hamiltonian;
}

const std::vector<double> &PointCache::store_eigenvalues(
    const CachedPoint &point,
    std::vector<double> values
) {
    auto &data = ensure(point);
    if (data.spectrum.has_value() && data.spectrum->has_vectors()) {
        return data.spectrum->eigenvalues;
    }
    const auto before = payload_bytes(data);
    data.spectrum = CachedSpectrum{
        .eigenvalues = std::move(values),
        .eigenvectors = std::nullopt,
    };
    update_payload(before, data);
    return data.spectrum->eigenvalues;
}

const CachedSpectrum &PointCache::store_spectrum(
    const CachedPoint &point,
    CachedSpectrum value
) {
    auto &data = ensure(point);
    if (
        data.spectrum.has_value() && data.spectrum->has_vectors() &&
        !value.has_vectors()
    ) {
        return *data.spectrum;
    }
    const auto before = payload_bytes(data);
    data.spectrum = std::move(value);
    update_payload(before, data);
    return *data.spectrum;
}

std::size_t PointCache::payload_bytes(const PointData &data) {
    auto result = std::size_t{0};
    if (data.hamiltonian.has_value()) {
        result += data.hamiltonian->capacity() * sizeof(Complex);
    }
    if (data.spectrum.has_value()) {
        result += data.spectrum->eigenvalues.capacity() * sizeof(double);
        if (data.spectrum->eigenvectors.has_value()) {
            result +=
                data.spectrum->eigenvectors->capacity() * sizeof(Complex);
        }
    }
    return result;
}

PointCache::PointData &PointCache::ensure(const CachedPoint &point) {
    const auto [entry, inserted] = points_.try_emplace(point);
    if (inserted && profile_ != nullptr) {
        ++profile_->error_cache_live_records;
        profile_->error_cache_peak_records = std::max(
            profile_->error_cache_peak_records,
            profile_->error_cache_live_records
        );
    }
    return entry->second;
}

void PointCache::update_payload(
    std::size_t before,
    const PointData &data
) {
    const auto after = payload_bytes(data);
    payload_bytes_ += after;
    payload_bytes_ -= before;
    if (profile_ == nullptr) {
        return;
    }
    profile_->error_cache_live_payload_bytes += after;
    profile_->error_cache_live_payload_bytes -= before;
    profile_->error_cache_peak_payload_bytes = std::max(
        profile_->error_cache_peak_payload_bytes,
        profile_->error_cache_live_payload_bytes
    );
}

ModelBackend::ModelBackend(
    SpectralMesh &mesh,
    double mu,
    ChargeErrorStats &stats,
    ChargeProfile *profile
) : mesh_(mesh),
    mu_(mu),
    stats_(stats),
    profile_(profile),
    tight_binding_(dynamic_cast<const TightBindingModel *>(&mesh.model())),
    hopping_gram_(make_hopping_gram(tight_binding_)) {}

bool ModelBackend::supports_frobenius_defect() const noexcept {
    return tight_binding_ != nullptr;
}

double ModelBackend::frobenius_defect(
    const Point &first,
    const Point &second,
    const Point &midpoint
) const {
    const auto terms =
        core_detail::TightBindingModelAccess::hoppings(*tight_binding_);
    const auto term_count = terms.size();
    const auto coefficients = defect_coefficients(first, second, midpoint);
    auto squared_norm = Complex{0.0, 0.0};
    for (std::size_t right = 0; right < term_count; ++right) {
        for (std::size_t left = 0; left < term_count; ++left) {
            squared_norm +=
                std::conj(coefficients[left]) *
                hopping_gram_[matrix_index(left, right, term_count)] *
                coefficients[right];
        }
    }
    return std::sqrt(std::max(0.0, squared_norm.real()));
}

CachedMatrix ModelBackend::make_hopping_gram(
    const TightBindingModel *model
) {
    if (model == nullptr) {
        return {};
    }
    const auto terms = core_detail::TightBindingModelAccess::hoppings(*model);
    const auto term_count = terms.size();
    auto result = Matrix(term_count * term_count);
    for (std::size_t right = 0; right < term_count; ++right) {
        for (std::size_t left = 0; left < term_count; ++left) {
            result[matrix_index(left, right, term_count)] = conjugate_dot(
                terms[left].matrix, terms[right].matrix
            );
        }
    }
    return result;
}

CachedMatrix ModelBackend::defect_coefficients(
    const Point &first,
    const Point &second,
    const Point &midpoint
) const {
    const auto terms =
        core_detail::TightBindingModelAccess::hoppings(*tight_binding_);
    const auto first_point = first.to_point();
    const auto second_point = second.to_point();
    const auto midpoint_point = midpoint.to_point();
    auto coefficients = Matrix(terms.size());
    for (std::size_t term = 0; term < terms.size(); ++term) {
        const auto phase = [&](std::span<const double> point) {
            auto argument = 0.0;
            for (std::size_t axis = 0; axis < tight_binding_->ndim(); ++axis) {
                argument +=
                    2.0 * std::numbers::pi_v<double> * point[axis] *
                    static_cast<double>(terms[term].lattice_vector[axis]);
            }
            return std::exp(Complex{0.0, -argument});
        };
        coefficients[term] = phase(midpoint_point) -
            0.5 * (phase(first_point) + phase(second_point));
    }
    return coefficients;
}

RootErrorWorkspace::RootErrorWorkspace(ModelBackend &backend)
    : backend_(backend), points_(backend.profile()) {
    if (profile() != nullptr) {
        ++profile()->error_root_caches_created;
    }
}

RootErrorWorkspace::~RootErrorWorkspace() {
    if (profile() != nullptr) {
        ++profile()->error_root_caches_destroyed;
    }
}

void RootErrorWorkspace::remember_spectrum(
    const Point &point,
    const Eigensystem &spectrum
) {
    const auto *cached = points_.find_spectrum(point);
    if (cached != nullptr && cached->has_vectors()) {
        return;
    }
    auto shifted = CachedSpectrum{
        .eigenvalues = spectrum.eigenvalues,
        .eigenvectors = spectrum.eigenvectors,
    };
    for (auto &value : shifted.eigenvalues) {
        value -= mu();
    }
    points_.store_spectrum(point, std::move(shifted));
}

const CachedMatrix &RootErrorWorkspace::matrix(const Point &point) {
    if (profile() != nullptr) {
        ++profile()->error_matrix_requests;
    }
    if (const auto *cached = points_.find_hamiltonian(point)) {
        if (profile() != nullptr) {
            ++profile()->error_matrix_hits;
        }
        return *cached;
    }

    auto value = Matrix{};
    {
        auto timer = ProfileTimer(
            profile(),
            &ChargeProfile::error_hamiltonian_seconds,
            &ChargeProfile::error_hamiltonian_scalar_seconds
        );
        value = mesh().hamiltonian(point.to_point());
    }
    if (profile() != nullptr) {
        ++profile()->error_hamiltonian_scalar_calls;
    }
    for (std::size_t index = 0; index < mesh().ndof(); ++index) {
        value[matrix_index(index, index, mesh().ndof())] -= mu();
    }
    ++stats().hamiltonian_evaluations;
    return points_.store_hamiltonian(point, std::move(value));
}

const CachedSpectrum &RootErrorWorkspace::spectrum(const Point &point) {
    if (profile() != nullptr) {
        ++profile()->error_spectrum_requests;
    }
    const auto *cached = points_.find_spectrum(point);
    if (cached != nullptr && cached->has_vectors()) {
        if (profile() != nullptr) {
            ++profile()->error_spectrum_hits;
        }
        return *cached;
    }
    const auto upgrades_values_only = cached != nullptr;
    const auto &value = matrix(point);
    auto result = CachedSpectrum{};
    {
        auto timer = ProfileTimer(
            profile(),
            &ChargeProfile::error_full_eigensystem_seconds,
            &ChargeProfile::error_full_vector_seconds
        );
        result = diagonalize(
            value,
            mesh().ndof(),
            true,
            "charge-error full Hamiltonian"
        );
    }
    if (profile() != nullptr) {
        ++profile()->error_full_vector_calls;
        if (upgrades_values_only) {
            ++profile()->error_values_to_vectors_upgrades;
        }
    }
    ++stats().full_eigensystems;
    return points_.store_spectrum(point, std::move(result));
}

const std::vector<double> &RootErrorWorkspace::eigenvalues(
    const Point &point
) {
    if (profile() != nullptr) {
        ++profile()->error_eigenvalue_requests;
    }
    if (const auto *cached = points_.find_eigenvalues(point)) {
        if (profile() != nullptr) {
            ++profile()->error_eigenvalue_hits;
        }
        return *cached;
    }
    const auto &value = matrix(point);
    auto result = CachedSpectrum{};
    {
        auto timer = ProfileTimer(
            profile(),
            &ChargeProfile::error_full_eigensystem_seconds,
            &ChargeProfile::error_full_values_seconds
        );
        result = diagonalize(
            value,
            mesh().ndof(),
            false,
            "charge-error full Hamiltonian eigenvalues"
        );
    }
    if (profile() != nullptr) {
        ++profile()->error_full_values_calls;
    }
    ++stats().full_eigensystems;
    return points_.store_eigenvalues(
        point, std::move(result.eigenvalues)
    );
}

bool RootErrorWorkspace::supports_frobenius_defect() const noexcept {
    return backend_.supports_frobenius_defect();
}

double RootErrorWorkspace::frobenius_defect(
    const Point &first,
    const Point &second,
    const Point &midpoint
) const {
    return backend_.frobenius_defect(first, second, midpoint);
}

EffectiveModel::EffectiveModel(RootErrorWorkspace &workspace)
    : workspace_(workspace),
      evaluation_(BaseEvaluation{workspace}),
      points_(workspace.profile()),
      dimension_(workspace.mesh().ndof()) {}

EffectiveModel::EffectiveModel(
    RootErrorWorkspace &workspace,
    HamiltonianEvaluation evaluation,
    std::size_t dimension
) : workspace_(workspace),
    evaluation_(std::move(evaluation)),
    points_(workspace.profile()),
    dimension_(dimension) {}

bool EffectiveModel::is_base() const noexcept {
    return std::holds_alternative<BaseEvaluation>(evaluation_);
}

bool EffectiveModel::supports_frobenius_defect() const noexcept {
    return is_base() && workspace_.supports_frobenius_defect();
}

double EffectiveModel::frobenius_defect(
    const Point &first,
    const Point &second,
    const Point &midpoint
) const {
    return workspace_.frobenius_defect(first, second, midpoint);
}

const CachedMatrix &EffectiveModel::matrix(const Point &point) const {
    if (const auto *base = std::get_if<BaseEvaluation>(&evaluation_)) {
        return base->workspace.get().matrix(point);
    }
    auto *profile = workspace_.profile();
    if (profile != nullptr) {
        ++profile->error_effective_matrix_requests;
    }
    if (const auto *cached = points_.find_hamiltonian(point)) {
        if (profile != nullptr) {
            ++profile->error_effective_matrix_hits;
        }
        return *cached;
    }

    auto value = Matrix{};
    if (const auto *dense = std::get_if<DenseSchurEvaluation>(&evaluation_)) {
        // Parent/base evaluation has its own exclusive timer. Only the Schur
        // algebra is attributed to this region.
        const auto &parent_matrix = dense->parent.get().matrix(point);
        {
            auto timer = ProfileTimer(
                profile,
                &ChargeProfile::error_schur_application_seconds,
                &ChargeProfile::error_general_schur_application_seconds
            );
            value = apply_schur_layer(
                parent_matrix, dense->layer, workspace_.stats()
            );
        }
    } else {
        const auto &projected =
            std::get<ProjectedTightBindingEvaluation>(evaluation_);
        {
            auto timer = ProfileTimer(
                profile,
                &ChargeProfile::error_schur_application_seconds,
                &ChargeProfile::error_projected_schur_application_seconds
            );
            value = projected.model.evaluate(point.to_point());
        }
        ++workspace_.stats().schur_evaluations;
        if (profile != nullptr) {
            ++profile->error_projected_schur_evaluations;
        }
    }
    return points_.store_hamiltonian(point, std::move(value));
}

const CachedSpectrum &EffectiveModel::spectrum(const Point &point) const {
    if (const auto *base = std::get_if<BaseEvaluation>(&evaluation_)) {
        return base->workspace.get().spectrum(point);
    }
    auto *profile = workspace_.profile();
    if (profile != nullptr) {
        ++profile->error_effective_spectrum_requests;
    }
    const auto *cached = points_.find_spectrum(point);
    if (cached != nullptr && cached->has_vectors()) {
        if (profile != nullptr) {
            ++profile->error_effective_spectrum_hits;
        }
        return *cached;
    }
    const auto &value = matrix(point);
    const auto started = profile == nullptr
        ? Clock::time_point{}
        : Clock::now();
    auto result = dimension_ == 1
        ? CachedSpectrum{
            .eigenvalues = {value.front().real()},
            .eigenvectors = Matrix{Complex{1.0, 0.0}},
        }
        : diagonalize(
            value,
            dimension_,
            true,
            "charge-error reduced Hamiltonian"
        );
    if (profile != nullptr) {
        profile->error_reduced_eigensystem_seconds +=
            elapsed_seconds(started);
        profile->record_reduced_dimension(dimension_);
    }
    ++workspace_.stats().reduced_eigensystems;
    return points_.store_spectrum(point, std::move(result));
}

const std::vector<double> &EffectiveModel::eigenvalues(
    const Point &point
) const {
    if (const auto *base = std::get_if<BaseEvaluation>(&evaluation_)) {
        return base->workspace.get().eigenvalues(point);
    }
    // Reduced certification is expected to need vectors at the same points,
    // so the baseline policy deliberately obtains the full reduced spectrum.
    return spectrum(point).eigenvalues;
}

std::unique_ptr<EffectiveModel> make_active_space_model(
    const EffectiveModel &parent,
    const CachedSpectrum &anchor,
    std::size_t active_begin,
    std::size_t active_end
) {
    auto &workspace = parent.workspace_;
    auto &backend = workspace.backend();
    auto *profile = workspace.profile();
    const auto started = profile == nullptr
        ? Clock::time_point{}
        : Clock::now();
    auto uses_projected = false;
    auto result = std::unique_ptr<EffectiveModel>{};

    if (parent.is_base() && backend.tight_binding_ != nullptr) {
        auto layer = make_schur_layer(
            schur_eigensystem_view(anchor),
            active_begin,
            active_end,
            false
        );
        auto projected = make_projected_hopping_model(
            *backend.tight_binding_, backend.mu(), layer
        );
        if (projected.has_value()) {
            uses_projected = true;
            const auto dimension = projected->dimension;
            result.reset(new EffectiveModel(
                workspace,
                ProjectedTightBindingEvaluation{std::move(*projected)},
                dimension
            ));
        } else {
            layer = make_schur_layer(
                schur_eigensystem_view(anchor),
                active_begin,
                active_end,
                true
            );
            const auto dimension = layer.active_dimension;
            result.reset(new EffectiveModel(
                workspace,
                DenseSchurEvaluation{parent, std::move(layer)},
                dimension
            ));
        }
    } else {
        auto layer = make_schur_layer(
            schur_eigensystem_view(anchor),
            active_begin,
            active_end,
            true
        );
        const auto dimension = layer.active_dimension;
        result.reset(new EffectiveModel(
            workspace,
            DenseSchurEvaluation{parent, std::move(layer)},
            dimension
        ));
    }

    if (profile != nullptr) {
        const auto seconds = elapsed_seconds(started);
        profile->error_schur_setup_seconds += seconds;
        if (uses_projected) {
            profile->error_projected_schur_setup_seconds += seconds;
            ++profile->error_projected_schur_setup_calls;
        } else {
            profile->error_general_schur_setup_seconds += seconds;
            ++profile->error_general_schur_setup_calls;
        }
    }
    return result;
}

}  // namespace fermisimplex::integration_detail
