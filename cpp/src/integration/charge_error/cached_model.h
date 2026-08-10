#pragma once

#include "integration/charge_error/projected_schur.h"

#include <fermisimplex/integration.h>
#include <fermisimplex/spectral_mesh.h>

#include <adaptivesimplex/core/dyadic_vertex.h>

#include <complex>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <unordered_map>
#include <variant>
#include <vector>

namespace fermisimplex::integration_detail {

struct ChargeProfile;

using CachedMatrix = std::vector<std::complex<double>>;
using CachedPoint = adaptivesimplex::core::DyadicVertex;

// A cached solve explicitly records whether eigenvectors were requested.
// Values-only entries may later be replaced by a full eigensystem.
struct CachedSpectrum {
    std::vector<double> eigenvalues;
    std::optional<CachedMatrix> eigenvectors;

    bool has_vectors() const noexcept { return eigenvectors.has_value(); }
};

SchurEigensystemView schur_eigensystem_view(
    const CachedSpectrum &spectrum
);

// Exact dyadic points map to one lazy record. The record representation is
// deliberately private: clients request and store typed payloads instead of
// coordinating nullable fields themselves.
class PointCache {
public:
    explicit PointCache(ChargeProfile *profile = nullptr);
    ~PointCache();

    PointCache(const PointCache &) = delete;
    PointCache &operator=(const PointCache &) = delete;
    PointCache(PointCache &&) = delete;
    PointCache &operator=(PointCache &&) = delete;

    const CachedMatrix *find_hamiltonian(const CachedPoint &point) const;
    const CachedSpectrum *find_spectrum(const CachedPoint &point) const;
    const std::vector<double> *find_eigenvalues(
        const CachedPoint &point
    ) const;

    const CachedMatrix &store_hamiltonian(
        const CachedPoint &point,
        CachedMatrix value
    );
    const std::vector<double> &store_eigenvalues(
        const CachedPoint &point,
        std::vector<double> values
    );
    const CachedSpectrum &store_spectrum(
        const CachedPoint &point,
        CachedSpectrum value
    );

    std::size_t size() const noexcept { return points_.size(); }
    std::size_t payload_bytes() const noexcept { return payload_bytes_; }

private:
    struct PointData {
        std::optional<CachedMatrix> hamiltonian;
        std::optional<CachedSpectrum> spectrum;
    };

    static std::size_t payload_bytes(const PointData &data);
    PointData &ensure(const CachedPoint &point);
    void update_payload(std::size_t before, const PointData &data);

    ChargeProfile *profile_ = nullptr;
    std::unordered_map<CachedPoint, PointData, CachedPoint::Hash> points_;
    std::size_t payload_bytes_ = 0;
};

// Long-lived, immutable model data shared by all root estimates. Point data
// is intentionally absent: it belongs to RootErrorWorkspace below.
class ModelBackend {
public:
    ModelBackend(
        SpectralMesh &mesh,
        double mu,
        ChargeErrorStats &stats,
        ChargeProfile *profile
    );

    bool supports_frobenius_defect() const noexcept;
    double frobenius_defect(
        const CachedPoint &first,
        const CachedPoint &second,
        const CachedPoint &midpoint
    ) const;

    SpectralMesh &mesh() const noexcept { return mesh_; }
    double mu() const noexcept { return mu_; }
    ChargeErrorStats &stats() const noexcept { return stats_; }
    ChargeProfile *profile() const noexcept { return profile_; }

private:
    friend class EffectiveModel;
    friend std::unique_ptr<class EffectiveModel> make_active_space_model(
        const class EffectiveModel &parent,
        const CachedSpectrum &anchor,
        std::size_t active_begin,
        std::size_t active_end
    );

    static CachedMatrix make_hopping_gram(const TightBindingModel *model);
    CachedMatrix defect_coefficients(
        const CachedPoint &first,
        const CachedPoint &second,
        const CachedPoint &midpoint
    ) const;

    SpectralMesh &mesh_;
    double mu_ = 0.0;
    ChargeErrorStats &stats_;
    ChargeProfile *profile_ = nullptr;
    const TightBindingModel *tight_binding_ = nullptr;
    const CachedMatrix hopping_gram_;
};

// Full-dimensional point values for exactly one outer-simplex error estimate.
// Construction/destruction is instrumented so tests can verify the lifetime.
class RootErrorWorkspace {
public:
    explicit RootErrorWorkspace(ModelBackend &backend);
    ~RootErrorWorkspace();

    RootErrorWorkspace(const RootErrorWorkspace &) = delete;
    RootErrorWorkspace &operator=(const RootErrorWorkspace &) = delete;

    void remember_spectrum(
        const CachedPoint &point,
        const Eigensystem &spectrum
    );
    const CachedMatrix &matrix(const CachedPoint &point);
    const CachedSpectrum &spectrum(const CachedPoint &point);
    const std::vector<double> &eigenvalues(const CachedPoint &point);

    SpectralMesh &mesh() const noexcept { return backend_.mesh(); }
    double mu() const noexcept { return backend_.mu(); }
    ChargeErrorStats &stats() const noexcept { return backend_.stats(); }
    ChargeProfile *profile() const noexcept { return backend_.profile(); }
    ModelBackend &backend() const noexcept { return backend_; }

    bool supports_frobenius_defect() const noexcept;
    double frobenius_defect(
        const CachedPoint &first,
        const CachedPoint &second,
        const CachedPoint &midpoint
    ) const;

    const PointCache &point_cache() const noexcept { return points_; }

private:
    ModelBackend &backend_;
    PointCache points_;
};

class EffectiveModel;

struct BaseEvaluation {
    std::reference_wrapper<RootErrorWorkspace> workspace;
};

struct DenseSchurEvaluation {
    std::reference_wrapper<const EffectiveModel> parent;
    SchurLayer layer;
};

struct ProjectedTightBindingEvaluation {
    ProjectedHoppingModel model;
};

using HamiltonianEvaluation = std::variant<
    BaseEvaluation,
    DenseSchurEvaluation,
    ProjectedTightBindingEvaluation
>;

// One model owns one reduced-space point cache. The evaluation variant makes
// the valid state explicit: base, dense Schur, or projected tight binding.
class EffectiveModel {
public:
    explicit EffectiveModel(RootErrorWorkspace &workspace);

    EffectiveModel(const EffectiveModel &) = delete;
    EffectiveModel &operator=(const EffectiveModel &) = delete;
    EffectiveModel(EffectiveModel &&) = delete;
    EffectiveModel &operator=(EffectiveModel &&) = delete;

    std::size_t dimension() const noexcept { return dimension_; }
    bool is_base() const noexcept;
    bool supports_frobenius_defect() const noexcept;
    double frobenius_defect(
        const CachedPoint &first,
        const CachedPoint &second,
        const CachedPoint &midpoint
    ) const;

    const CachedMatrix &matrix(const CachedPoint &point) const;
    const CachedSpectrum &spectrum(const CachedPoint &point) const;
    const std::vector<double> &eigenvalues(const CachedPoint &point) const;

private:
    friend std::unique_ptr<EffectiveModel> make_active_space_model(
        const EffectiveModel &parent,
        const CachedSpectrum &anchor,
        std::size_t active_begin,
        std::size_t active_end
    );

    EffectiveModel(
        RootErrorWorkspace &workspace,
        HamiltonianEvaluation evaluation,
        std::size_t dimension
    );

    RootErrorWorkspace &workspace_;
    HamiltonianEvaluation evaluation_;
    mutable PointCache points_;
    std::size_t dimension_ = 0;
};

// Chooses the only optimized root reduction in one place. Reductions below an
// already reduced model always use dense Schur evaluation.
std::unique_ptr<EffectiveModel> make_active_space_model(
    const EffectiveModel &parent,
    const CachedSpectrum &anchor,
    std::size_t active_begin,
    std::size_t active_end
);

}  // namespace fermisimplex::integration_detail
