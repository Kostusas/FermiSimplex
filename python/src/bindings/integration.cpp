#include "arrays.h"
#include "bindings.h"
#include "python_hamiltonian.h"

#include "integration/charge.h"
#include "integration/charge_certificate_cache.h"
#include "integration/density.h"
#include "integration/workspace.h"

#include <adaptivesimplex/adaptive/adaptive_loop.h>
#include <adaptivesimplex/adaptive/evaluation.h>
#include <adaptivesimplex/adaptive/simplex_integrand.h>
#include <adaptivesimplex/adaptive/types.h>

#include <nanobind/stl/shared_ptr.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace lineartetrahedron::bindings {

namespace adaptive = adaptivesimplex::adaptive;
namespace core = adaptivesimplex::core;

namespace {

struct DensityStoppingError {
    template <class IntegralValue> using state_type = IntegralValue;

    template <class IntegralValue> state_type<IntegralValue> zero() const {
        return {};
    }
    template <class IntegralValue>
    state_type<IntegralValue> contribution(
        const adaptive::SimplexEstimate<IntegralValue> &estimate
    ) const {
        return estimate.correction;
    }
    template <class IntegralValue>
    void add(state_type<IntegralValue> &state, const state_type<IntegralValue> &value) const {
        state += value;
    }
    template <class IntegralValue>
    void remove(state_type<IntegralValue> &state, const state_type<IntegralValue> &value) const {
        state -= value;
    }
    template <class IntegralValue> double error(const state_type<IntegralValue> &state) const {
        return state.max_abs();
    }
};

struct DensityRefinementScore {
    template <class Context> double operator()(const Context &context) const {
        return context.correction.max_abs();
    }
};

struct ChargeStoppingError {
    template <class IntegralValue> using state_type = IntegralValue;

    template <class IntegralValue> state_type<IntegralValue> zero() const {
        return {};
    }
    template <class IntegralValue>
    state_type<IntegralValue> contribution(
        const adaptive::SimplexEstimate<IntegralValue> &estimate
    ) const {
        auto contribution = IntegralValue{};
        contribution.certificate_error = estimate.preview.certificate_error;
        return contribution;
    }
    template <class IntegralValue>
    void add(state_type<IntegralValue> &state, const state_type<IntegralValue> &value) const {
        state += value;
    }
    template <class IntegralValue>
    void remove(state_type<IntegralValue> &state, const state_type<IntegralValue> &value) const {
        state -= value;
    }
    template <class IntegralValue>
    double error(const state_type<IntegralValue> &correction_state) const {
        return correction_state.certificate_error;
    }
};

struct ChargeRefinementScore {
    template <class Context> double operator()(const Context &context) const {
        return context.preview.certificate_error;
    }
};

struct ChargeEstimateContext {
    const ChargeValue &preview;
    const ChargeValue &correction;
};

class ChargeIntegrand {
public:
    using vertex_value_type = VertexSpectra;
    using integral_value_type = ChargeValue;
    using cache_type = core::VertexCache<VertexSpectra>;

    ChargeIntegrand(
        IntegrationWorkspace &workspace,
        double mu,
        bool certify_preview,
        ChargeCertificateCache &certificate_cache
    ) : workspace_(workspace),
        cache_(workspace.cache()),
        mu_(mu),
        certify_preview_(certify_preview),
        certificate_cache_(certificate_cache) {}

    cache_type &cache() {
        return cache_;
    }

    auto stopping_global_error() const {
        return ChargeStoppingError{};
    }

    vertex_value_type evaluate_vertex(core::Geometry &geometry, core::VertexId vertex_id) const {
        const auto point = geometry.vertices().dyadic_vertex(vertex_id).to_point();
        return workspace_.evaluate_vertex(std::span<const double>(point.data(), point.size()));
    }

    std::vector<adaptive::SimplexEstimate<integral_value_type>> estimate_simplices(
        core::Geometry &geometry,
        std::span<const core::SimplexId> simplex_ids,
        std::uint32_t /*preview_depth*/
    ) const {
        std::vector<adaptive::SimplexEstimate<integral_value_type>> estimates;
        estimates.reserve(simplex_ids.size());
        for (const auto simplex_id : simplex_ids) {
            auto coarse = charge_on_simplex_with_energy_bound(
                mu_,
                workspace_,
                geometry,
                simplex_id,
                cache_,
                false,
                nullptr,
                0.0
            );

            auto preview = integral_value_type{};
            const auto preview_ids = geometry.preview_active(simplex_id, 1);
            for (const auto preview_id : preview_ids) {
                preview += charge_on_simplex_with_energy_bound(
                    mu_,
                    workspace_,
                    geometry,
                    preview_id,
                    cache_,
                    certify_preview_,
                    certify_preview_ ? &certificate_cache_ : nullptr,
                    0.0
                );
            }

            auto correction = preview;
            correction -= coarse;
            const auto context = ChargeEstimateContext{
                .preview = preview,
                .correction = correction,
            };
            const auto refinement_score =
                static_cast<double>(ChargeRefinementScore{}(context));

            estimates.push_back(adaptive::SimplexEstimate<integral_value_type>{
                .coarse = std::move(coarse),
                .preview = std::move(preview),
                .correction = std::move(correction),
                .refinement_score = refinement_score,
            });
        }
        return estimates;
    }

private:
    IntegrationWorkspace &workspace_;
    cache_type &cache_;
    double mu_ = 0.0;
    bool certify_preview_ = false;
    ChargeCertificateCache &certificate_cache_;
};

class PythonIntegrationRuntime {
public:
    PythonIntegrationRuntime(
        std::shared_ptr<const HamiltonianModel> model,
        double tol
    ) : workspace_(std::move(model), tol) {}

    size_t ndim() const noexcept { return workspace_.ndim(); }
    size_t ndof() const noexcept { return workspace_.ndof(); }
    size_t n_cached_nodes() const noexcept { return workspace_.n_cached_nodes(); }
    std::int64_t n_active_simplices() const noexcept {
        return workspace_.n_active_simplices();
    }
    std::int64_t n_active_vertices() const { return workspace_.n_active_vertices(); }

    ChargeIntegrateResult integrate_charge(
        double mu,
        const adaptive::Options &options,
        bool refine,
        bool certify,
        nb::object hessian_bound,
        double anharmonicity_bound
    ) {
        (void)hessian_bound;
        (void)anharmonicity_bound;
        auto charge_options = options;
        charge_options.preview_depth = 1;
        auto integrand = ChargeIntegrand(
            workspace_,
            mu,
            refine ? true : certify,
            charge_certificate_cache_
        );

        if (refine) {
            const auto raw = adaptive::run(workspace_.geometry(), integrand, charge_options);
            return ChargeIntegrateResult{
                .charge = raw.integral.charge,
                .charge_error = raw.stopping_error,
                .dcharge_dmu = raw.integral.derivative,
                .visible_cut_error = raw.integral.visible_cut_error,
                .visible_cut_inconclusive_style_error =
                    raw.integral.visible_cut_inconclusive_style_error,
                .visible_cut_count = raw.integral.visible_cut_count,
                .inconclusive_error = raw.integral.inconclusive_error,
                .inconclusive_count = raw.integral.inconclusive_count,
                .work = raw.evaluations,
                .refinements = raw.refinements,
                .n_active_simplices = n_active_simplices(),
                .n_active_vertices = n_active_vertices(),
                .converged = raw.converged,
            };
        }

        const auto preview_depth = std::uint32_t{1};
        const auto active = workspace_.geometry().simplices().active_simplices();
        const auto active_simplices = std::vector<core::SimplexId>(active.begin(), active.end());

        ChargeValue value;
        auto stopping_global_error = ChargeStoppingError{};
        auto stopping_state = stopping_global_error.template zero<ChargeValue>();
        std::int64_t evaluations = 0;
        for (const auto &estimate :
             adaptive::estimate_simplices(
                 workspace_.geometry(),
                 integrand,
                 active_simplices,
                 preview_depth,
                 evaluations
             )) {
            value += estimate.preview;
            auto stopping_contribution =
                stopping_global_error.template contribution<ChargeValue>(estimate);
            stopping_global_error.template add<ChargeValue>(stopping_state, stopping_contribution);
        }
        const auto charge_error =
            stopping_global_error.template error<ChargeValue>(stopping_state);

        return ChargeIntegrateResult{
            .charge = value.charge,
            .charge_error = charge_error,
            .dcharge_dmu = value.derivative,
            .visible_cut_error = value.visible_cut_error,
            .visible_cut_inconclusive_style_error =
                value.visible_cut_inconclusive_style_error,
            .visible_cut_count = value.visible_cut_count,
            .inconclusive_error = value.inconclusive_error,
            .inconclusive_count = value.inconclusive_count,
            .work = evaluations,
            .refinements = 0,
            .n_active_simplices = n_active_simplices(),
            .n_active_vertices = n_active_vertices(),
            .converged = charge_error <= options.target_error,
        };
    }

    DensityIntegrateResult integrate_density(
        double mu,
        const adaptive::Options &options,
        std::vector<std::int64_t> keys,
        std::vector<std::int64_t> component_rows,
        std::vector<std::int64_t> component_cols,
        std::vector<std::int64_t> component_key_indices,
        bool refine
    ) {
        auto density = DensityComponents(
            workspace_.ndim(),
            workspace_.ndof(),
            std::move(keys),
            std::move(component_rows),
            std::move(component_cols),
            std::move(component_key_indices)
        );
        auto integrand = adaptive::simplex_integrand(
            workspace_.cache(),
            [this](std::span<const double> point) {
                return workspace_.evaluate_vertex(point);
            },
            [this, mu, &density](
                const core::Geometry &geometry,
                core::SimplexId simplex_id,
                const core::VertexCache<VertexSpectra> &cache
            ) {
                return density.on_simplex(mu, workspace_, geometry, simplex_id, cache);
            },
            adaptive::estimation_policies<DensityStoppingError, DensityRefinementScore>{}
        );
        auto effective_options = options;
        if (!refine) {
            effective_options.max_refinements = 0;
        }
        const auto raw = adaptive::run(workspace_.geometry(), integrand, effective_options);

        DensityIntegrateResult result;
        result.estimate = std::vector<std::complex<double>>(raw.integral.values());
        result.error_vector.resize(raw.correction.size());
        for (size_t index = 0; index < raw.correction.size(); ++index) {
            result.error_vector[index] = std::abs(raw.correction[index]);
        }
        result.error_scalar = raw.stopping_error;
        result.work = raw.evaluations;
        result.refinements = raw.refinements;
        result.n_active_simplices = n_active_simplices();
        result.n_active_vertices = n_active_vertices();
        result.converged = raw.converged;
        return result;
    }

private:
    IntegrationWorkspace workspace_;
    ChargeCertificateCache charge_certificate_cache_;
};

}  // namespace

void bind_integration(nb::module_ &m) {
    using namespace nb::literals;

    nb::class_<ChargeIntegrateResult>(m, "ChargeIntegrateResult")
        .def_prop_ro("charge", [](const ChargeIntegrateResult &self) { return self.charge; })
        .def_prop_ro("charge_error", [](const ChargeIntegrateResult &self) { return self.charge_error; })
        .def_prop_ro("dcharge_dmu", [](const ChargeIntegrateResult &self) { return self.dcharge_dmu; })
        .def_prop_ro("visible_cut_error", [](const ChargeIntegrateResult &self) { return self.visible_cut_error; })
        .def_prop_ro("visible_cut_inconclusive_style_error", [](const ChargeIntegrateResult &self) { return self.visible_cut_inconclusive_style_error; })
        .def_prop_ro("visible_cut_count", [](const ChargeIntegrateResult &self) { return self.visible_cut_count; })
        .def_prop_ro("inconclusive_error", [](const ChargeIntegrateResult &self) { return self.inconclusive_error; })
        .def_prop_ro("inconclusive_count", [](const ChargeIntegrateResult &self) { return self.inconclusive_count; })
        .def_prop_ro("work", [](const ChargeIntegrateResult &self) { return self.work; })
        .def_prop_ro("refinements", [](const ChargeIntegrateResult &self) { return self.refinements; })
        .def_prop_ro("n_active_simplices", [](const ChargeIntegrateResult &self) { return self.n_active_simplices; })
        .def_prop_ro("n_active_vertices", [](const ChargeIntegrateResult &self) { return self.n_active_vertices; })
        .def_prop_ro("converged", [](const ChargeIntegrateResult &self) { return self.converged; });

    nb::class_<DensityIntegrateResult>(m, "DensityIntegrateResult")
        .def(
            "estimate_array",
            [](const DensityIntegrateResult &self) {
                return make_array(
                    std::vector<std::complex<double>>(self.estimate),
                    {self.estimate.size()}
                );
            }
        )
        .def(
            "error_vector_array",
            [](const DensityIntegrateResult &self) {
                return make_array(std::vector<double>(self.error_vector), {self.error_vector.size()});
            }
        )
        .def_prop_ro("error_scalar", [](const DensityIntegrateResult &self) { return self.error_scalar; })
        .def_prop_ro("work", [](const DensityIntegrateResult &self) { return self.work; })
        .def_prop_ro("refinements", [](const DensityIntegrateResult &self) { return self.refinements; })
        .def_prop_ro("n_active_simplices", [](const DensityIntegrateResult &self) { return self.n_active_simplices; })
        .def_prop_ro("n_active_vertices", [](const DensityIntegrateResult &self) { return self.n_active_vertices; })
        .def_prop_ro("converged", [](const DensityIntegrateResult &self) { return self.converged; });

    nb::class_<adaptive::Options>(m, "AdaptiveOptions")
        .def(
            "__init__",
            [](
                adaptive::Options *self,
                double target_error,
                std::int64_t max_refinements,
                std::uint32_t preview_depth,
                std::int64_t min_refinement_batch_size,
                std::int64_t max_refinement_batch_size
            ) {
                new (self) adaptive::Options{
                    .target_error = target_error,
                    .max_refinements = static_cast<std::int64_t>(max_refinements),
                    .preview_depth = static_cast<std::uint32_t>(preview_depth),
                    .min_refinement_batch_size =
                        static_cast<std::size_t>(min_refinement_batch_size),
                    .max_refinement_batch_size =
                        static_cast<std::size_t>(max_refinement_batch_size),
                };
            },
            "target_error"_a,
            "max_refinements"_a = -1,
            "preview_depth"_a = 1,
            "min_refinement_batch_size"_a = 1,
            "max_refinement_batch_size"_a = 100
        )
        .def_rw("target_error", &adaptive::Options::target_error)
        .def_rw("max_refinements", &adaptive::Options::max_refinements)
        .def_rw("preview_depth", &adaptive::Options::preview_depth)
        .def_rw("min_refinement_batch_size", &adaptive::Options::min_refinement_batch_size)
        .def_rw("max_refinement_batch_size", &adaptive::Options::max_refinement_batch_size);

    nb::class_<PythonIntegrationRuntime>(m, "IntegrationRuntime")
        .def(
            "__init__",
            [](
                PythonIntegrationRuntime *self,
                std::shared_ptr<TightBindingModel> model,
                double tol
            ) {
                new (self) PythonIntegrationRuntime(
                    std::static_pointer_cast<const HamiltonianModel>(std::move(model)),
                    tol
                );
            },
            "model"_a,
            "tol"_a = 1e-14
        )
        .def(
            "__init__",
            [](
                PythonIntegrationRuntime *self,
                nb::object callable,
                size_t ndim,
                size_t ndof,
                double tol
            ) {
                new (self) PythonIntegrationRuntime(
                    make_python_hamiltonian_model(std::move(callable), ndim, ndof),
                    tol
                );
            },
            "callable"_a,
            "ndim"_a,
            "ndof"_a,
            "tol"_a = 1e-14
        )
        .def_prop_ro("ndim", &PythonIntegrationRuntime::ndim)
        .def_prop_ro("ndof", &PythonIntegrationRuntime::ndof)
        .def_prop_ro("n_cached_nodes", &PythonIntegrationRuntime::n_cached_nodes)
        .def_prop_ro("n_active_simplices", &PythonIntegrationRuntime::n_active_simplices)
        .def_prop_ro("n_active_vertices", &PythonIntegrationRuntime::n_active_vertices)
        .def(
            "integrate_charge",
            &PythonIntegrationRuntime::integrate_charge,
            "mu"_a,
            "options"_a,
            "refine"_a = true,
            "certify"_a = true,
            "hessian_bound"_a = 0.0,
            "anharmonicity_bound"_a = 0.0
        )
        .def(
            "integrate_density",
            [](
                PythonIntegrationRuntime &self,
                double mu,
                const adaptive::Options &options,
                KeyArray keys,
                ComponentIndexArray component_rows,
                ComponentIndexArray component_cols,
                ComponentIndexArray component_key_indices,
                bool refine
            ) {
                return self.integrate_density(
                    mu,
                    options,
                    copy_keys(keys),
                    copy_1d(component_rows),
                    copy_1d(component_cols),
                    copy_1d(component_key_indices),
                    refine
                );
            },
            "mu"_a,
            "options"_a,
            "keys"_a,
            "component_rows"_a,
            "component_cols"_a,
            "component_key_indices"_a,
            "refine"_a = true
        );
}

}  // namespace lineartetrahedron::bindings
