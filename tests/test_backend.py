from __future__ import annotations

import numpy as np
import pytest

from lineartetrahedron import (
    AdaptiveOptions,
    ChargeEvaluator,
    build_runtime,
    density_matrix_at_mu_zero_temp,
    full_density_components,
    prepare_density_components,
    tb_to_tight_binding_model,
)

from .conftest import requires_native
from .helpers import dense_reference, dimerized_chain, max_density_error, qiwuzhang, tb_k_matrix


def _constant_insulator(ndim: int) -> dict[tuple[int, ...], np.ndarray]:
    return {(0,) * ndim: np.diag([-1.0, 1.0]).astype(complex)}


@requires_native
def test_compiled_tight_binding_model_matches_python_fourier_evaluation():
    tb = qiwuzhang()
    compiled_model = tb_to_tight_binding_model(tb)
    point = np.array([0.3, -0.7], dtype=float)

    assert compiled_model.ndim == 2
    assert compiled_model.ndof == 2
    assert compiled_model.nterms == len(tb)
    assert np.allclose(compiled_model.evaluate_point(point), tb_k_matrix(tb, point))


@requires_native
@pytest.mark.parametrize("ndim", (1, 2, 3))
def test_runtime_accepts_physical_dimensions(ndim):
    tb = _constant_insulator(ndim)
    key = (0,) * ndim
    rho, _error, info = density_matrix_at_mu_zero_temp(
        tb,
        mu=0.0,
        keys=[key],
        density_components=full_density_components([key], size=2),
        density_atol=1e-12,
        max_subdivisions=8,
    )

    assert np.allclose(rho[key], np.diag([1.0, 0.0]), atol=1e-12)
    assert info.n_cached_nodes > 0


@requires_native
def test_runtime_accepts_adaptive_options():
    tb = _constant_insulator(1)
    key = (0,)
    prepared = prepare_density_components(
        tb,
        [key],
        full_density_components([key], size=2),
    )
    runtime = build_runtime(
        tb,
        keys=list(prepared.keys),
        component_rows=prepared.rows,
        component_cols=prepared.cols,
        component_key_indices=prepared.key_indices,
    )
    options = AdaptiveOptions(
        target_error=1e-12,
        max_refinements=8,
        preview_depth=2,
        min_refinement_batch_size=1,
        max_refinement_batch_size=100,
    )

    charge = runtime.integrate_charge(0.0, options)
    fixed_charge = runtime.evaluate_charge(0.0, options)
    density = runtime.integrate_density(0.0, options)

    assert charge.charge == pytest.approx(1.0)
    assert fixed_charge.charge == pytest.approx(1.0)
    assert fixed_charge.refinements == 0
    assert np.allclose(density.estimate_array(), np.diag([1.0, 0.0]).reshape(-1))


@requires_native
def test_runtime_rejects_higher_dimensions():
    tb = _constant_insulator(4)
    key = (0, 0, 0, 0)

    with pytest.raises(ValueError, match="supports dimensions 1, 2, and 3"):
        density_matrix_at_mu_zero_temp(
            tb,
            mu=0.0,
            keys=[key],
            density_components=full_density_components([key], size=2),
            density_atol=1e-12,
            max_subdivisions=8,
        )


@requires_native
def test_selected_density_components_match_full_density_slice():
    tb = dimerized_chain()
    keys = [(0,), (1,), (-1,)]
    selected_components = [(0, 0, (0,)), (1, 1, (0,)), (0, 1, (1,))]
    full_rho, _full_error, _full_info = density_matrix_at_mu_zero_temp(
        tb,
        mu=0.0,
        keys=keys,
        density_components=full_density_components(keys, size=2),
        density_atol=1e-3,
        max_subdivisions=80,
    )
    selected_rho, _selected_error, _selected_info = density_matrix_at_mu_zero_temp(
        tb,
        mu=0.0,
        keys=keys,
        density_components=selected_components,
        density_atol=1e-3,
        max_subdivisions=80,
    )

    for row, col, key in selected_components:
        assert selected_rho[key][row, col] == pytest.approx(full_rho[key][row, col])


@requires_native
def test_fixed_mu_density_matches_dense_reference():
    tb = dimerized_chain()
    keys = [(0,), (1,), (-1,)]
    reference = dense_reference(tb, mu=0.0, keys=keys, nk=2001)
    rho, error, info = density_matrix_at_mu_zero_temp(
        tb,
        mu=0.0,
        keys=keys,
        density_components=full_density_components(keys, size=2),
        density_atol=5e-3,
        max_subdivisions=512,
    )

    assert max_density_error(rho, reference.rho) <= 5e-3
    assert max(float(np.max(np.abs(block))) for block in error.values()) <= 5e-3
    assert info.n_cached_nodes > 0


@requires_native
def test_charge_evaluator_does_not_change_active_simplex_count():
    tb = dimerized_chain()
    keys = [(0,), (1,), (-1,)]
    prepared = prepare_density_components(
        tb,
        keys,
        full_density_components(keys, size=2),
    )
    runtime = build_runtime(
        tb,
        keys=list(prepared.keys),
        component_rows=prepared.rows,
        component_cols=prepared.cols,
        component_key_indices=prepared.key_indices,
    )
    runtime.integrate_charge(
        0.0,
        AdaptiveOptions(target_error=2e-3, max_refinements=64),
    )
    evaluator = ChargeEvaluator(
        runtime,
        AdaptiveOptions(target_error=2e-3, max_refinements=0, preview_depth=2),
    )

    active_before = runtime.n_active_simplices
    values = [evaluator(mu) for mu in (-1.0, 0.0, 1.0, 0.25)]
    active_after = runtime.n_active_simplices

    assert active_after == active_before
    assert all(len(value) == 3 for value in values)
    assert all(np.isfinite(value[0]) for value in values)
    assert all(value[1] >= 0.0 for value in values)


@requires_native
def test_cached_density_initial_path_reuses_charge_vertices():
    tb = dimerized_chain()
    keys = [(0,), (1,), (-1,)]
    prepared = prepare_density_components(
        tb,
        keys,
        full_density_components(keys, size=2),
    )
    runtime = build_runtime(
        tb,
        keys=list(prepared.keys),
        component_rows=prepared.rows,
        component_cols=prepared.cols,
        component_key_indices=prepared.key_indices,
    )
    runtime.integrate_charge(
        0.0,
        AdaptiveOptions(target_error=2e-3, max_refinements=64, preview_depth=2),
    )
    cached_before = runtime.n_cached_nodes

    density = runtime.integrate_density(
        0.0,
        AdaptiveOptions(target_error=1.0, max_refinements=0, preview_depth=2),
    )

    assert density.converged
    assert density.work == 0
    assert density.refinements == 0
    assert runtime.n_cached_nodes == cached_before
    assert density.estimate_array().shape == (prepared.value_count,)
    assert density.error_vector_array().shape == (prepared.value_count,)
    assert density.error_scalar <= 1.0


@requires_native
def test_density_tight_target_refines_with_blas_estimator():
    tb = dimerized_chain()
    keys = [(0,), (1,), (-1,)]
    prepared = prepare_density_components(
        tb,
        keys,
        full_density_components(keys, size=2),
    )
    runtime = build_runtime(
        tb,
        keys=list(prepared.keys),
        component_rows=prepared.rows,
        component_cols=prepared.cols,
        component_key_indices=prepared.key_indices,
    )
    runtime.integrate_charge(
        0.0,
        AdaptiveOptions(target_error=2e-3, max_refinements=64, preview_depth=2),
    )

    density = runtime.integrate_density(
        0.0,
        AdaptiveOptions(target_error=1e-3, max_refinements=64, preview_depth=2),
    )

    assert density.refinements > 0
    assert density.converged
    assert density.error_scalar <= 1e-3
    assert density.estimate_array().shape == (prepared.value_count,)
    assert density.error_vector_array().shape == (prepared.value_count,)


@requires_native
def test_charge_evaluator_matches_meanfi_solver_callback_contract():
    tb = dimerized_chain()
    keys = [(0,), (1,), (-1,)]
    prepared = prepare_density_components(
        tb,
        keys,
        full_density_components(keys, size=2),
    )
    runtime = build_runtime(
        tb,
        keys=list(prepared.keys),
        component_rows=prepared.rows,
        component_cols=prepared.cols,
        component_key_indices=prepared.key_indices,
    )
    runtime.integrate_charge(
        0.0,
        AdaptiveOptions(target_error=2e-3, max_refinements=64),
    )
    evaluator = ChargeEvaluator(
        runtime,
        AdaptiveOptions(target_error=2e-3, max_refinements=0, preview_depth=2),
    )

    charge, charge_error, derivative = evaluator(0.0)

    assert charge == pytest.approx(1.0, abs=2e-3)
    assert charge_error <= 2e-3
    assert derivative >= 0.0
