from __future__ import annotations

import numpy as np
import pytest

from lineartetrahedron import (
    AdaptiveOptions,
    Runtime,
    density_matrix_at_mu_zero_temp,
)
from lineartetrahedron.backend import _tb_to_tight_binding_model

from .conftest import requires_native
from .helpers import dense_reference, dimerized_chain, max_density_error, qiwuzhang, tb_k_matrix


def _constant_insulator(ndim: int) -> dict[tuple[int, ...], np.ndarray]:
    return {(0,) * ndim: np.diag([-1.0, 1.0]).astype(complex)}


def _cosine_band(ndim: int) -> dict[tuple[int, ...], np.ndarray]:
    tb = {(0,) * ndim: np.zeros((1, 1), dtype=complex)}
    for axis in range(ndim):
        key = [0] * ndim
        key[axis] = 1
        tb[tuple(key)] = np.array([[0.5]], dtype=complex)
        key[axis] = -1
        tb[tuple(key)] = np.array([[0.5]], dtype=complex)
    return tb


def _aliased_pocket_band() -> dict[tuple[int, ...], np.ndarray]:
    return {
        (0,): np.array([[0.5]], dtype=complex),
        (8,): np.array([[0.5]], dtype=complex),
        (-8,): np.array([[0.5]], dtype=complex),
    }


def _derivative_matrix(
    tb: dict[tuple[int, ...], np.ndarray],
    point: np.ndarray,
    axis: int,
) -> np.ndarray:
    return sum(
        (-1j * key[axis]) * matrix * np.exp(-1j * np.dot(point, np.asarray(key, dtype=float)))
        for key, matrix in tb.items()
    )


@requires_native
def test_compiled_tight_binding_model_matches_python_fourier_evaluation():
    tb = qiwuzhang()
    compiled_model = _tb_to_tight_binding_model(tb)
    point = np.array([0.3, -0.7], dtype=float)

    assert compiled_model.ndim == 2
    assert compiled_model.ndof == 2
    assert compiled_model.nterms == len(tb)
    assert np.allclose(compiled_model.evaluate_point(point), tb_k_matrix(tb, point))
    assert compiled_model.reduced_lipschitz_bound > 0.0


@requires_native
def test_lanczos_spectral_norms_match_numpy_reference():
    tb = dimerized_chain()
    compiled_model = _tb_to_tight_binding_model(tb)
    point = np.array([0.37], dtype=float)

    expected_hopping_norms = [
        np.linalg.norm(matrix, ord=2)
        for matrix in tb.values()
    ]
    assert np.allclose(np.asarray(compiled_model.hopping_spectral_norms), expected_hopping_norms)

    derivative = _derivative_matrix(tb, point, axis=0)
    assert compiled_model.derivative_spectral_norm(point, 0) == pytest.approx(
        np.linalg.norm(derivative, ord=2)
    )


@requires_native
def test_hessian_local_bound_can_certify_global_uncertainty_safe():
    tb = {
        (0,): np.array([[2.0]], dtype=complex),
        (1,): np.array([[0.5j]], dtype=complex),
        (-1,): np.array([[-0.5j]], dtype=complex),
    }
    compiled_model = _tb_to_tight_binding_model(tb)
    center = np.array([0.5 * np.pi], dtype=float)
    rho = 0.125
    mu = 2.85
    vertex_values = [2.0 + np.sin(center[0] - rho), 2.0 + np.sin(center[0] + rho)]
    global_derivative_bounds = np.asarray(compiled_model.global_derivative_bounds)
    hessian_bounds = np.asarray(compiled_model.hessian_bounds).reshape((1, 1))
    global_delta = 2.0 * float(global_derivative_bounds[0]) * rho
    local_axis_bound = (
        compiled_model.derivative_spectral_norm(center, 0)
        + float(hessian_bounds[0, 0]) * rho
    )
    local_delta = 2.0 * local_axis_bound * rho

    assert min(vertex_values) - global_delta <= mu
    assert min(vertex_values) - local_delta > mu


@requires_native
@pytest.mark.parametrize("ndim", (1, 2, 3))
def test_runtime_accepts_physical_dimensions(ndim):
    tb = _constant_insulator(ndim)
    key = (0,) * ndim
    rho, _error, info = density_matrix_at_mu_zero_temp(
        tb,
        mu=0.0,
        keys=[key],
        density_atol=1e-12,
        max_subdivisions=8,
    )

    assert np.allclose(rho[key], np.diag([1.0, 0.0]), atol=1e-12)
    assert info.n_cached_nodes > 0


@requires_native
def test_runtime_accepts_adaptive_options():
    tb = _constant_insulator(1)
    key = (0,)
    runtime = Runtime(tb, keys=[key])
    options = AdaptiveOptions(
        target_error=1e-12,
        max_refinements=8,
        preview_depth=2,
        min_refinement_batch_size=1,
        max_refinement_batch_size=100,
    )

    charge = runtime.integrate_charge(0.0, options)
    fixed_charge = runtime.evaluate_charge(0.0, options)
    rho, _error, _info = runtime.integrate_density(0.0, options)

    assert charge.charge == pytest.approx(1.0)
    assert fixed_charge.charge == pytest.approx(1.0)
    assert fixed_charge.refinements == 0
    assert np.allclose(rho[key], np.diag([1.0, 0.0]))


@requires_native
def test_degenerate_band_at_mu_uses_on_level_half_occupation():
    tb = {(0,): np.zeros((2, 2), dtype=complex)}
    key = (0,)
    runtime = Runtime(tb, keys=[key])
    options = AdaptiveOptions(target_error=1e-12, max_refinements=0)

    charge = runtime.integrate_charge(0.0, options)
    rho, error, _info = runtime.integrate_density(0.0, options)

    assert charge.charge == pytest.approx(1.0)
    assert charge.charge_error == pytest.approx(0.0)
    assert charge.dcharge_dmu == pytest.approx(0.0)
    assert np.allclose(rho[key], 0.5 * np.eye(2))
    assert np.allclose(error[key], 0.0)


@requires_native
def test_charge_derivative_matches_linear_1d_reference():
    tb = _cosine_band(1)
    runtime = Runtime(tb, keys=[(0,)])
    result = runtime.evaluate_charge(
        0.25,
        AdaptiveOptions(target_error=1e-12, max_refinements=0, preview_depth=1),
    )

    # This is the piecewise-linear derivative on the root preview mesh, not the exact cosine DOS.
    assert result.dcharge_dmu == pytest.approx(1.0 / (2.0 * np.sqrt(2.0)))


@requires_native
@pytest.mark.parametrize("ndim", (2, 3))
def test_charge_derivative_is_finite_and_nonnegative_for_higher_dimensions(ndim):
    tb = _cosine_band(ndim)
    runtime = Runtime(tb, keys=[(0,) * ndim])
    result = runtime.evaluate_charge(
        0.0,
        AdaptiveOptions(target_error=1e-12, max_refinements=0, preview_depth=1),
    )

    assert np.isfinite(result.dcharge_dmu)
    assert result.dcharge_dmu >= 0.0


@requires_native
def test_weyl_bounds_mark_simplex_when_vertices_miss_pocket():
    runtime = Runtime(_aliased_pocket_band(), keys=[(0,)])
    options = AdaptiveOptions(target_error=1e-3, max_refinements=0, preview_depth=1)

    disabled = runtime.evaluate_charge(0.0, options)
    enabled = runtime.evaluate_charge(0.0, options, use_weyl_bounds=True)

    assert disabled.charge == pytest.approx(0.0)
    assert disabled.charge_error == pytest.approx(0.0)
    assert disabled.converged
    assert enabled.charge == pytest.approx(0.0)
    assert enabled.charge_error > options.target_error
    assert not enabled.converged


@requires_native
def test_weyl_bounds_participate_in_charge_stopping_error():
    runtime = Runtime(_aliased_pocket_band(), keys=[(0,)])
    options = AdaptiveOptions(target_error=1e-3, max_refinements=0, preview_depth=1)

    with pytest.raises(RuntimeError, match="did not converge"):
        runtime.integrate_charge(0.0, options, use_weyl_bounds=True)


@requires_native
def test_runtime_rejects_higher_dimensions():
    tb = _constant_insulator(4)
    key = (0, 0, 0, 0)

    with pytest.raises(ValueError, match="supports dimensions 1, 2, and 3"):
        density_matrix_at_mu_zero_temp(
            tb,
            mu=0.0,
            keys=[key],
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
        density_atol=5e-3,
        max_subdivisions=512,
    )

    assert max_density_error(rho, reference.rho) <= 5e-3
    assert max(float(np.max(np.abs(block))) for block in error.values()) <= 5e-3
    assert info.n_cached_nodes > 0


@requires_native
def test_charge_evaluation_does_not_change_active_simplex_count():
    tb = dimerized_chain()
    keys = [(0,), (1,), (-1,)]
    runtime = Runtime(tb, keys=keys)
    runtime.integrate_charge(
        0.0,
        AdaptiveOptions(target_error=2e-3, max_refinements=64),
    )
    options = AdaptiveOptions(target_error=2e-3, max_refinements=0, preview_depth=2)

    active_before = runtime.n_active_simplices
    values = [runtime.evaluate_charge(mu, options) for mu in (-1.0, 0.0, 1.0, 0.25)]
    active_after = runtime.n_active_simplices

    assert active_after == active_before
    assert all(np.isfinite(value.charge) for value in values)
    assert all(value.charge_error >= 0.0 for value in values)


@requires_native
def test_cached_density_initial_path_reuses_charge_vertices():
    tb = dimerized_chain()
    keys = [(0,), (1,), (-1,)]
    runtime = Runtime(tb, keys=keys)
    runtime.integrate_charge(
        0.0,
        AdaptiveOptions(target_error=2e-3, max_refinements=64, preview_depth=2),
    )
    cached_before = runtime.n_cached_nodes

    rho, error, info = runtime.integrate_density(
        0.0,
        AdaptiveOptions(target_error=1.0, max_refinements=0, preview_depth=2),
    )

    assert info.n_evaluator_evals == 0
    assert info.subdivisions == 0
    assert runtime.n_cached_nodes == cached_before
    assert set(rho) == set(keys)
    assert set(error) == set(keys)
    assert max(float(np.max(np.abs(block))) for block in error.values()) <= 1.0


@requires_native
def test_density_tight_target_refines():
    tb = dimerized_chain()
    keys = [(0,), (1,), (-1,)]
    runtime = Runtime(tb, keys=keys)
    runtime.integrate_charge(
        0.0,
        AdaptiveOptions(target_error=2e-3, max_refinements=64, preview_depth=2),
    )

    rho, error, info = runtime.integrate_density(
        0.0,
        AdaptiveOptions(target_error=1e-3, max_refinements=64, preview_depth=2),
    )

    assert info.subdivisions > 0
    assert set(rho) == set(keys)
    assert set(error) == set(keys)
    assert max(float(np.max(np.abs(block))) for block in error.values()) <= 1e-3


@requires_native
def test_charge_evaluation_can_be_wrapped_for_solver_callbacks():
    tb = dimerized_chain()
    keys = [(0,), (1,), (-1,)]
    runtime = Runtime(tb, keys=keys)
    runtime.integrate_charge(
        0.0,
        AdaptiveOptions(target_error=2e-3, max_refinements=64),
    )
    options = AdaptiveOptions(target_error=2e-3, max_refinements=0, preview_depth=2)

    def evaluate(mu):
        result = runtime.evaluate_charge(mu, options)
        return result.charge, result.charge_error, result.dcharge_dmu

    charge, charge_error, derivative = evaluate(0.0)

    assert charge == pytest.approx(1.0, abs=2e-3)
    assert charge_error <= 2e-3
    assert derivative >= 0.0
