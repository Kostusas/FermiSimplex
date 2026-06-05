from __future__ import annotations

import numpy as np
import pytest

from lineartetrahedron import (
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
    assert max(float(np.max(block)) for block in error.values()) <= 5e-3
    assert info.error_estimate_available is True
    assert info.n_cached_nodes > 0


@requires_native
def test_fixed_filling_workflow_is_driven_from_python():
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
        refinement_depth=0,
    )

    charge_calls = 0

    def residual(mu: float) -> float:
        nonlocal charge_calls
        charge_calls += 1
        result = runtime.integrate_charge(mu, 2e-3, 512)
        return result.charge - 1.0

    lower = -3.0
    upper = 3.0
    lower_residual = residual(lower)
    upper_residual = residual(upper)
    assert lower_residual <= 0.0 <= upper_residual
    for _ in range(64):
        mu = 0.5 * (lower + upper)
        value = residual(mu)
        if abs(value) <= 2e-3 or upper - lower <= 1e-6:
            break
        if value < 0.0:
            lower = mu
        else:
            upper = mu
    else:  # pragma: no cover - loop always returns or breaks
        raise AssertionError("bisection did not converge")
    density = runtime.integrate_density(mu, 5e-3, 512)
    rho, _error = prepared.values_and_errors_to_tb(
        density.estimate_array(),
        density.error_vector_array(),
    )
    reference = dense_reference(tb, mu=mu, keys=keys, nk=2001)

    assert charge_calls > 1
    assert abs(residual(mu)) <= 2e-3
    assert max_density_error(rho, reference.rho) <= 5e-3
