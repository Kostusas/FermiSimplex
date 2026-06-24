from __future__ import annotations

import numpy as np
import pytest

from lineartetrahedron import (
    AdaptiveOptions,
    Runtime,
    density_matrix_at_mu_zero_temp,
    fermi_surface,
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


def _axis_cosine_band(ndim: int) -> dict[tuple[int, ...], np.ndarray]:
    positive = [0] * ndim
    positive[0] = 1
    negative = [0] * ndim
    negative[0] = -1
    return {
        tuple(positive): np.array([[0.5]], dtype=complex),
        tuple(negative): np.array([[0.5]], dtype=complex),
    }


def _aliased_pocket_band() -> dict[tuple[int, ...], np.ndarray]:
    return {
        (0,): np.array([[0.5]], dtype=complex),
        (8,): np.array([[0.5]], dtype=complex),
        (-8,): np.array([[0.5]], dtype=complex),
    }


def _rotating_constant_gap_band() -> dict[tuple[int, ...], np.ndarray]:
    return _winding_constant_gap_band(2)


def _winding_constant_gap_band(winding: int) -> dict[tuple[int, ...], np.ndarray]:
    sigma_z = np.diag([1.0, -1.0]).astype(complex)
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    return {
        (winding,): 0.5 * sigma_z + 0.5j * sigma_x,
        (-winding,): 0.5 * sigma_z - 0.5j * sigma_x,
    }


@requires_native
def test_compiled_tight_binding_model_matches_python_fourier_evaluation():
    tb = qiwuzhang()
    compiled_model = _tb_to_tight_binding_model(tb)
    point = np.array([0.3, -0.7], dtype=float)

    assert compiled_model.ndim == 2
    assert compiled_model.ndof == 2
    assert compiled_model.nterms == len(tb)
    assert np.allclose(compiled_model.evaluate_point(point), tb_k_matrix(tb, point))


@requires_native
def test_product_simplex_triangulation_is_dimension_general():
    from lineartetrahedron import _native

    assert _native._product_simplex_triangulation_cells(1, 1) == [0]
    assert _native._product_simplex_triangulation_cells(1, 2) == [0, 1]
    assert _native._product_simplex_triangulation_cells(1, 3) == [0, 1, 2]
    assert len(_native._product_simplex_triangulation_cells(2, 2)) == 6
    assert len(_native._product_simplex_triangulation_cells(2, 3)) == 12


@requires_native
def test_lanczos_min_eigenvalue_matches_numpy_reference():
    from lineartetrahedron import _native

    matrix = np.array(
        [
            [2.0, 0.25 + 0.5j, -0.2j],
            [0.25 - 0.5j, -1.0, 0.75],
            [0.2j, 0.75, 0.5],
        ],
        dtype=np.complex128,
    )

    expected = np.linalg.eigvalsh(matrix)[0]
    estimated = _native._hermitian_min_eigenvalue_lanczos(
        np.ascontiguousarray(matrix),
        gap_atol=1e-12,
        gap_rtol=0.0,
    )

    assert estimated == pytest.approx(expected, abs=1e-10)


@requires_native
def test_generalized_lanczos_gap_bound_matches_numpy_reference():
    from lineartetrahedron import _native

    matrix = np.array(
        [
            [1.7, 0.2 - 0.1j],
            [0.2 + 0.1j, 2.4],
        ],
        dtype=np.complex128,
    )
    frame = np.array(
        [
            [1.25, 0.15 + 0.05j],
            [0.15 - 0.05j, 1.4],
        ],
        dtype=np.complex128,
    )
    cholesky = np.linalg.cholesky(frame)
    left_solved = np.linalg.solve(cholesky, matrix)
    transformed = np.linalg.solve(cholesky, left_solved.conj().T).conj().T
    transformed = 0.5 * (transformed + transformed.conj().T)

    expected = np.linalg.eigvalsh(transformed)[0]
    estimated = _native._generalized_hermitian_min_eigenvalue_lanczos(
        np.ascontiguousarray(matrix),
        np.ascontiguousarray(frame),
        gap_atol=1e-12,
        gap_rtol=0.0,
    )

    assert estimated == pytest.approx(expected, abs=1e-10)


@requires_native
def test_root_simplex_certificate_gap_bound_reports_physical_gap():
    from lineartetrahedron import _native

    model = _tb_to_tight_binding_model(_constant_insulator(2))

    unbuffered = _native._root_mesh_certificate_gap_bound(
        model,
        0.0,
        margin=0.0,
        gap_atol=1e-12,
        gap_rtol=0.0,
    )
    buffered = _native._root_mesh_certificate_gap_bound(
        model,
        0.0,
        margin=0.5,
        gap_atol=1e-12,
        gap_rtol=0.0,
    )

    assert unbuffered == pytest.approx(1.0, abs=1e-10)
    assert buffered == pytest.approx(1.0, abs=1e-10)


@requires_native
@pytest.mark.parametrize("ndim", (1, 2, 3, 4))
def test_fermi_surface_extracts_nd_level_set(ndim):
    mu = 0.3
    feature_size = {1: 0.1, 2: 0.1, 3: 0.24, 4: 0.56}[ndim]
    surface = fermi_surface(
        _axis_cosine_band(ndim),
        mu=mu,
        min_feature_size=feature_size,
        max_diagonalizations=20000,
    )

    assert surface.points.shape[1] == ndim
    assert surface.cells.shape[1] == ndim
    assert surface.converged
    assert surface.stats.unresolved_simplices == 0
    assert surface.points.shape[0] > 0
    assert surface.cells.shape[0] > 0
    assert np.all(surface.points >= -1e-14)
    assert np.all(surface.points <= 1.0 + 1e-14)
    assert np.max(np.abs(np.cos(2.0 * np.pi * surface.points[:, 0]) - mu)) < 0.2
    assert surface.parameters.min_feature_size == pytest.approx(feature_size)


@requires_native
def test_fermi_surface_callable_matches_tight_binding_dict():
    def callable_band(kx: float) -> np.ndarray:
        return np.array([[np.cos(2.0 * np.pi * kx)]], dtype=complex)

    tb_surface = fermi_surface(
        _axis_cosine_band(1),
        mu=0.25,
        min_feature_size=0.03,
    )
    callable_surface = fermi_surface(
        callable_band,
        mu=0.25,
        min_feature_size=0.03,
    )

    assert callable_surface.parameters.ndim == 1
    assert callable_surface.parameters.ndof == 1
    assert callable_surface.converged
    assert np.allclose(
        np.sort(callable_surface.points[:, 0]),
        np.sort(tb_surface.points[:, 0]),
        atol=0.04,
    )
    assert callable_surface.states is None


@requires_native
def test_fermi_surface_can_return_states():
    def two_band(kx: float) -> np.ndarray:
        value = np.cos(2.0 * np.pi * kx)
        return np.array([[value, 0.0], [0.0, 2.0]], dtype=complex)

    surface = fermi_surface(
        two_band,
        mu=0.25,
        min_feature_size=0.03,
        return_states=True,
    )

    assert surface.states is not None
    states = surface.states
    assert states.band_indices.shape == (surface.points.shape[0],)
    assert states.eigenvalues.shape == (surface.points.shape[0],)
    assert states.eigenvectors.shape == (surface.points.shape[0], 2)
    assert np.all(states.band_indices == 0)
    assert np.max(np.abs(np.linalg.norm(states.eigenvectors, axis=1) - 1.0)) < 1e-12
    orbital_weight = np.abs(states.eigenvectors[:, 0]) ** 2
    assert np.allclose(orbital_weight, 1.0)


@requires_native
def test_fermi_surface_rejects_unknown_return_states_mode():
    with pytest.raises(TypeError, match="return_states"):
        fermi_surface(
            _axis_cosine_band(1),
            mu=0.25,
            min_feature_size=0.03,
            return_states="interpolated",
        )


@requires_native
def test_fermi_surface_accepts_two_scalar_callable_arguments():
    def callable_band(kx: float, ky: float) -> np.ndarray:
        return np.array([[kx + ky - 0.8]], dtype=complex)

    surface = fermi_surface(callable_band, mu=0.0, min_feature_size=0.12)

    assert surface.parameters.ndim == 2
    assert surface.parameters.ndof == 1
    assert surface.points.shape[1] == 2
    assert surface.points.shape[0] > 0
    assert np.all(surface.points >= -1e-14)
    assert np.all(surface.points <= 1.0 + 1e-14)


@requires_native
def test_fermi_surface_rejects_unsupported_callable_signatures():
    def vector_style(k):
        kx, ky = k
        return np.array([[kx + ky]], dtype=complex)

    def default_coordinate(kx, ky=0.0):
        return np.array([[kx + ky]], dtype=complex)

    def varargs(*coords):
        return np.array([[sum(coords)]], dtype=complex)

    with pytest.raises(TypeError, match="scalar coordinate"):
        fermi_surface(vector_style, min_feature_size=0.1)
    with pytest.raises(TypeError, match="defaults"):
        fermi_surface(default_coordinate, min_feature_size=0.1)
    with pytest.raises(TypeError, match=r"\*args"):
        fermi_surface(varargs, min_feature_size=0.1)


@requires_native
def test_fermi_surface_rejects_invalid_callable_matrix_outputs():
    def nonsquare(kx):
        return np.zeros((1, 2), dtype=complex)

    def shape_changing(kx):
        if kx == 0.0:
            return np.array([[1.0]], dtype=complex)
        return np.eye(2, dtype=complex)

    with pytest.raises(ValueError, match="square dense matrix"):
        fermi_surface(nonsquare, min_feature_size=0.1)
    with pytest.raises(ValueError, match="inconsistent shape"):
        fermi_surface(shape_changing, min_feature_size=0.1, max_diagonalizations=8)


@requires_native
def test_fermi_surface_respects_max_diagonalizations():
    surface = fermi_surface(
        _axis_cosine_band(2),
        mu=0.0,
        min_feature_size=0.02,
        max_diagonalizations=3,
    )

    assert not surface.converged
    assert surface.stats.evaluated_vertices <= 3
    assert surface.stats.unresolved_simplices > 0


@requires_native
def test_fermi_surface_reports_unresolved_when_diagonalization_budget_exhausted():
    surface = fermi_surface(
        _axis_cosine_band(1),
        mu=0.0,
        min_feature_size=0.1,
        max_diagonalizations=0,
    )

    assert not surface.converged
    assert surface.stats.unresolved_simplices > 0
    assert surface.stats.evaluated_vertices <= 0


@requires_native
def test_fermi_surface_inertia_certificate_certifies_constant_insulator():
    surface = fermi_surface(
        _constant_insulator(2),
        mu=0.0,
        min_feature_size=0.4,
    )

    assert surface.converged
    assert surface.stats.cut_simplices == 0
    assert surface.stats.feature_size_simplices == 0
    assert surface.stats.unresolved_simplices == 0
    assert surface.points.shape[1] == 2
    assert surface.stats.evaluated_vertices > 0


@requires_native
def test_fermi_surface_margin_is_reported_and_can_block_certification():
    safe = fermi_surface(
        _constant_insulator(1),
        mu=0.0,
        min_feature_size=1.0,
        margin=0.5,
    )
    strict = fermi_surface(
        _constant_insulator(1),
        mu=0.0,
        min_feature_size=1.0,
        margin=2.0,
    )

    assert safe.parameters.margin == pytest.approx(0.5)
    assert safe.converged
    assert safe.stats.feature_size_simplices == 0
    assert strict.parameters.margin == pytest.approx(2.0)
    assert strict.stats.feature_size_simplices > 0


@requires_native
def test_fermi_surface_rejects_negative_margin():
    with pytest.raises(ValueError, match="margin"):
        fermi_surface(_constant_insulator(1), mu=0.0, min_feature_size=0.1, margin=-1e-3)


@requires_native
def test_fermi_surface_inertia_certificate_accepts_cut_below_feature_size():
    surface = fermi_surface(
        _axis_cosine_band(1),
        mu=0.0,
        min_feature_size=0.4,
    )

    assert surface.converged
    assert surface.stats.cut_simplices > 0
    assert surface.stats.unresolved_simplices == 0


@requires_native
def test_fermi_surface_inertia_certificate_repeated_runs_preserve_metadata():
    first = fermi_surface(
        _aliased_pocket_band(),
        mu=0.0,
        min_feature_size=0.4,
    )
    second = fermi_surface(
        _aliased_pocket_band(),
        mu=0.0,
        min_feature_size=0.4,
    )

    assert second.converged == first.converged
    assert second.stats.evaluated_vertices == first.stats.evaluated_vertices
    assert second.stats.unresolved_simplices == first.stats.unresolved_simplices
    assert second.points.shape == first.points.shape
    assert second.cells.shape == first.cells.shape


@requires_native
@pytest.mark.parametrize("ndim", (1, 2, 3, 4))
def test_runtime_accepts_reduced_coordinate_dimensions(ndim):
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
@pytest.mark.parametrize("ndim", (2, 3, 4))
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
def test_charge_certificate_uses_preview_error_for_stopping():
    runtime = Runtime(_rotating_constant_gap_band(), keys=[(0,)])
    options = AdaptiveOptions(target_error=1e-3, max_refinements=0, preview_depth=1)

    result = runtime.evaluate_charge(0.0, options)

    assert result.charge == pytest.approx(1.0)
    assert result.charge_error == pytest.approx(0.0)
    assert result.converged


@requires_native
def test_preview_certified_charge_integrates_without_refinement():
    runtime = Runtime(_rotating_constant_gap_band(), keys=[(0,)])
    options = AdaptiveOptions(target_error=1e-3, max_refinements=0, preview_depth=1)

    result = runtime.integrate_charge(0.0, options)

    assert result.charge == pytest.approx(1.0)
    assert result.charge_error == pytest.approx(0.0)
    assert result.converged


@requires_native
def test_volume_bounded_certificate_error_can_be_within_charge_tolerance():
    runtime = Runtime(_winding_constant_gap_band(3), keys=[(0,)])
    options = AdaptiveOptions(target_error=1.1, max_refinements=0, preview_depth=1)

    result = runtime.evaluate_charge(0.0, options)

    assert result.charge == pytest.approx(1.0)
    assert result.charge_error == pytest.approx(1.0)
    assert result.converged


@requires_native
def test_volume_bounded_certificate_error_can_block_charge_convergence():
    runtime = Runtime(_winding_constant_gap_band(3), keys=[(0,)])
    options = AdaptiveOptions(target_error=0.75, max_refinements=0, preview_depth=1)

    result = runtime.evaluate_charge(0.0, options)

    assert result.charge == pytest.approx(1.0)
    assert result.charge_error == pytest.approx(1.0)
    assert not result.converged


@requires_native
def test_uncertified_charge_evaluation_suppresses_certificate_error():
    runtime = Runtime(_winding_constant_gap_band(3), keys=[(0,)])
    options = AdaptiveOptions(target_error=0.75, max_refinements=0, preview_depth=1)

    certified = runtime.evaluate_charge(0.0, options)
    uncertified = runtime.evaluate_charge(0.0, options, certify=False)

    assert certified.charge == pytest.approx(uncertified.charge)
    assert certified.charge_error == pytest.approx(1.0)
    assert not certified.converged
    assert uncertified.charge_error == pytest.approx(0.0)
    assert uncertified.converged


@requires_native
def test_uncertified_charge_evaluation_does_not_refine_active_mesh():
    runtime = Runtime(_winding_constant_gap_band(3), keys=[(0,)])
    options = AdaptiveOptions(target_error=0.75, max_refinements=0, preview_depth=1)

    active_before = runtime.n_active_simplices
    result = runtime.evaluate_charge(0.0, options, certify=False)
    active_after = runtime.n_active_simplices

    assert result.refinements == 0
    assert active_after == active_before


@requires_native
def test_integrate_charge_remains_certified():
    runtime = Runtime(_winding_constant_gap_band(3), keys=[(0,)])
    options = AdaptiveOptions(target_error=0.75, max_refinements=0, preview_depth=1)

    with pytest.raises(RuntimeError, match="did not converge"):
        runtime.integrate_charge(0.0, options)


@requires_native
def test_inertia_certificate_repeated_charge_evaluation_is_deterministic():
    runtime = Runtime(_axis_cosine_band(1), keys=[(0,)])
    options = AdaptiveOptions(target_error=1e-3, max_refinements=0, preview_depth=2)

    first = runtime.evaluate_charge(0.0, options)
    second = runtime.evaluate_charge(0.0, options)

    assert second.charge == pytest.approx(first.charge)
    assert second.charge_error == pytest.approx(first.charge_error)
    assert second.dcharge_dmu == pytest.approx(first.dcharge_dmu)
    assert second.converged == first.converged


@requires_native
def test_inertia_certificate_does_not_add_error_for_visible_charge_cut():
    runtime = Runtime(_axis_cosine_band(1), keys=[(0,)])
    options = AdaptiveOptions(target_error=1e-3, max_refinements=0, preview_depth=1)

    result = runtime.evaluate_charge(0.0, options)

    assert result.charge_error < options.target_error


@requires_native
def test_inertia_certificate_does_not_add_error_for_certified_gapped_simplex():
    runtime = Runtime(_constant_insulator(1), keys=[(0,)])
    options = AdaptiveOptions(target_error=1e-3, max_refinements=0, preview_depth=1)

    result = runtime.evaluate_charge(0.0, options)

    assert result.charge_error == pytest.approx(0.0)


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
