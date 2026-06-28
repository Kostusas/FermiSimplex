from __future__ import annotations

import numpy as np
import pytest

from lineartetrahedron import (
    AdaptiveOptions,
    SpectralMesh,
    certify_simplex,
    charge,
    density_matrix_components,
    fermi_surface,
    tight_binding_hessian_bound,
    tight_binding_hamiltonian,
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


def _mesh(tb: dict[tuple[int, ...], np.ndarray]) -> SpectralMesh:
    return SpectralMesh(tight_binding_hamiltonian(tb))


def _all_components(ndof: int) -> list[tuple[int, int]]:
    return [(row, col) for row in range(ndof) for col in range(ndof)]


def _components_to_density_dict(result, ndof: int):
    rho = {key: np.zeros((ndof, ndof), dtype=complex) for key in result.keys}
    error = {key: np.zeros((ndof, ndof), dtype=float) for key in result.keys}
    for key_index, key in enumerate(result.keys):
        for component_index, (row, col) in enumerate(result.components):
            rho[key][row, col] = result.values[key_index, component_index]
            error[key][row, col] = result.errors[key_index, component_index]
    return rho, error


def _density_dict(
    mesh: SpectralMesh,
    *,
    mu: float,
    keys,
    components=None,
    target_error: float,
    max_refinements: int | None = None,
    options=None,
    refine: bool = True,
):
    selected_components = _all_components(mesh.ndof) if components is None else components
    result = density_matrix_components(
        mesh,
        mu=mu,
        keys=keys,
        components=selected_components,
        target_error=target_error if options is None else None,
        options=options,
        max_refinements=max_refinements,
        refine=refine,
    )
    rho, error = _components_to_density_dict(result, mesh.ndof)
    return rho, error, result.stats


def _spectra_for_points(tb: dict[tuple[int, ...], np.ndarray], points: list[float]):
    eigenvalues = []
    eigenvectors = []
    for point in points:
        values, vectors = np.linalg.eigh(tb_k_matrix(tb, np.array([point], dtype=float)))
        eigenvalues.append(values)
        eigenvectors.append(vectors)
    return np.asarray(eigenvalues), np.asarray(eigenvectors)


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
def test_tight_binding_hessian_bound_matches_scalar_cosine_band():
    bound = tight_binding_hessian_bound(_cosine_band(1))

    assert bound(0.0) == pytest.approx(4.0 * np.pi**2)
    assert bound(0.25) == pytest.approx(0.0, abs=1e-12)
    assert bound(0.5) == pytest.approx(4.0 * np.pi**2)

    with pytest.raises(TypeError, match="expected 1 coordinates"):
        bound(0.0, 0.0)


@requires_native
def test_tight_binding_hessian_bound_is_publicly_exported():
    import lineartetrahedron as lt

    assert callable(lt.tight_binding_hessian_bound)


@requires_native
def test_product_simplex_triangulation_is_dimension_general():
    from lineartetrahedron import _native

    assert _native._product_simplex_triangulation_cells(1, 1) == [0]
    assert _native._product_simplex_triangulation_cells(1, 2) == [0, 1]
    assert _native._product_simplex_triangulation_cells(1, 3) == [0, 1, 2]
    assert len(_native._product_simplex_triangulation_cells(2, 2)) == 6
    assert len(_native._product_simplex_triangulation_cells(2, 3)) == 12


@requires_native
@pytest.mark.parametrize("ndim", (1, 2, 3, 4))
def test_fermi_surface_extracts_nd_level_set(ndim):
    mu = 0.3
    feature_size = {1: 0.1, 2: 0.1, 3: 0.24, 4: 0.56}[ndim]
    surface = fermi_surface(
        tight_binding_hamiltonian(_axis_cosine_band(ndim)),
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
        tight_binding_hamiltonian(_axis_cosine_band(1)),
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
def test_fermi_surface_accepts_spectral_mesh():
    def callable_band(kx: float) -> np.ndarray:
        return np.array([[np.cos(2.0 * np.pi * kx)]], dtype=complex)

    mesh = SpectralMesh(callable_band)
    surface = fermi_surface(mesh, mu=0.25, min_feature_size=0.03)

    assert surface.parameters.ndim == 1
    assert surface.parameters.ndof == 1
    assert surface.converged
    assert surface.points.shape[0] > 0


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
            tight_binding_hamiltonian(_axis_cosine_band(1)),
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
def test_spectral_mesh_accepts_callable_and_tight_binding_helper_inputs():
    tb_mesh = _mesh(_constant_insulator(1))
    assert tb_mesh.ndim == 1
    assert tb_mesh.ndof == 2

    def callable_insulator(kx: float) -> np.ndarray:
        return np.diag([-1.0, 1.0]).astype(complex)

    callable_mesh = SpectralMesh(callable_insulator)
    result = charge(callable_mesh, target_error=1e-12, max_refinements=0)

    assert callable_mesh.ndim == 1
    assert callable_mesh.ndof == 2
    assert result.charge == pytest.approx(1.0)


@requires_native
def test_public_hamiltonian_inputs_reject_raw_tight_binding_dicts():
    with pytest.raises(TypeError, match="callable"):
        SpectralMesh(_constant_insulator(1))
    with pytest.raises(TypeError, match="callable"):
        fermi_surface(_constant_insulator(1), min_feature_size=0.1)


@requires_native
def test_spectral_mesh_rejects_shape_changing_callable_during_evaluation():
    def shape_changing(kx: float) -> np.ndarray:
        if kx == 0.0:
            return np.array([[1.0]], dtype=complex)
        return np.eye(2, dtype=complex)

    mesh = SpectralMesh(shape_changing)
    with pytest.raises(ValueError, match="inconsistent shape"):
        charge(mesh, target_error=1e-12, max_refinements=0)


@requires_native
def test_fermi_surface_respects_max_diagonalizations():
    surface = fermi_surface(
        tight_binding_hamiltonian(_axis_cosine_band(2)),
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
        tight_binding_hamiltonian(_axis_cosine_band(1)),
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
        tight_binding_hamiltonian(_constant_insulator(2)),
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
def test_fermi_surface_hessian_bound_is_reported_and_can_block_certification():
    safe = fermi_surface(
        tight_binding_hamiltonian(_constant_insulator(1)),
        mu=0.0,
        min_feature_size=1.0,
        hessian_bound=0.5,
    )
    strict = fermi_surface(
        tight_binding_hamiltonian(_constant_insulator(1)),
        mu=0.0,
        min_feature_size=1.0,
        hessian_bound=1.0e6,
    )

    assert safe.parameters.hessian_bound == pytest.approx(0.5)
    assert not safe.parameters.hessian_bound_is_callable
    assert safe.parameters.anharmonicity_bound == pytest.approx(0.0)
    assert safe.converged
    assert safe.stats.feature_size_simplices == 0
    assert strict.parameters.hessian_bound == pytest.approx(1.0e6)
    assert strict.stats.feature_size_simplices > 0


@requires_native
def test_fermi_surface_accepts_callable_hessian_bound():
    calls = []

    def hessian_bound(k0):
        calls.append(k0)
        return 1.0e6

    surface = fermi_surface(
        tight_binding_hamiltonian(_constant_insulator(1)),
        mu=0.0,
        min_feature_size=1.0,
        hessian_bound=hessian_bound,
    )

    assert calls
    assert surface.parameters.hessian_bound is None
    assert surface.parameters.hessian_bound_is_callable
    assert surface.stats.feature_size_simplices > 0


@requires_native
def test_fermi_surface_rejects_invalid_energy_bounds():
    with pytest.raises(ValueError, match="hessian_bound"):
        fermi_surface(
            tight_binding_hamiltonian(_constant_insulator(1)),
            mu=0.0,
            min_feature_size=0.1,
            hessian_bound=-1e-3,
        )
    with pytest.raises(ValueError, match="anharmonicity_bound"):
        fermi_surface(
            tight_binding_hamiltonian(_constant_insulator(1)),
            mu=0.0,
            min_feature_size=0.1,
            anharmonicity_bound=float("nan"),
        )
    with pytest.raises(RuntimeError, match="hessian_bound callable"):
        fermi_surface(
            tight_binding_hamiltonian(_constant_insulator(1)),
            mu=0.0,
            min_feature_size=0.1,
            hessian_bound=lambda k0: -1.0,
        )


@requires_native
def test_fermi_surface_inertia_certificate_accepts_cut_below_feature_size():
    surface = fermi_surface(
        tight_binding_hamiltonian(_axis_cosine_band(1)),
        mu=0.0,
        min_feature_size=0.4,
    )

    assert surface.converged
    assert surface.stats.cut_simplices > 0
    assert surface.stats.unresolved_simplices == 0


@requires_native
def test_fermi_surface_inertia_certificate_repeated_runs_preserve_metadata():
    first = fermi_surface(
        tight_binding_hamiltonian(_aliased_pocket_band()),
        mu=0.0,
        min_feature_size=0.4,
    )
    second = fermi_surface(
        tight_binding_hamiltonian(_aliased_pocket_band()),
        mu=0.0,
        min_feature_size=0.4,
    )

    assert second.converged == first.converged
    assert second.stats.evaluated_vertices == first.stats.evaluated_vertices
    assert second.stats.unresolved_simplices == first.stats.unresolved_simplices
    assert second.points.shape == first.points.shape
    assert second.cells.shape == first.cells.shape


@requires_native
def test_certify_simplex_reports_certified_gapped_spectra():
    eigenvalues = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    eigenvectors = np.repeat(np.eye(2, dtype=complex)[None, :, :], 2, axis=0)

    certificate = certify_simplex(eigenvalues, eigenvectors, margin=0.25)

    assert certificate.status == "certified_gapped"
    assert certificate.occupation_bounds.lower == 1
    assert certificate.occupation_bounds.upper == 1
    assert certificate.occupation_width == 0
    assert certificate.energy_bound == pytest.approx(0.25)
    assert certificate.reusable_at(0.0)


@requires_native
def test_certify_simplex_reports_visible_gapless_spectra():
    eigenvalues = np.array([[0.0, 1.0], [-1.0, 1.0]])
    eigenvectors = np.repeat(np.eye(2, dtype=complex)[None, :, :], 2, axis=0)

    certificate = certify_simplex(eigenvalues, eigenvectors, margin=0.125)

    assert certificate.status == "visible_gapless"
    assert certificate.occupation_bounds.lower == 0
    assert certificate.occupation_bounds.upper == 1
    assert certificate.energy_bound == pytest.approx(0.125)
    assert certificate.has_mu_interval


@requires_native
def test_certify_simplex_reports_inconclusive_occupation_bounds():
    eigenvalues, eigenvectors = _spectra_for_points(_winding_constant_gap_band(2), [0.0, 0.25])

    certificate = certify_simplex(eigenvalues, eigenvectors)

    assert certificate.status == "inconclusive"
    assert certificate.occupation_bounds.lower == 0
    assert certificate.occupation_bounds.upper == 2
    assert certificate.occupation_width == 2


@requires_native
def test_removed_python_api_names_are_not_exported():
    import lineartetrahedron as lt

    assert not hasattr(lt, "Runtime")
    assert not hasattr(lt, "density_matrix_at_mu_zero_temp")


@requires_native
@pytest.mark.parametrize("ndim", (1, 2, 3, 4))
def test_runtime_accepts_reduced_coordinate_dimensions(ndim):
    tb = _constant_insulator(ndim)
    key = (0,) * ndim
    mesh = _mesh(tb)
    rho, _error, info = _density_dict(
        mesh,
        mu=0.0,
        keys=[key],
        target_error=1e-12,
        max_refinements=8,
    )

    assert np.allclose(rho[key], np.diag([1.0, 0.0]), atol=1e-12)
    assert info.n_cached_nodes > 0


@requires_native
def test_runtime_accepts_adaptive_options():
    tb = _constant_insulator(1)
    key = (0,)
    mesh = _mesh(tb)
    options = AdaptiveOptions(
        target_error=1e-12,
        max_refinements=8,
        preview_depth=2,
        min_refinement_batch_size=1,
        max_refinement_batch_size=100,
    )

    charge_result = charge(mesh, mu=0.0, options=options)
    fixed_charge = charge(mesh, mu=0.0, options=options, refine=False)
    rho, _error, _info = _density_dict(
        mesh,
        mu=0.0,
        keys=[key],
        target_error=1e-12,
        options=options,
    )

    assert charge_result.charge == pytest.approx(1.0)
    assert fixed_charge.charge == pytest.approx(1.0)
    assert fixed_charge.refinements == 0
    assert np.allclose(rho[key], np.diag([1.0, 0.0]))


@requires_native
def test_degenerate_band_at_mu_uses_on_level_half_occupation():
    tb = {(0,): np.zeros((2, 2), dtype=complex)}
    key = (0,)
    mesh = _mesh(tb)
    options = AdaptiveOptions(target_error=1e-12, max_refinements=0)

    charge_result = charge(mesh, mu=0.0, options=options)
    rho, error, _info = _density_dict(
        mesh,
        mu=0.0,
        keys=[key],
        target_error=1e-12,
        options=options,
    )

    assert charge_result.charge == pytest.approx(1.0)
    assert charge_result.charge_error == pytest.approx(0.0)
    assert charge_result.dcharge_dmu == pytest.approx(0.0)
    assert np.allclose(rho[key], 0.5 * np.eye(2))
    assert np.allclose(error[key], 0.0)


@requires_native
def test_charge_derivative_matches_linear_1d_reference():
    tb = _cosine_band(1)
    mesh = _mesh(tb)
    result = charge(
        mesh,
        mu=0.25,
        options=AdaptiveOptions(target_error=1e-12, max_refinements=0, preview_depth=1),
        refine=False,
    )

    # This is the piecewise-linear derivative on the root preview mesh, not the exact cosine DOS.
    assert result.dcharge_dmu == pytest.approx(1.0 / (2.0 * np.sqrt(2.0)))


@requires_native
@pytest.mark.parametrize("ndim", (2, 3, 4))
def test_charge_derivative_is_finite_and_nonnegative_for_higher_dimensions(ndim):
    tb = _cosine_band(ndim)
    mesh = _mesh(tb)
    result = charge(
        mesh,
        mu=0.0,
        options=AdaptiveOptions(target_error=1e-12, max_refinements=0, preview_depth=1),
        refine=False,
    )

    assert np.isfinite(result.dcharge_dmu)
    assert result.dcharge_dmu >= 0.0


@requires_native
def test_charge_certificate_uses_preview_error_for_stopping():
    mesh = _mesh(_rotating_constant_gap_band())
    options = AdaptiveOptions(target_error=1e-3, max_refinements=0, preview_depth=1)

    result = charge(mesh, mu=0.0, options=options, refine=False)

    assert result.charge == pytest.approx(1.0)
    assert result.charge_error == pytest.approx(0.0)
    assert result.converged


@requires_native
def test_preview_certified_charge_integrates_without_refinement():
    mesh = _mesh(_rotating_constant_gap_band())
    options = AdaptiveOptions(target_error=1e-3, max_refinements=0, preview_depth=1)

    result = charge(mesh, mu=0.0, options=options)

    assert result.charge == pytest.approx(1.0)
    assert result.charge_error == pytest.approx(0.0)
    assert result.converged


@requires_native
def test_occupation_bounded_certificate_error_can_be_within_charge_tolerance():
    mesh = _mesh(_winding_constant_gap_band(3))
    options = AdaptiveOptions(target_error=2.1, max_refinements=0, preview_depth=1)

    result = charge(mesh, mu=0.0, options=options, refine=False)

    assert result.charge == pytest.approx(1.0)
    assert result.charge_error == pytest.approx(2.0)
    assert result.converged


@requires_native
def test_occupation_bounded_certificate_error_can_block_charge_convergence():
    mesh = _mesh(_winding_constant_gap_band(3))
    options = AdaptiveOptions(target_error=1.75, max_refinements=0, preview_depth=1)

    result = charge(mesh, mu=0.0, options=options, refine=False)

    assert result.charge == pytest.approx(1.0)
    assert result.charge_error == pytest.approx(2.0)
    assert not result.converged


@requires_native
def test_uncertified_charge_integration_suppresses_certificate_error():
    mesh = _mesh(_winding_constant_gap_band(3))
    options = AdaptiveOptions(target_error=1.75, max_refinements=0, preview_depth=2)

    certified = charge(mesh, mu=0.0, options=options, refine=False)
    uncertified = charge(mesh, mu=0.0, options=options, refine=False, certify=False)

    assert certified.charge == pytest.approx(uncertified.charge)
    assert certified.charge_error == pytest.approx(2.0)
    assert not certified.converged
    assert uncertified.charge_error == pytest.approx(0.0)
    assert uncertified.converged


@requires_native
def test_uncertified_charge_integration_does_not_refine_active_mesh():
    mesh = _mesh(_winding_constant_gap_band(3))
    options = AdaptiveOptions(target_error=0.75, max_refinements=0, preview_depth=1)

    active_before = mesh.n_active_simplices
    result = charge(mesh, mu=0.0, options=options, refine=False, certify=False)
    active_after = mesh.n_active_simplices

    assert result.refinements == 0
    assert active_after == active_before


@requires_native
def test_integrate_charge_remains_certified():
    mesh = _mesh(_winding_constant_gap_band(3))
    options = AdaptiveOptions(target_error=0.75, max_refinements=0, preview_depth=1)

    with pytest.raises(RuntimeError, match="did not converge"):
        charge(mesh, mu=0.0, options=options)


@requires_native
def test_inertia_certificate_repeated_charge_evaluation_is_deterministic():
    mesh = _mesh(_axis_cosine_band(1))
    options = AdaptiveOptions(target_error=1e-3, max_refinements=0, preview_depth=2)

    first = charge(mesh, mu=0.0, options=options, refine=False)
    second = charge(mesh, mu=0.0, options=options, refine=False)

    assert second.charge == pytest.approx(first.charge)
    assert second.charge_error == pytest.approx(first.charge_error)
    assert second.dcharge_dmu == pytest.approx(first.dcharge_dmu)
    assert second.converged == first.converged


@requires_native
def test_inertia_certificate_does_not_add_error_for_visible_charge_cut():
    mesh = _mesh(_axis_cosine_band(1))
    options = AdaptiveOptions(target_error=1e-3, max_refinements=0, preview_depth=1)

    result = charge(mesh, mu=0.0, options=options, refine=False)

    assert result.charge_error == pytest.approx(0.0)


@requires_native
def test_visible_charge_cut_uses_hessian_energy_shell_error():
    mesh = _mesh(_axis_cosine_band(1))
    options = AdaptiveOptions(target_error=1.0, max_refinements=0, preview_depth=1)

    result = charge(mesh, mu=0.0, options=options, refine=False, hessian_bound=1.0e6)

    assert result.charge_error > 0.0


@requires_native
def test_inertia_certificate_does_not_add_error_for_certified_gapped_simplex():
    mesh = _mesh(_constant_insulator(1))
    options = AdaptiveOptions(target_error=1e-3, max_refinements=0, preview_depth=1)

    result = charge(mesh, mu=0.0, options=options, refine=False)

    assert result.charge_error == pytest.approx(0.0)


@requires_native
def test_charge_hessian_bound_is_validated_and_can_increase_error():
    mesh = _mesh(_constant_insulator(1))
    options = AdaptiveOptions(target_error=3.0, max_refinements=0, preview_depth=1)

    baseline = charge(mesh, mu=0.0, options=options, refine=False, hessian_bound=0.0)
    strict = charge(mesh, mu=0.0, options=options, refine=False, hessian_bound=1.0e6)

    assert baseline.charge_error == pytest.approx(0.0)
    assert strict.charge_error >= baseline.charge_error
    assert strict.charge_error > 0.0

    with pytest.raises(ValueError, match="hessian_bound"):
        charge(mesh, mu=0.0, options=options, refine=False, hessian_bound=-1e-3)
    with pytest.raises(ValueError, match="hessian_bound"):
        charge(mesh, mu=0.0, options=options, refine=False, hessian_bound=float("nan"))
    with pytest.raises(ValueError, match="anharmonicity_bound"):
        charge(mesh, mu=0.0, options=options, refine=False, anharmonicity_bound=-1e-3)
    with pytest.raises(RuntimeError, match="hessian_bound callable"):
        charge(mesh, mu=0.0, options=options, refine=False, hessian_bound=lambda k0: -1.0)


@requires_native
def test_charge_callable_hessian_and_anharmonicity_bounds_increase_error():
    mesh = _mesh(_constant_insulator(1))
    options = AdaptiveOptions(target_error=3.0, max_refinements=0, preview_depth=1)
    calls = []

    def hessian_bound(k0):
        calls.append(k0)
        return 1.0e6

    callable_result = charge(mesh, mu=0.0, options=options, refine=False, hessian_bound=hessian_bound)
    anharmonic_result = charge(
        mesh,
        mu=0.0,
        options=options,
        refine=False,
        anharmonicity_bound=1.0e6,
    )

    assert calls
    assert callable_result.charge_error > 0.0
    assert anharmonic_result.charge_error > 0.0


@requires_native
def test_charge_accepts_native_tight_binding_hessian_bound_callable():
    tb = {
        (1,): np.array([[5.0e5]], dtype=complex),
        (-1,): np.array([[5.0e5]], dtype=complex),
    }
    mesh = _mesh(tb)
    options = AdaptiveOptions(target_error=2.0, max_refinements=0, preview_depth=1)

    result = charge(
        mesh,
        mu=0.0,
        options=options,
        refine=False,
        hessian_bound=tight_binding_hessian_bound(tb),
    )

    assert result.charge_error > 0.0


@requires_native
def test_selected_density_components_match_full_density_slice():
    tb = dimerized_chain()
    keys = [(0,), (1,), (-1,)]
    selected_entries = [(0, 0, (0,)), (1, 1, (0,)), (0, 1, (1,))]
    selected_components = [(0, 0), (1, 1), (0, 1)]
    mesh = _mesh(tb)
    full_rho, _full_error, _full_info = _density_dict(
        mesh,
        mu=0.0,
        keys=keys,
        target_error=1e-3,
        max_refinements=80,
    )
    selected = density_matrix_components(
        mesh,
        mu=0.0,
        keys=keys,
        components=selected_components,
        target_error=1e-3,
        max_refinements=80,
    )

    for row, col, key in selected_entries:
        key_index = selected.keys.index(key)
        component_index = selected.components.index((row, col))
        assert selected.values[key_index, component_index] == pytest.approx(full_rho[key][row, col])


@requires_native
def test_fixed_mu_density_matches_dense_reference():
    tb = dimerized_chain()
    keys = [(0,), (1,), (-1,)]
    reference = dense_reference(tb, mu=0.0, keys=keys, nk=2001)
    mesh = _mesh(tb)
    rho, error, info = _density_dict(
        mesh,
        mu=0.0,
        keys=keys,
        target_error=5e-3,
        max_refinements=512,
    )

    assert max_density_error(rho, reference.rho) <= 5e-3
    assert max(float(np.max(np.abs(block))) for block in error.values()) <= 5e-3
    assert info.n_cached_nodes > 0


@requires_native
def test_charge_current_mesh_integration_does_not_change_active_simplex_count():
    tb = dimerized_chain()
    mesh = _mesh(tb)
    charge(
        mesh,
        mu=0.0,
        options=AdaptiveOptions(target_error=2e-3, max_refinements=64),
    )
    options = AdaptiveOptions(target_error=2e-3, max_refinements=0, preview_depth=2)

    active_before = mesh.n_active_simplices
    values = [
        charge(mesh, mu=mu, options=options, refine=False)
        for mu in (-1.0, 0.0, 1.0, 0.25)
    ]
    active_after = mesh.n_active_simplices

    assert active_after == active_before
    assert all(np.isfinite(value.charge) for value in values)
    assert all(value.charge_error >= 0.0 for value in values)


@requires_native
def test_cached_density_initial_path_reuses_charge_vertices():
    tb = dimerized_chain()
    keys = [(0,), (1,), (-1,)]
    mesh = _mesh(tb)
    charge(
        mesh,
        mu=0.0,
        options=AdaptiveOptions(target_error=2e-3, max_refinements=64, preview_depth=2),
    )
    cached_before = mesh.n_cached_nodes

    rho, error, info = _density_dict(
        mesh,
        mu=0.0,
        keys=keys,
        target_error=1.0,
        options=AdaptiveOptions(target_error=1.0, max_refinements=0, preview_depth=1),
    )

    assert info.n_evaluator_evals == 0
    assert info.subdivisions == 0
    assert mesh.n_cached_nodes == cached_before
    assert set(rho) == set(keys)
    assert set(error) == set(keys)
    assert max(float(np.max(np.abs(block))) for block in error.values()) <= 1.0


@requires_native
def test_density_tight_target_refines():
    tb = dimerized_chain()
    keys = [(0,), (1,), (-1,)]
    mesh = _mesh(tb)
    charge(
        mesh,
        mu=0.0,
        options=AdaptiveOptions(target_error=2e-3, max_refinements=64, preview_depth=2),
    )

    rho, error, info = _density_dict(
        mesh,
        mu=0.0,
        keys=keys,
        target_error=1e-3,
        options=AdaptiveOptions(target_error=1e-3, max_refinements=64, preview_depth=2),
    )

    assert info.subdivisions > 0
    assert set(rho) == set(keys)
    assert set(error) == set(keys)
    assert max(float(np.max(np.abs(block))) for block in error.values()) <= 1e-3


@requires_native
def test_charge_current_mesh_integration_can_be_wrapped_for_solver_callbacks():
    tb = dimerized_chain()
    mesh = _mesh(tb)
    charge(
        mesh,
        mu=0.0,
        options=AdaptiveOptions(target_error=2e-3, max_refinements=64),
    )
    options = AdaptiveOptions(target_error=2e-3, max_refinements=0, preview_depth=2)

    def evaluate(mu):
        result = charge(mesh, mu=mu, options=options, refine=False)
        return result.charge, result.charge_error, result.dcharge_dmu

    charge_value, charge_error, derivative = evaluate(0.0)

    assert charge_value == pytest.approx(1.0, abs=2e-3)
    assert charge_error <= 2e-3
    assert derivative >= 0.0
