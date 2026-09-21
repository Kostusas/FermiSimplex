"""Physical and controller regressions for density cubature on a frozen mesh."""

import numpy as np
import pytest
from fermisimplex import SpectralMesh


def integrate(mesh, *, keys=None, components=None, **kwargs):
    keys = [tuple([0] * mesh.ndim)] if keys is None else keys
    components = (
        [
            [k, i, j]
            for k in range(len(keys))
            for i in range(mesh.ndof)
            for j in range(mesh.ndof)
        ]
        if components is None
        else components
    )
    return mesh.integrate_density_components_p(
        mu=kwargs.pop("mu", 0.0),
        lattice_vectors=keys,
        components=components,
        target_error=kwargs.pop("target_error", 1e-7),
        **kwargs,
    )


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("mu,expected", [(-2.0, 0.0), (0.0, 1.0), (2.0, 2.0)])
def test_constant_projector_and_empty_cells(ndim, mu, expected):
    mesh = SpectralMesh({(0,) * ndim: np.diag([-1.0, 1.0])})
    mesh.estimate_charge_on_current_mesh(mu=mu)
    n = mesh.active_simplices
    result = integrate(mesh, mu=mu)
    assert result.stats.target_reached
    assert np.trace(result.values.reshape(2, 2)) == pytest.approx(expected)
    assert result.stats.refinements == result.stats.p_refinements == 0
    assert result.stats.cubature_evaluations == (0 if expected == 0 else n)


def test_half_occupation_and_duplicate_components():
    mesh = SpectralMesh({(0,): np.zeros((2, 2))})
    result = integrate(mesh, components=[[0, 0, 0], [0, 0, 0], [0, 1, 0]])
    np.testing.assert_allclose(result.values, [0.5, 0.5, 0], atol=1e-14)


def test_nested_samples_and_fourier_phase():
    calls = []

    def model(k):
        calls.append(k)
        return np.diag([-1.0, 1.0]).astype(complex)

    mesh = SpectralMesh(model)
    mesh.estimate_charge_on_current_mesh(mu=0.0)
    calls.clear()
    before = mesh.points.copy(), mesh.simplices.copy(), mesh.cached_vertices
    result = integrate(mesh, keys=[(0,), (1,), (-1,)], target_error=1e-9)
    assert result.stats.target_reached
    values = result.values.reshape(3, 2, 2)
    np.testing.assert_allclose(values[0], np.diag([1.0, 0.0]), atol=1e-12)
    np.testing.assert_allclose(values[1:], 0.0, atol=1e-9)
    assert result.stats.p_refinements > 0
    assert len(calls) == len(set(calls)) == result.stats.cubature_evaluations
    np.testing.assert_array_equal(mesh.points, before[0])
    np.testing.assert_array_equal(mesh.simplices, before[1])
    assert mesh.cached_vertices == before[2]
    assert result.stats.refinements == 0


def test_rotating_projector_exact_fourier_coefficients():
    # H(k)=[[0, exp(-2 pi i k)], [exp(2 pi i k), 0]], energies +/-1.
    hopping = np.array([[0.0, 1.0], [0.0, 0.0]], complex)
    mesh = SpectralMesh({(1,): hopping, (-1,): hopping.T.conj()})
    result = integrate(mesh, keys=[(0,), (1,), (-1,)], target_error=1e-8)
    assert result.stats.target_reached
    expected = np.zeros((3, 2, 2), complex)
    expected[0] = np.eye(2) / 2
    # Native TB convention is exp(-2 pi i k R), density uses the inverse.
    expected[1, 0, 1] = -0.5
    expected[2, 1, 0] = -0.5
    np.testing.assert_allclose(result.values.reshape(3, 2, 2), expected, atol=1e-8)


def test_cut_occupation_trace_preserves_charge_without_refinement():
    mesh = SpectralMesh(
        {(0,): np.diag([-0.3, 0.7]), (1,): np.eye(2) * 0.5, (-1,): np.eye(2) * 0.5}
    )
    charge = mesh.integrate_charge(mu=0.0, target_error=1e-4)
    before = mesh.simplices.copy()
    result = integrate(mesh)
    assert result.stats.target_reached
    assert np.trace(result.values.reshape(2, 2)) == pytest.approx(
        charge.value, abs=1e-12
    )
    np.testing.assert_array_equal(mesh.simplices, before)


@pytest.mark.parametrize(
    "budget", [dict(max_degree=2), dict(max_refinements=0), dict(max_refinements=1)]
)
def test_exhaustion_does_not_claim_convergence(budget):
    mesh = SpectralMesh({(0,): np.array([[-1.0]])})
    result = integrate(mesh, keys=[(1,)], target_error=1e-12, **budget)
    assert not result.stats.target_reached
    assert result.stopping_error > 1e-12
    assert result.stats.refinements == 0
    if "max_refinements" in budget:
        assert result.stats.p_refinements <= budget["max_refinements"]


@pytest.mark.parametrize("degree", [0, 1, 4, 22, 23, 2.5])
def test_degree_validation(degree):
    mesh = SpectralMesh({(0,): np.array([[-1.0]])})
    with pytest.raises((ValueError, TypeError), match="max_degree"):
        integrate(mesh, max_degree=degree)


def test_frozen_cut_error_is_separate_from_cubature_error():
    # An affine band gives exact linear charge, but replacing the occupied
    # subinterval by a fractional weight misses its correlation with exp(ik).
    def model(k):
        return np.array([[k]], complex)

    mesh = SpectralMesh(model)
    mu = 0.37
    charge = mesh.estimate_charge_on_current_mesh(mu=mu)
    assert charge.value == pytest.approx(mu)
    result = integrate(mesh, mu=mu, keys=[(1,)], target_error=1e-9)
    exact = np.expm1(2j * np.pi * mu) / (2j * np.pi)
    assert result.stats.target_reached
    assert result.stopping_error < 1e-9
    assert abs(result.values[0] - exact) > 0.05
