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
    assert result.stats.cubature_evaluations == (
        0 if expected == 0 else n * (ndim + 2)
    )


def test_half_occupation_and_duplicate_components():
    mesh = SpectralMesh({(0,): np.zeros((2, 2))})
    result = integrate(mesh, components=[[0, 0, 0], [0, 0, 0], [0, 1, 0]])
    np.testing.assert_allclose(result.values, [0.5, 0.5, 0], atol=1e-14)


def test_initial_and_promoted_errors_compare_degrees_two_apart():
    mesh = SpectralMesh({(0,): np.array([[-1.0]])}, root_level=0)
    request = dict(keys=[(1,)], components=[[0, 0, 0]], target_error=0)
    q3 = integrate(mesh, max_degree=3, **request)
    q5 = integrate(mesh, max_degree=5, **request)
    default = integrate(mesh, **request)
    assert default.stats.max_degree == 7
    assert q3.stats.max_degree == 3
    assert q3.stats.p_refinements == 0
    assert q3.values[0] == pytest.approx(1 / 3)
    assert q3.stopping_error == pytest.approx(2 / 3)  # Q3 - Q1
    assert q5.stats.max_degree == 5
    assert q5.stats.p_refinements == 1
    assert q5.stopping_error == pytest.approx(abs(q5.values[0] - q3.values[0]))


def test_nested_samples_and_fourier_phase():
    calls = []

    def model(k):
        calls.append(k)
        return np.diag([-1.0, 1.0]).astype(complex)

    mesh = SpectralMesh(model)
    mesh.estimate_charge_on_current_mesh(mu=0.0)
    calls.clear()
    before = mesh.points.copy(), mesh.simplices.copy(), mesh.cached_vertices
    result = integrate(mesh, keys=[(0,), (1,), (-1,)], target_error=1e-9, max_degree=21)
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
    result = integrate(mesh, keys=[(0,), (1,), (-1,)], target_error=1e-8, max_degree=21)
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
    "budget", [dict(max_degree=3), dict(max_refinements=0), dict(max_refinements=1)]
)
def test_exhaustion_does_not_claim_convergence(budget):
    mesh = SpectralMesh({(0,): np.array([[-1.0]])})
    result = integrate(mesh, keys=[(1,)], target_error=1e-12, **budget)
    assert not result.stats.target_reached
    assert result.stopping_error > 1e-12
    assert result.stats.refinements == 0
    if "max_refinements" in budget:
        assert result.stats.p_refinements <= budget["max_refinements"]


@pytest.mark.parametrize("degree", [0, 1, 2, 4, 22, 23, 2.5])
def test_degree_validation(degree):
    mesh = SpectralMesh({(0,): np.array([[-1.0]])})
    with pytest.raises((ValueError, TypeError), match="max_degree"):
        integrate(mesh, max_degree=degree)


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_linear_projector_components_are_integrated_exactly_on_cuts(ndim):
    def h(k):
        # Lower energy k, with affine diagonal projector components.
        g = 0.2 + 0.4 * k
        x = 2 * np.sqrt(g * (1 - g))
        return np.array([[2 * g - 1, x], [x, 1 - 2 * g]]) + (k + 1) * np.eye(2)

    def h1(k):
        return h(k)

    def h2(k, y):
        return h(k)

    def h3(k, y, z):
        return h(k)

    mesh = SpectralMesh([h1, h2, h3][ndim - 1])
    mu = 0.37
    mesh.estimate_charge_on_current_mesh(mu=mu)
    before = mesh.cached_vertices
    result = integrate(
        mesh, mu=mu, components=[[0, 0, 0], [0, 1, 1]], target_error=1e-12
    )
    np.testing.assert_allclose(
        result.values, [0.8 * mu - 0.2 * mu**2, 0.2 * mu + 0.2 * mu**2], atol=2e-14
    )
    assert result.stats.target_reached
    assert result.stats.p_refinements == 0
    assert mesh.cached_vertices == before
    occupied_cells = np.any(mesh.eigenvalues[mesh.simplices, 0] < mu, axis=1)
    assert result.stats.evaluations == (ndim + 2) * np.count_nonzero(occupied_cells)


def test_cut_correction_has_no_new_samples_and_removes_linear_error():
    def model(k):
        return np.array([[k]], complex)

    errors = []
    for level in [2, 3, 4, 5]:
        mesh = SpectralMesh(model, root_level=level)
        h = 2.0 ** (-level)
        mu = 0.37 * h
        charge = mesh.estimate_charge_on_current_mesh(mu=mu)
        assert charge.value == pytest.approx(mu)
        result = integrate(mesh, mu=mu, keys=[(1,)], target_error=1e-12, max_degree=21)
        exact = np.expm1(2j * np.pi * mu) / (2j * np.pi)
        frozen = 0.37 * np.expm1(2j * np.pi * h) / (2j * np.pi)
        m1 = mu**2 / (2 * h)
        correction = (m1 - mu / 2) * np.expm1(2j * np.pi * h)
        assert result.stats.target_reached
        np.testing.assert_allclose(result.values[0], frozen + correction, atol=1e-12)
        errors.append(abs(result.values[0] - exact))
        assert errors[-1] < abs(frozen - exact)
        assert result.stats.refinements == 0
    # The cut-cell residual is now O(h^3), rather than O(h^2).
    assert errors[-1] < errors[-2] / 7
    # Higher-order occupation error remains outside the cubature estimate.
    assert errors[-1] > result.stopping_error


def test_parallel_large_matrix_is_deterministic_and_respects_budget():
    threadpoolctl = pytest.importorskip("threadpoolctl")
    hopping = np.array([[0, 1], [0, 0]], complex)
    tb = {
        (1,): np.kron(np.eye(16), hopping),
        (-1,): np.kron(np.eye(16), hopping.conj().T),
    }
    mesh = SpectralMesh(tb)
    mesh.estimate_charge_on_current_mesh(mu=0.0)
    geometry = mesh.simplices.copy()
    request = dict(
        keys=[(0,), (1,)],
        components=[[0, 0, 0], [1, 0, 1], [1, 0, 0]],
        target_error=1e-9,
        max_degree=21,
    )
    with threadpoolctl.threadpool_limits(limits=1):
        serial = integrate(mesh, **request)
    with threadpoolctl.threadpool_limits(limits=4, user_api="openmp"):
        parallel = integrate(mesh, **request)
        repeat = integrate(mesh, **request)
        limited = integrate(mesh, **{**request, "max_refinements": 1})
    assert serial.stats.target_reached and parallel.stats.target_reached
    np.testing.assert_allclose(serial.values, [0.5, -0.5, 0.0], atol=1e-9)
    np.testing.assert_allclose(parallel.values, serial.values, atol=1e-9)
    np.testing.assert_array_equal(parallel.values, repeat.values)
    assert parallel.stats.evaluations == repeat.stats.evaluations
    assert limited.stats.p_refinements == 1
    assert not limited.stats.target_reached
    np.testing.assert_array_equal(mesh.simplices, geometry)


def test_parallel_callback_exception_is_propagated():
    threadpoolctl = pytest.importorskip("threadpoolctl")

    def model(k):
        if abs(k - 0.125) < 1e-12:
            raise ValueError("interior sample failed")
        return -np.eye(32, dtype=complex)

    mesh = SpectralMesh(model)
    mesh.estimate_charge_on_current_mesh(mu=0.0)
    with threadpoolctl.threadpool_limits(limits=4, user_api="openmp"):
        with pytest.raises(ValueError, match="interior sample failed"):
            integrate(mesh, keys=[(1,)], components=[[0, 0, 0]], target_error=1e-9)


@pytest.mark.parametrize("mass", [3.0, 1.3])
def test_hp_fallback_resolves_bulk_degree_cap_without_changing_charge_mesh(mass):
    # A rotated and shifted gapped projector defeats degree 21 on this coarse
    # charge mesh. A denser, independently converged integral is the reference.
    sigma_x = np.array([[0, 1], [1, 0]], complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], complex)
    sigma_z = np.diag([1, -1]).astype(complex)
    tb = {(0, 0): mass * sigma_z}
    keys = [(0, 0), (1, 0), (0, 1)]
    for axis, sigma in enumerate((sigma_x, sigma_y)):
        key = keys[axis + 1]
        hopping = (sigma_z + 1j * sigma) / 2
        tb[key] = hopping
        tb[tuple(-v for v in key)] = hopping.conj().T
    rng = np.random.default_rng(2)
    rotation, _ = np.linalg.qr(
        rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    )
    shift = rng.uniform(0, 1, 2)
    tb = {
        key: np.exp(-2j * np.pi * np.dot(key, shift))
        * (rotation @ hopping @ rotation.conj().T)
        for key, hopping in tb.items()
    }
    request = dict(keys=keys, target_error=1e-5, max_degree=21)
    coarse = SpectralMesh(tb, root_level=1)
    charge = coarse.integrate_charge(mu=0.0, target_error=1e-4)
    assert charge.stats.target_reached
    before = coarse.points.copy(), coarse.simplices.copy(), coarse.cached_vertices
    p_only = integrate(coarse, **request)
    hp = integrate(coarse, **request, max_h_refinements=32)
    fine = SpectralMesh(tb, root_level=3)
    reference = integrate(fine, keys=keys, target_error=1e-9, max_degree=21)
    finer = SpectralMesh(tb, root_level=4)
    reference_check = integrate(finer, keys=keys, target_error=1e-10, max_degree=21)
    assert reference.stats.target_reached and reference_check.stats.target_reached
    np.testing.assert_allclose(reference.values, reference_check.values, atol=1e-8)
    assert not p_only.stats.target_reached
    assert hp.stats.target_reached
    assert hp.stats.refinements > 0
    assert hp.stats.p_refinements > 0
    if mass == 3.0:
        assert hp.stats.evaluations < p_only.stats.evaluations
    np.testing.assert_allclose(hp.values, reference.values, atol=1e-5)
    np.testing.assert_array_equal(coarse.points, before[0])
    np.testing.assert_array_equal(coarse.simplices, before[1])
    assert coarse.cached_vertices == before[2]


def test_hp_cut_bisection_preserves_charge_stage_occupation():
    mesh = SpectralMesh(lambda k: np.array([[k]], complex), root_level=1)
    charge = mesh.estimate_charge_on_current_mesh(mu=0.37)
    before = mesh.points.copy(), mesh.simplices.copy(), mesh.cached_vertices
    result = integrate(
        mesh,
        mu=0.37,
        keys=[(0,), (1,)],
        components=[[0, 0, 0], [1, 0, 0]],
        target_error=1e-5,
        max_degree=3,
        max_h_refinements=300,
    )
    assert result.stats.target_reached
    assert result.stats.refinements > 0
    assert result.values[0] == pytest.approx(charge.value, abs=1e-13)
    exact = np.expm1(2j * np.pi * 0.37) / (2j * np.pi)
    assert abs(result.values[1] - exact) < 1e-5
    np.testing.assert_array_equal(mesh.points, before[0])
    np.testing.assert_array_equal(mesh.simplices, before[1])
    assert mesh.cached_vertices == before[2]


def test_hp_split_budget_reports_exhaustion():
    mesh = SpectralMesh(lambda k: np.array([[k]], complex), root_level=1)
    result = integrate(
        mesh,
        mu=0.37,
        keys=[(1,)],
        components=[[0, 0, 0]],
        target_error=1e-10,
        max_degree=3,
        max_h_refinements=1,
    )
    assert result.stats.refinements == 1
    assert not result.stats.target_reached
    assert result.stopping_error > 1e-10
