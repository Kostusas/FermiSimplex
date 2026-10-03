"""Retained spectral export is independent of transient density cubature."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from fractions import Fraction
from itertools import combinations
import math

import numpy as np
import pytest

from fermisimplex import EvaluatedSnapshot, SpectralMesh
from .helpers import constant_insulator, dimerized_chain


def exact_points(snapshot):
    return [
        tuple(Fraction(int(n), 1 << int(level)) for n in numerator)
        for numerator, level in zip(snapshot.dyadic_numerators, snapshot.dyadic_levels)
    ]


def average(points):
    return tuple(sum(axis) / len(points) for axis in zip(*points))


def assert_same_snapshot(a, b):
    for field in fields(a):
        np.testing.assert_array_equal(getattr(a, field.name), getattr(b, field.name))


@pytest.mark.parametrize("dimension", [1, 2, 3, 4])
@pytest.mark.parametrize("level", [0, 1])
def test_retained_cache_and_partition(dimension, level):
    mesh = SpectralMesh(constant_insulator(dimension), root_level=level)
    mesh.estimate_charge_on_current_mesh(mu=0)
    snapshot = mesh.evaluated_snapshot()
    assert isinstance(snapshot, EvaluatedSnapshot)
    assert snapshot.ndim == dimension
    assert snapshot.ndof == mesh.ndof
    assert snapshot.cached_vertices == mesh.cached_vertices
    assert snapshot.active_vertices == mesh.active_vertices
    assert snapshot.active_simplices == mesh.active_simplices
    assert snapshot.partition_simplices == mesh.active_simplices
    assert snapshot.preview_vertices == 0
    assert snapshot.partition_vertices == snapshot.cached_vertices
    assert snapshot.vertex_is_active.sum() == mesh.active_vertices
    assert np.all(snapshot.simplex_is_active)
    assert np.all(np.diff(snapshot.vertex_ids) > 0)
    assert np.unique(snapshot.simplex_ids).size == snapshot.partition_simplices
    assert np.min(snapshot.simplices) >= 0
    assert np.max(snapshot.simplices) < snapshot.cached_vertices
    assert snapshot.simplices.shape == (snapshot.partition_simplices, dimension + 1)
    assert snapshot.volumes.sum() == pytest.approx(1, abs=1e-12)
    cells = snapshot.points[snapshot.simplices]
    volumes = np.abs(np.linalg.det(cells[:, 1:] - cells[:, :1])) / math.factorial(
        dimension
    )
    np.testing.assert_allclose(snapshot.volumes, volumes, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(
        np.asarray(exact_points(snapshot), dtype=float), snapshot.points
    )
    np.testing.assert_array_equal(
        snapshot.points[snapshot.vertex_is_active], mesh.points
    )
    np.testing.assert_array_equal(
        snapshot.eigenvalues[snapshot.vertex_is_active], mesh.eigenvalues
    )
    hamiltonian = constant_insulator(dimension)[(0,) * dimension]
    reconstructed = (
        snapshot.eigenvectors * snapshot.eigenvalues[:, None, :]
    ) @ snapshot.eigenvectors.conj().transpose(0, 2, 1)
    np.testing.assert_allclose(
        reconstructed,
        np.broadcast_to(hamiltonian, reconstructed.shape),
        atol=1e-12,
    )


def test_export_is_read_only_evaluation_free_and_survives_refinement():
    calls = []

    def hamiltonian(k):
        calls.append(k)
        return np.array([[k * k - 0.3, 0.25j], [-0.25j, k * k + 2]], complex)

    mesh = SpectralMesh(hamiltonian, root_level=0)
    before = len(calls)
    with pytest.raises(RuntimeError, match="no complete evaluated covering"):
        mesh.evaluated_snapshot()
    assert len(calls) == before
    mesh.estimate_charge_on_current_mesh(mu=0)
    before = len(calls)
    counts = mesh.cached_vertices, mesh.active_vertices, mesh.active_simplices
    snapshot = mesh.evaluated_snapshot()
    assert_same_snapshot(snapshot, mesh.evaluated_snapshot())
    assert mesh.evaluated_snapshot(include_eigenvectors=False).eigenvectors is None
    assert len(calls) == before
    assert counts == (mesh.cached_vertices, mesh.active_vertices, mesh.active_simplices)
    assert set(calls[1:]) == set(snapshot.points[:, 0])
    for field in fields(snapshot):
        value = getattr(snapshot, field.name)
        if isinstance(value, np.ndarray):
            assert not value.flags.writeable
            with pytest.raises(ValueError, match="read-only"):
                value.flat[0] = 0
    with pytest.raises(FrozenInstanceError):
        snapshot.ndim = 2
    matrices = np.array([hamiltonian(k) for k in snapshot.points[:, 0]])
    np.testing.assert_allclose(
        snapshot.eigenvalues, np.linalg.eigvalsh(matrices), atol=1e-12
    )
    np.testing.assert_allclose(
        matrices @ snapshot.eigenvectors,
        snapshot.eigenvectors * snapshot.eigenvalues[:, None, :],
        atol=1e-12,
    )
    saved = snapshot.points.copy()
    saved_ids = snapshot.vertex_ids.copy()
    mesh.integrate_charge(mu=0, target_error=1e-5, max_refinements=500)
    expanded = mesh.evaluated_snapshot()
    assert len(expanded.points) > len(saved)
    np.testing.assert_array_equal(expanded.vertex_ids[: len(saved_ids)], saved_ids)
    np.testing.assert_array_equal(expanded.points[: len(saved_ids)], saved)
    del mesh
    np.testing.assert_array_equal(snapshot.points, saved)


def test_export_preserves_subsequent_adaptive_integration():
    exported = SpectralMesh(dimerized_chain(), root_level=1)
    control = SpectralMesh(dimerized_chain(), root_level=1)
    for mesh in [exported, control]:
        mesh.integrate_charge(mu=0, target_error=1e-3)
    for _ in range(3):
        exported.evaluated_snapshot()
    request = dict(mu=0.0, lattice_vectors=[(0,), (1,)], target_error=1e-3)
    a = exported.integrate_density_matrix(**request)
    b = control.integrate_density_matrix(**request)
    np.testing.assert_array_equal(a.matrices, b.matrices)
    assert_same_snapshot(exported.evaluated_snapshot(), control.evaluated_snapshot())


def test_exact_midpoint_centroid_dedup_and_quadrature():
    calls = []

    def hamiltonian(k):
        calls.append(k)
        return np.array([[k * k - 2]], complex)

    mesh = SpectralMesh(hamiltonian, root_level=2)
    mesh.estimate_charge_on_current_mesh(mu=0)
    snapshot = mesh.evaluated_snapshot(include_eigenvectors=False)
    keys = exact_points(snapshot)
    spectra = dict(zip(keys, snapshot.eigenvalues[:, 0]))
    requests = []
    for indices in snapshot.simplices:
        vertices = [keys[int(i)] for i in indices]
        requests.extend(vertices)
        requests.extend(average(edge) for edge in combinations(vertices, 2))
        requests.append(average(vertices))
    unique = set(requests)
    missing = sorted(unique - spectra.keys())
    assert len(requests) == 16
    assert len(unique) == 9
    assert len(unique & spectra.keys()) == 5
    assert len(missing) == 4
    before = len(calls)
    matrices = np.stack([mesh.evaluate(float(key[0])) for key in missing])
    spectra.update(zip(missing, np.linalg.eigvalsh(matrices)[:, 0]))
    assert len(calls) - before == 4
    trapezoid, simpson = 0.0, 0.0
    for indices, volume in zip(snapshot.simplices, snapshot.volumes):
        a, b = (keys[int(i)] for i in indices)
        mid = average([a, b])
        trapezoid += volume * (spectra[a] + spectra[b]) / 2
        simpson += volume * (spectra[a] + 4 * spectra[mid] + spectra[b]) / 6
    exact = 1 / 3 - 2
    assert trapezoid - exact == pytest.approx(1 / 96, abs=1e-12)
    assert simpson == pytest.approx(exact, abs=1e-12)


def test_density_subdivision_does_not_enter_retained_snapshot():
    mesh = SpectralMesh(constant_insulator(2), root_level=0)
    mesh.integrate_charge(mu=0, target_error=1e-8)
    before = mesh.evaluated_snapshot()
    density = mesh.integrate_density_matrix(
        mu=0,
        lattice_vectors=[(1, 0)],
        target_error=1e-4,
        max_degree=3,
    )
    assert density.stats.target_reached
    assert density.stats.refinements > 0
    assert density.stats.cubature_evaluations > 0
    np.testing.assert_allclose(density.matrices, 0, atol=1e-4)
    assert_same_snapshot(before, mesh.evaluated_snapshot())
