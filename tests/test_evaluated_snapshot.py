from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from fractions import Fraction
from itertools import combinations
import math

import numpy as np
import pytest

from fermisimplex import EvaluatedSnapshot, SpectralMesh
from .helpers import constant_insulator, dimerized_chain


def density(mesh, depth):
    return mesh.integrate_density_matrix(
        mu=0.0, lattice_vectors=[(0,) * mesh.ndim], target_error=10.0,
        max_refinements=0, preview_depth=depth,
    )


def exact_points(snapshot):
    return [
        tuple(Fraction(int(n), 1 << int(level)) for n in numerator)
        for numerator, level in zip(snapshot.dyadic_numerators, snapshot.dyadic_levels)
    ]


def average(points):
    return tuple(sum(axis) / len(points) for axis in zip(*points))


@pytest.mark.parametrize('dimension', [1, 2, 3, 4])
@pytest.mark.parametrize('depth', [0, 1, 3])
def test_complete_preview_cache_and_partition(dimension, depth):
    mesh = SpectralMesh(constant_insulator(dimension), root_level=0)
    density(mesh, depth)
    snapshot = mesh.evaluated_snapshot()
    assert isinstance(snapshot, EvaluatedSnapshot)
    assert snapshot.ndim == dimension
    assert snapshot.ndof == mesh.ndof
    assert snapshot.cached_vertices == mesh.cached_vertices
    assert snapshot.active_vertices == mesh.active_vertices
    assert snapshot.active_simplices == mesh.active_simplices
    assert snapshot.partition_simplices == mesh.active_simplices * 2**depth
    assert snapshot.preview_vertices == mesh.cached_vertices - mesh.active_vertices
    assert snapshot.partition_vertices == snapshot.cached_vertices
    assert snapshot.vertex_is_active.sum() == mesh.active_vertices
    assert np.all(snapshot.simplex_is_active == (depth == 0))
    assert np.all(np.diff(snapshot.vertex_ids) > 0)
    assert np.unique(snapshot.simplex_ids).size == snapshot.partition_simplices
    assert np.min(snapshot.simplices) >= 0
    assert np.max(snapshot.simplices) < snapshot.cached_vertices
    assert snapshot.simplices.shape == (snapshot.partition_simplices, dimension + 1)
    assert snapshot.volumes.sum() == pytest.approx(1.0, abs=1e-12)
    # Independent geometric volume check against exported connectivity.
    cells = snapshot.points[snapshot.simplices]
    volumes = np.abs(np.linalg.det(cells[:, 1:] - cells[:, :1])) / math.factorial(dimension)
    np.testing.assert_allclose(snapshot.volumes, volumes, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(np.asarray(exact_points(snapshot), dtype=float), snapshot.points)
    np.testing.assert_array_equal(snapshot.points[snapshot.vertex_is_active], mesh.points)
    np.testing.assert_array_equal(snapshot.eigenvalues[snapshot.vertex_is_active], mesh.eigenvalues)
    # Constant diagonal Hamiltonian has exact eigenvalues and projectors everywhere.
    hamiltonian = constant_insulator(dimension)[(0,) * dimension]
    reconstructed = (
        snapshot.eigenvectors * snapshot.eigenvalues[:, None, :]
    ) @ snapshot.eigenvectors.conj().transpose(0, 2, 1)
    np.testing.assert_allclose(
        reconstructed, np.broadcast_to(hamiltonian, reconstructed.shape), atol=1e-12
    )


def test_snapshot_never_evaluates_and_owns_readonly_arrays():
    calls = []

    def hamiltonian(k):
        calls.append(k)
        return np.array([[k - 2, 0.25j], [-0.25j, k + 2]], dtype=complex)

    mesh = SpectralMesh(hamiltonian, root_level=0)
    before = len(calls)
    with pytest.raises(RuntimeError, match='no complete evaluated covering'):
        mesh.evaluated_snapshot()
    assert len(calls) == before
    density(mesh, 3)
    before = len(calls)
    counts = mesh.cached_vertices, mesh.active_vertices, mesh.active_simplices
    active_points, active_cells = mesh.points, mesh.simplices
    snapshot = mesh.evaluated_snapshot()
    again = mesh.evaluated_snapshot()
    values_only = mesh.evaluated_snapshot(include_eigenvectors=False)
    assert values_only.eigenvectors is None
    assert len(calls) == before
    assert counts == (mesh.cached_vertices, mesh.active_vertices, mesh.active_simplices)
    np.testing.assert_array_equal(mesh.points, active_points)
    np.testing.assert_array_equal(mesh.simplices, active_cells)
    assert set(calls[1:]) == set(snapshot.points[:, 0])  # constructor probes the origin once
    assert before - 1 == snapshot.cached_vertices
    for field in fields(snapshot):
        a, b = getattr(snapshot, field.name), getattr(again, field.name)
        np.testing.assert_array_equal(a, b)
        if isinstance(a, np.ndarray):
            assert not a.flags.writeable
            with pytest.raises(ValueError, match='read-only'):
                a.flat[0] = 0
    with pytest.raises(FrozenInstanceError):
        snapshot.ndim = 2
    matrices = np.array([[[k - 2, 0.25j], [-0.25j, k + 2]] for k in snapshot.points[:, 0]])
    np.testing.assert_allclose(snapshot.eigenvalues, np.linalg.eigvalsh(matrices), atol=1e-12)
    np.testing.assert_allclose(
        matrices @ snapshot.eigenvectors,
        snapshot.eigenvectors * snapshot.eigenvalues[:, None, :], atol=1e-12,
    )
    saved = snapshot.points.copy()
    saved_ids = snapshot.vertex_ids.copy()
    density(mesh, 4)
    expanded = mesh.evaluated_snapshot()
    np.testing.assert_array_equal(expanded.vertex_ids[:len(saved_ids)], saved_ids)
    np.testing.assert_array_equal(expanded.points[:len(saved_ids)], saved)
    del mesh
    np.testing.assert_array_equal(snapshot.points, saved)


def test_export_preserves_subsequent_adaptive_integration():
    exported = SpectralMesh(dimerized_chain(), root_level=1)
    control = SpectralMesh(dimerized_chain(), root_level=1)
    density(exported, 1)
    density(control, 1)
    for _ in range(3):
        exported.evaluated_snapshot()
    kwargs = dict(mu=0.0, lattice_vectors=[(0,), (1,)], target_error=1e-3,
                  max_refinements=4096, preview_depth=2,
                  min_refinement_batch_size=1, max_refinement_batch_size=1)
    a = exported.integrate_density_matrix(**kwargs)
    b = control.integrate_density_matrix(**kwargs)
    np.testing.assert_array_equal(a.matrices, b.matrices)
    assert exported.cached_vertices == control.cached_vertices
    assert exported.active_simplices == control.active_simplices
    exported_snapshot = exported.evaluated_snapshot()
    control_snapshot = control.evaluated_snapshot()
    for field in fields(exported_snapshot):
        np.testing.assert_array_equal(
            getattr(exported_snapshot, field.name),
            getattr(control_snapshot, field.name),
        )


def test_exact_midpoint_centroid_dedup_and_quadrature():
    calls = []

    def hamiltonian(k):
        calls.append(k)
        return np.array([[k*k - 2]], dtype=complex)

    mesh = SpectralMesh(hamiltonian, root_level=0)
    density(mesh, 2)
    snapshot = mesh.evaluated_snapshot(include_eigenvectors=False)
    keys = exact_points(snapshot)
    spectra = dict(zip(keys, snapshot.eigenvalues[:, 0]))
    # Two- and three-node rules in 1D share endpoints and the
    # midpoint/centroid. Form all requests for both rules before evaluating.
    requests = []
    for indices in snapshot.simplices:
        vertices = [keys[int(i)] for i in indices]
        requests.extend(vertices)
        requests.extend(average(edge) for edge in combinations(vertices, 2))
        requests.append(average(vertices))
    unique = set(requests)
    missing = sorted(unique - spectra.keys())
    preview_keys = {key for key, active in zip(keys, snapshot.vertex_is_active) if not active}
    assert len(requests) == 16
    assert len(unique) == 9
    assert len(unique & spectra.keys()) == 5
    assert len(unique & preview_keys) == 3
    assert len(missing) == 4
    assert len(missing) / snapshot.cached_vertices == pytest.approx(4 / 5)
    before = len(calls)
    matrices = np.stack([mesh.evaluate(float(key[0])) for key in missing])
    new_values = np.linalg.eigvalsh(matrices)[:, 0]  # one batch of missing spectra
    spectra.update(zip(missing, new_values))
    assert len(calls) - before == 4
    trapezoid, simpson = 0.0, 0.0
    for indices, volume in zip(snapshot.simplices, snapshot.volumes):
        a, b = (keys[int(i)] for i in indices)
        mid = average([a, b])
        trapezoid += volume * (spectra[a] + spectra[b]) / 2
        simpson += volume * (spectra[a] + 4*spectra[mid] + spectra[b]) / 6
    exact = 1/3 - 2
    # For x^2, composite trapezoid error is h^2/6; Simpson is exact.
    assert trapezoid - exact == pytest.approx(1/96, abs=1e-12)
    assert simpson == pytest.approx(exact, abs=1e-12)
    print(f'Trapezoid error={abs(trapezoid-exact):.3g}; Simpson error={abs(simpson-exact):.3g}')


def test_shared_triangle_midpoints_and_cache_hits():
    mesh = SpectralMesh(constant_insulator(2), root_level=0)
    density(mesh, 1)
    snapshot = mesh.evaluated_snapshot(include_eigenvectors=False)
    keys = exact_points(snapshot)
    requests = []
    for indices in snapshot.simplices:
        vertices = [keys[int(i)] for i in indices]
        requests.extend(vertices)
        requests.extend(average(edge) for edge in combinations(vertices, 2))
        requests.append(average(vertices))
    unique = set(requests)
    assert len(requests) == 28
    assert len(unique) == 17
    assert len(unique & set(keys)) == 5
    assert len(unique - set(keys)) == 12
    assert (Fraction(1, 2), Fraction(1, 2)) in unique & set(keys)


def test_nonuniform_midpoints_reuse_preview_spectra():
    mesh = SpectralMesh(constant_insulator(2), root_level=0)
    mesh.integrate_density_matrix(
        mu=0.0, lattice_vectors=[(1, 0)], target_error=0.1,
        max_refinements=4096, preview_depth=1,
        min_refinement_batch_size=1, max_refinement_batch_size=1,
    )
    snapshot = mesh.evaluated_snapshot(include_eigenvectors=False)
    keys = exact_points(snapshot)
    requests = []
    for indices in snapshot.simplices:
        vertices = [keys[int(i)] for i in indices]
        requests.extend(average(edge) for edge in combinations(vertices, 2))
        requests.append(average(vertices))
    unique = set(requests)
    hits = unique & set(keys)
    preview_keys = {key for key, active in zip(keys, snapshot.vertex_is_active) if not active}
    # Here the requests contain only edge midpoints and centroids, not vertices.
    # Nonuniform neighbouring cells share midpoints with cached preview vertices.
    assert len(requests) == 48
    assert len(unique) == 40
    assert len(hits) == 4
    assert hits == hits & preview_keys
    assert len(unique - set(keys)) == 36
    assert len(set(snapshot.volumes)) > 1


def test_shallower_later_request_keeps_deepest_retained_preview():
    mesh = SpectralMesh(constant_insulator(2), root_level=0)
    density(mesh, 3)
    before = mesh.evaluated_snapshot()
    density(mesh, 1)
    after = mesh.evaluated_snapshot()
    for field in fields(before):
        np.testing.assert_array_equal(
            getattr(before, field.name), getattr(after, field.name)
        )
