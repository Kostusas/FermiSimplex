from __future__ import annotations

import numpy as np
import pytest

from fermisimplex import DensityComponentsResult, SpectralMesh

from .helpers import (
    constant_insulator,
    dense_reference,
    dimerized_chain,
    qiwuzhang,
    tb_k_matrix,
)


@pytest.fixture(params=("tight_binding", "callable"))
def public_hamiltonian(request):
    hoppings = dimerized_chain()
    if request.param == "tight_binding":
        return hoppings

    def hamiltonian(k):
        return tb_k_matrix(hoppings, np.asarray([k]))

    return hamiltonian


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"target_error": -1.0}, "target_error"),
        ({"target_error": np.inf}, "target_error"),
        ({"target_error": np.nan}, "target_error"),
        ({"target_error": 1.0, "max_refinements": -2}, "max_refinements"),
        ({"target_error": 1.0, "error_depth": -1}, "error_depth"),
        (
            {"target_error": 1.0, "min_refinement_batch_size": 0},
            "min_refinement_batch_size",
        ),
        (
            {
                "target_error": 1.0,
                "min_refinement_batch_size": 3,
                "max_refinement_batch_size": 2,
            },
            "max_refinement_batch_size",
        ),
    ),
)
def test_adaptive_arguments_reject_invalid_values(kwargs, message):
    mesh = SpectralMesh(constant_insulator(1))
    with pytest.raises(ValueError, match=message):
        mesh.integrate_charge(mu=0.0, **kwargs)


@pytest.mark.parametrize("mu", (np.nan, np.inf, -np.inf))
def test_integration_requires_a_finite_chemical_potential(mu):
    mesh = SpectralMesh(constant_insulator(1))

    with pytest.raises(ValueError, match="mu must be finite"):
        mesh.integrate_charge(mu=mu, target_error=1.0, max_refinements=0)
    with pytest.raises(ValueError, match="mu must be finite"):
        mesh.estimate_charge_on_current_mesh(mu=mu)
    with pytest.raises(ValueError, match="mu must be finite"):
        mesh.integrate_density_matrix(
            mu=mu,
            lattice_vectors=[(0,)],
            target_error=1.0,
            max_refinements=0,
        )


def test_density_matrix_preview_zero_reuses_the_current_mesh():
    mesh = SpectralMesh(constant_insulator(1))
    mesh.estimate_charge_on_current_mesh(mu=0.0)
    cached_vertices = mesh.cached_vertices

    result = mesh.integrate_density_matrix(
        mu=0.0,
        lattice_vectors=[(0,)],
        target_error=0.0,
        max_refinements=0,
        preview_depth=0,
    )

    assert result.matrices[0] == pytest.approx(np.diag([1.0, 0.0]))
    assert result.stopping_error == pytest.approx(0.0)
    assert result.stats.evaluations == 0
    assert result.stats.refinements == 0
    assert mesh.cached_vertices == cached_vertices


def test_density_matrix_rejects_negative_preview_depth():
    mesh = SpectralMesh(constant_insulator(1))

    with pytest.raises(ValueError, match="preview_depth"):
        mesh.integrate_density_matrix(
            mu=0.0,
            lattice_vectors=[(0,)],
            target_error=1.0,
            preview_depth=-1,
        )


@pytest.mark.parametrize("ndim", (1, 2, 3, 4))
def test_density_matrix_has_one_full_matrix_per_lattice_key(ndim):
    mesh = SpectralMesh(constant_insulator(ndim))
    key = (0,) * ndim

    result = mesh.integrate_density_matrix(
        mu=0.0,
        lattice_vectors=[key],
        target_error=1e-12,
        max_refinements=8,
    )

    assert result.matrices.shape == (1, 2, 2)
    assert np.allclose(result.matrices[0], np.diag([1.0, 0.0]), atol=1e-12)
    assert result.stopping_error <= 1e-12


def test_density_matrix_matches_a_dense_reference():
    hoppings = dimerized_chain()
    keys = [(0,), (1,), (-1,)]
    reference = dense_reference(hoppings, mu=0.0, keys=keys, nk=1201)
    mesh = SpectralMesh(hoppings)

    result = mesh.integrate_density_matrix(
        mu=0.0,
        lattice_vectors=keys,
        target_error=5e-3,
        max_refinements=512,
        preview_depth=2,
    )

    assert result.matrices.shape == (len(keys), 2, 2)
    for index, key in enumerate(keys):
        assert (
            np.max(np.abs(result.matrices[index] - reference.density_matrices[key]))
            <= 5e-3
        )
    assert result.stopping_error <= 5e-3


def test_adaptive_density_components_match_independently_refined_full_density():
    hoppings = qiwuzhang()
    keys = [(0, 0), (1, 0), (0, 1)]
    components = [(0, 0, 0), (1, 1, 0), (2, 0, 1)]
    common = {
        "mu": 0.0,
        "lattice_vectors": keys,
        "max_refinements": 250,
        "preview_depth": 2,
    }

    selected = SpectralMesh(hoppings, root_level=1).integrate_density_components(
        components=components,
        target_error=5e-2,
        **common,
    )
    full = SpectralMesh(hoppings, root_level=1).integrate_density_matrix(
        target_error=2e-2,
        **common,
    )
    expected = np.asarray(
        [full.matrices[key, row, column] for key, row, column in components]
    )

    assert selected.stats.refinements > 0
    assert selected.stats.evaluations > 0
    assert selected.stopping_error <= 5e-2
    assert selected.values == pytest.approx(expected, abs=2e-3)


def test_density_components_match_full_matrices_in_request_order():
    hoppings = dimerized_chain()
    keys = [(0,), (1,), (-1,)]
    components = [(2, 1, 0), (0, 0, 1), (1, 1, 0), (2, 1, 0)]
    options = {
        "mu": 0.0,
        "lattice_vectors": keys,
        "target_error": 1e6,
        "max_refinements": 0,
        "preview_depth": 2,
    }

    full = SpectralMesh(hoppings).integrate_density_matrix(**options)
    selected = SpectralMesh(hoppings).integrate_density_components(
        components=components,
        **options,
    )
    expected = np.asarray(
        [full.matrices[key, row, column] for key, row, column in components]
    )

    assert isinstance(selected, DensityComponentsResult)
    assert selected.values == pytest.approx(expected)
    assert selected.values[0] == pytest.approx(selected.values[3])
    assert selected.stopping_error <= full.stopping_error
    assert selected.stats.simplex_visits == full.stats.simplex_visits


def test_shared_mesh_charge_and_selected_density_workflow():
    mesh = SpectralMesh(qiwuzhang(), root_level=1)
    charge = mesh.integrate_charge(
        mu=0.2,
        target_error=3e-2,
        max_refinements=100,
    )
    assert charge.stats.refinements > 0

    active_simplices = mesh.active_simplices
    cached_vertices = mesh.cached_vertices
    current_charge = mesh.estimate_charge_on_current_mesh(mu=-0.1)
    assert np.isfinite(current_charge.value)
    assert mesh.active_simplices == active_simplices
    assert mesh.cached_vertices == cached_vertices

    options = {
        "mu": -0.1,
        "lattice_vectors": [(0, 0), (1, 0)],
        "components": [(0, 0, 0), (1, 0, 1)],
        "target_error": 1e9,
        "max_refinements": 0,
    }
    current_mesh_density = mesh.integrate_density_components(
        preview_depth=0,
        **options,
    )
    assert current_mesh_density.stats.evaluations == 0
    assert current_mesh_density.stats.refinements == 0
    assert mesh.active_simplices == active_simplices
    assert mesh.cached_vertices == cached_vertices

    preview_density = mesh.integrate_density_components(
        preview_depth=1,
        **options,
    )
    assert preview_density.stats.evaluations > 0
    assert preview_density.stats.refinements == 0
    assert mesh.active_simplices == active_simplices
    assert np.all(np.isfinite(current_mesh_density.values))
    assert np.all(np.isfinite(preview_density.values))


def test_density_components_preview_zero_reuses_the_current_mesh():
    mesh = SpectralMesh(constant_insulator(1))
    mesh.estimate_charge_on_current_mesh(mu=0.0)
    cached_vertices = mesh.cached_vertices

    result = mesh.integrate_density_components(
        mu=0.0,
        lattice_vectors=[(0,)],
        components=[(0, 0, 0)],
        target_error=0.0,
        max_refinements=0,
        preview_depth=0,
    )

    assert result.values == pytest.approx([1.0])
    assert result.stopping_error == pytest.approx(0.0)
    assert result.stats.evaluations == 0
    assert result.stats.refinements == 0
    assert mesh.cached_vertices == cached_vertices


@pytest.mark.parametrize(
    ("components", "exception", "message"),
    (
        ([], ValueError, "shape"),
        ([(0, 0)], ValueError, "shape"),
        ([(0.5, 0, 0)], TypeError, "integers"),
        ([(-1, 0, 0)], ValueError, "non-negative"),
        ([(1, 0, 0)], RuntimeError, "out of range"),
        ([(0, 2, 0)], RuntimeError, "out of range"),
        ([(0, 0, 2)], RuntimeError, "out of range"),
    ),
)
def test_density_components_validate_indices(components, exception, message):
    mesh = SpectralMesh(constant_insulator(1))

    with pytest.raises(exception, match=message):
        mesh.integrate_density_components(
            mu=0.0,
            lattice_vectors=[(0,)],
            components=components,
            target_error=1.0,
            max_refinements=0,
            preview_depth=0,
        )


def test_density_lattice_vectors_must_match_the_model_dimension():
    mesh = SpectralMesh(constant_insulator(2))

    with pytest.raises(ValueError, match="lattice_vectors must have shape"):
        mesh.integrate_density_matrix(
            mu=0.0,
            lattice_vectors=[(0,)],
            target_error=1.0,
            max_refinements=0,
        )


@pytest.mark.parametrize("path", ("adaptive", "current_mesh"))
def test_public_charge_integration_paths(public_hamiltonian, path):
    mesh = SpectralMesh(public_hamiltonian)

    if path == "adaptive":
        result = mesh.integrate_charge(
            mu=0.0,
            target_error=1e6,
            max_refinements=0,
        )
        assert np.isfinite(result.value)
        assert np.isfinite(result.dcharge_dmu)
        assert result.stats.simplex_visits > 0
    else:
        result = mesh.estimate_charge_on_current_mesh(mu=0.0)
        assert np.isfinite(result.value)
        assert np.isfinite(result.dcharge_dmu)


@pytest.mark.parametrize("selected", (False, True), ids=("full", "components"))
@pytest.mark.parametrize("preview_depth", (0, 1))
def test_public_density_integration_paths(
    public_hamiltonian,
    selected,
    preview_depth,
):
    mesh = SpectralMesh(public_hamiltonian)
    mesh.estimate_charge_on_current_mesh(mu=0.0)
    options = {
        "mu": 0.0,
        "lattice_vectors": [(0,), (1,)],
        "target_error": 1e6,
        "max_refinements": 0,
        "preview_depth": preview_depth,
    }

    if selected:
        result = mesh.integrate_density_components(
            components=[(0, 0, 0), (1, 0, 1)],
            **options,
        )
        assert result.values.shape == (2,)
        assert np.all(np.isfinite(result.values))
    else:
        result = mesh.integrate_density_matrix(**options)
        assert result.matrices.shape == (2, 2, 2)
        assert np.all(np.isfinite(result.matrices))

    assert result.stats.refinements == 0
    if preview_depth == 0:
        assert result.stats.evaluations == 0
    else:
        assert result.stats.evaluations > 0
