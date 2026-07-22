from __future__ import annotations

import numpy as np
import pytest

from fermisimplex import SpectralMesh

from .helpers import constant_insulator, dense_reference, dimerized_chain


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"target_error": -1.0}, "target_error"),
        ({"target_error": np.inf}, "target_error"),
        ({"target_error": np.nan}, "target_error"),
        ({"target_error": 1.0, "max_refinements": -2}, "max_refinements"),
        ({"target_error": 1.0, "preview_depth": 0}, "preview_depth"),
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
        mesh.estimate_charge_on_current_mesh(mu=mu, target_error=1.0)
    with pytest.raises(ValueError, match="mu must be finite"):
        mesh.integrate_density_matrix(
            mu=mu,
            lattice_vectors=[(0,)],
            target_error=1.0,
            max_refinements=0,
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


def test_density_lattice_vectors_must_match_the_model_dimension():
    mesh = SpectralMesh(constant_insulator(2))

    with pytest.raises(ValueError, match="lattice_vectors must have shape"):
        mesh.integrate_density_matrix(
            mu=0.0,
            lattice_vectors=[(0,)],
            target_error=1.0,
            max_refinements=0,
        )
