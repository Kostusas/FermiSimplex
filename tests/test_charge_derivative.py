"""Analytic references for narrow affine bands and band endpoints."""

import numpy as np
import pytest

from fermisimplex import SpectralMesh


@pytest.mark.parametrize("width", [1e-4, 1e-6, 1e-8])
def test_fully_occupied_narrow_band_has_zero_derivative(width):
    def model(x, y):
        return np.array([[width * (x + y)]])

    mesh = SpectralMesh(model, root_level=0)
    for result in (
        mesh.estimate_charge_on_current_mesh(mu=1),
        mesh.integrate_charge(mu=1, target_error=1e-12, max_refinements=0),
    ):
        assert result.value == 1
        assert result.dcharge_dmu == 0


@pytest.mark.parametrize("width", [1e-4, 1e-8])
@pytest.mark.parametrize("level", [-1.0, 0.0, 0.25, 1.0, 1.75, 2.0, 3.0])
def test_affine_band_charge_and_derivative_against_exact_area(width, level):
    def model(x, y):
        return np.array([[width * (x + y)]])

    result = SpectralMesh(model, root_level=0).estimate_charge_on_current_mesh(
        mu=width * level
    )
    t = np.clip(level, 0, 2)
    charge = 0.5 * t**2 if t <= 1 else 1 - 0.5 * (2 - t) ** 2
    slope = t if t <= 1 else 2 - t
    # Scale out bandwidth: both exact references are O(1). The tolerance is
    # much smaller than integration targets, with room for cut-rule roundoff.
    assert abs(result.value - charge) < 1e-12
    assert abs(width * result.dcharge_dmu - slope) < 1e-12


def test_interval_band_endpoint_derivative_convention():
    mesh = SpectralMesh(lambda x: np.array([[x]]), root_level=0)
    # Retain the existing left derivative at the upper band edge.
    assert mesh.estimate_charge_on_current_mesh(mu=0).dcharge_dmu == 0
    assert mesh.estimate_charge_on_current_mesh(mu=1).dcharge_dmu == 1
    assert (
        mesh.estimate_charge_on_current_mesh(mu=np.nextafter(1, np.inf)).dcharge_dmu
        == 0
    )


@pytest.mark.parametrize("dimension", [3, 4])
@pytest.mark.parametrize("epsilon", [1e-4, 1e-8, 1e-10])
@pytest.mark.parametrize("root_level", [0, 1])
def test_clustered_vertex_energies_have_exact_affine_derivative(
    dimension, epsilon, root_level
):
    def three(x, y, z):
        return np.array([[x + epsilon * (y + z)]])

    def four(x, y, z, w):
        return np.array([[x + epsilon * (y + z + w)]])

    mesh = SpectralMesh({3: three, 4: four}[dimension], root_level=root_level)
    result = mesh.estimate_charge_on_current_mesh(mu=0.5)
    # Integrate x first: all transverse slices have an interior crossing.
    exact_charge = 0.5 - epsilon * (dimension - 1) / 2
    assert abs(result.value - exact_charge) < 1e-12
    assert abs(result.dcharge_dmu - 1) < 1e-12
