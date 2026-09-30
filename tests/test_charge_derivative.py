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
@pytest.mark.parametrize("level", [-1.0, 0.0, 0.25, 0.9, 1.1, 1.75, 2.0, 3.0])
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


@pytest.mark.parametrize("tolerance", [0, 1e-14, 0.05])
def test_interval_band_endpoint_derivative_convention(tolerance):
    mesh = SpectralMesh(lambda x: np.array([[x]]), root_level=0, tolerance=tolerance)
    for mu in (0, 1, np.nextafter(1, np.inf)):
        expected = 1 if tolerance == 0 and mu == 1 else 0
        assert mesh.estimate_charge_on_current_mesh(mu=mu).dcharge_dmu == expected


@pytest.mark.parametrize("mu,charge", [(0.025, 0), (0.975, 1)])
def test_snapped_interval_endpoint_has_zero_slope(mu, charge):
    mesh = SpectralMesh(lambda x: np.array([[x]]), root_level=0, tolerance=0.05)
    for shift in (-1e-5, 0, 1e-5):
        result = mesh.estimate_charge_on_current_mesh(mu=mu + shift)
        assert result.value == charge
        assert result.dcharge_dmu == 0


@pytest.mark.parametrize(
    "width,tolerance,level,delta",
    [(1, 0.05, 1.025, 1e-5), (1e-8, 1e-14, 1.00000025, 1e-16)],
)
def test_snapped_triangle_charge_has_its_actual_derivative(
    width, tolerance, level, delta
):
    mesh = SpectralMesh(
        lambda x, y: np.array([[width * (x + 2 * y)]]),
        root_level=0,
        tolerance=tolerance,
    )
    result = mesh.estimate_charge_on_current_mesh(mu=width * level)
    # Exact reported charge while the vertex at energy `width` is snapped.
    charge = level / 6 + level**2 / 12
    slope = (1 + level) / 6
    difference = (
        mesh.estimate_charge_on_current_mesh(mu=width * level + delta).value
        - mesh.estimate_charge_on_current_mesh(mu=width * level - delta).value
    ) / (2 * delta)
    assert abs(result.value - charge) < 1e-12
    assert abs(width * result.dcharge_dmu - slope) < 1e-12
    assert abs(width * difference - slope) < 1e-8


@pytest.mark.parametrize("tolerance,slope", [(0, 1), (1e-14, 0.5), (0.05, 0.5)])
def test_exact_interior_knot_derivative(tolerance, slope):
    mesh = SpectralMesh(
        lambda x, y: np.array([[x + y]]), root_level=0, tolerance=tolerance
    )
    result = mesh.estimate_charge_on_current_mesh(mu=1)
    assert result.value == 0.5
    assert result.dcharge_dmu == slope


@pytest.mark.parametrize("dimension", [3, 4])
@pytest.mark.parametrize("epsilon", [1e-4, 1e-8, 1e-10])
@pytest.mark.parametrize("root_level", [0, 1])
@pytest.mark.parametrize("level,tolerance", [(0.43, 1e-14), (0.5, 0)])
def test_clustered_vertex_energies_have_exact_affine_derivative(
    dimension, epsilon, root_level, level, tolerance
):
    def three(x, y, z):
        return np.array([[x + epsilon * (y + z)]])

    def four(x, y, z, w):
        return np.array([[x + epsilon * (y + z + w)]])

    mesh = SpectralMesh(
        {3: three, 4: four}[dimension], root_level=root_level, tolerance=tolerance
    )
    result = mesh.estimate_charge_on_current_mesh(mu=level)
    # Integrate x first: all transverse slices have an interior crossing.
    exact_charge = level - epsilon * (dimension - 1) / 2
    assert abs(result.value - exact_charge) < 1e-12
    assert abs(result.dcharge_dmu - 1) < 1e-12
