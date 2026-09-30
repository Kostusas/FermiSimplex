"""Exact occupation and hidden quartic pockets beyond three dimensions."""

import numpy as np
import pytest

from fermisimplex import SpectralMesh


@pytest.mark.parametrize("dimension", [4, 5])
def test_affine_occupation_in_higher_dimensions(dimension):
    def four(x, y, z, w):
        return np.array([[x - 0.37]])

    def five(x, y, z, w, u):
        return np.array([[x - 0.37]])

    mesh = SpectralMesh({4: four, 5: five}[dimension], root_level=0)
    result = mesh.integrate_charge(mu=0, target_error=1e-12, max_refinements=0)
    # A half-space of the unit cube has exact occupied volume .37.
    assert result.stats.target_reached
    assert abs(result.value - 0.37) < 1e-12
    assert result.stopping_error < 1e-12


def test_four_dimensional_quartic_face_pocket():
    mesh = SpectralMesh({(0, 0, 0, 0): np.array([[1.0]])}, root_level=0)
    vertices = mesh.points[mesh.simplices[0]]
    barycentric = np.linalg.inv(np.vstack([vertices.T, np.ones(5)]))

    def model(x, y, z, w):
        a, b, c, d, e = barycentric @ [x, y, z, w, 1]
        return np.array([[0.001 + a * b * c * (d - e)]])

    # Vertices, edges, three-vertex faces and the whole-simplex centroid all
    # give +.001. A four-vertex face contains a negative value, hence a crossing.
    witness = np.array([0.25, 0.25, 0.25, 0, 0.25]) @ vertices
    assert model(*witness)[0, 0] == pytest.approx(0.001 - 1 / 256)
    enclosure = SpectralMesh(model, root_level=0).occupation_enclosures(mu=0)[0]
    assert not enclosure.fixed_occupation
    assert enclosure.interpolation_error >= 1 / 256
