from __future__ import annotations

import numpy as np
import pytest

from fermisimplex import SpectralMesh

from .helpers import axis_cosine_band, constant_insulator


@pytest.mark.parametrize("ndim", (1, 2, 3))
def test_extracts_a_fermi_surface_in_reduced_coordinates(ndim):
    mu = 0.3
    feature_size = {1: 0.05, 2: 0.1, 3: 0.24}[ndim]
    mesh = SpectralMesh(axis_cosine_band(ndim))

    surface = mesh.fermi_surface(
        mu=mu,
        min_feature_size=feature_size,
        max_evaluations=20_000,
    )

    assert surface.completed
    assert surface.points.shape[1] == ndim
    assert surface.cells.shape[1] == ndim
    assert surface.cell_bands.shape == (surface.cells.shape[0],)
    assert surface.points.shape[0] > 0
    assert surface.cells.shape[0] > 0
    assert np.all(surface.points >= -1e-14)
    assert np.all(surface.points <= 1.0 + 1e-14)
    assert np.max(np.abs(np.cos(2.0 * np.pi * surface.points[:, 0]) - mu)) < 0.2


def test_callable_affine_hamiltonian_needs_no_curvature_margin():
    mesh = SpectralMesh(
        lambda kx, ky: np.array([[kx + ky - 0.8]], dtype=complex)
    )

    surface = mesh.fermi_surface(
        mu=0.0,
        min_feature_size=0.12,
        curvature_bound=0.0,
    )

    assert surface.completed
    assert surface.coverage_certified
    assert surface.points.shape[1] == 2
    assert surface.points.shape[0] > 0
    assert surface.stats.terminal_visible_simplices > 0
    assert surface.stats.terminal_inconclusive_simplices == 0


def test_constant_insulator_is_certified_without_a_surface():
    mesh = SpectralMesh(constant_insulator(2))

    surface = mesh.fermi_surface(
        mu=0.0,
        min_feature_size=0.4,
        curvature_bound=0.0,
    )

    assert surface.completed
    assert surface.coverage_certified
    assert surface.points.shape == (0, 2)
    assert surface.cells.shape == (0, 2)
    assert surface.cell_bands.shape == (0,)


def test_none_and_zero_curvature_bounds_are_equivalent_for_fermi_surface():
    hamiltonian = constant_insulator(2)

    implicit_zero = SpectralMesh(hamiltonian).fermi_surface(
        mu=0.0,
        min_feature_size=0.4,
    )
    explicit_none = SpectralMesh(hamiltonian).fermi_surface(
        mu=0.0,
        min_feature_size=0.4,
        curvature_bound=None,
    )
    explicit_zero = SpectralMesh(hamiltonian).fermi_surface(
        mu=0.0,
        min_feature_size=0.4,
        curvature_bound=0.0,
    )

    assert implicit_zero.completed
    assert implicit_zero.coverage_certified
    assert explicit_none.coverage_certified
    assert explicit_zero.coverage_certified
    assert implicit_zero.stats.evaluations == explicit_none.stats.evaluations
    assert implicit_zero.stats.evaluations == explicit_zero.stats.evaluations
    assert (
        implicit_zero.stats.terminal_visible_simplices
        == explicit_none.stats.terminal_visible_simplices
        == explicit_zero.stats.terminal_visible_simplices
    )
    assert (
        implicit_zero.stats.terminal_inconclusive_simplices
        == explicit_none.stats.terminal_inconclusive_simplices
        == explicit_zero.stats.terminal_inconclusive_simplices
    )


def test_evaluation_budget_reports_an_incomplete_surface():
    mesh = SpectralMesh(axis_cosine_band(2))

    surface = mesh.fermi_surface(
        mu=0.0,
        min_feature_size=0.02,
        max_evaluations=0,
    )

    assert not surface.completed
    assert not surface.coverage_certified
    assert surface.stats.evaluations == 0


def test_flat_band_at_mu_is_not_coverage_certified():
    hamiltonian = lambda _kx, _ky: np.zeros((1, 1), dtype=complex)

    surface = SpectralMesh(hamiltonian).fermi_surface(
        mu=0.0,
        min_feature_size=0.4,
        curvature_bound=0.0,
    )

    assert surface.completed
    assert not surface.coverage_certified
    assert surface.cells.shape == (0, 2)
    assert surface.cell_bands.shape == (0,)


def test_cells_retain_their_band_identity_without_duplicates():
    def two_affine_bands(kx, ky):
        coordinate_sum = kx + ky
        return np.diag([coordinate_sum - 1.5, coordinate_sum - 0.5]).astype(
            complex
        )

    mesh = SpectralMesh(two_affine_bands)

    surface = mesh.fermi_surface(
        mu=0.0,
        min_feature_size=0.2,
        curvature_bound=0.0,
    )

    assert surface.coverage_certified
    assert surface.cell_bands.shape == (surface.cells.shape[0],)
    assert set(surface.cell_bands.tolist()) == {0, 1}

    canonical_cells = np.sort(surface.cells, axis=1)
    labelled_cells = np.column_stack((surface.cell_bands, canonical_cells))
    assert np.unique(labelled_cells, axis=0).shape[0] == surface.cells.shape[0]

    for band, expected_sum in ((0, 1.5), (1, 0.5)):
        band_cells = surface.cells[surface.cell_bands == band]
        band_points = surface.points[band_cells]
        assert np.max(np.abs(np.sum(band_points, axis=2) - expected_sum)) < 1e-12


@pytest.mark.parametrize("mu", (np.nan, np.inf, -np.inf))
def test_fermi_surface_requires_a_finite_chemical_potential(mu):
    mesh = SpectralMesh(constant_insulator(1))

    with pytest.raises(ValueError, match="mu must be finite"):
        mesh.fermi_surface(mu=mu, min_feature_size=0.1)


@pytest.mark.parametrize("feature_size", (0.0, -1.0, np.nan, np.inf))
def test_fermi_surface_requires_a_finite_positive_feature_size(feature_size):
    mesh = SpectralMesh(constant_insulator(1))

    with pytest.raises(ValueError, match="min_feature_size"):
        mesh.fermi_surface(mu=0.0, min_feature_size=feature_size)


@pytest.mark.parametrize("curvature_bound", (-1.0, np.nan, np.inf))
def test_fermi_surface_rejects_invalid_curvature_bounds(curvature_bound):
    mesh = SpectralMesh(constant_insulator(1))

    with pytest.raises(ValueError, match="curvature_bound"):
        mesh.fermi_surface(
            mu=0.0,
            min_feature_size=0.1,
            curvature_bound=curvature_bound,
        )
