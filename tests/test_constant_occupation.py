"""Constant spectra: distinguish exact structure from eigensolver roundoff."""

import numpy as np
import pytest

from fermisimplex import SpectralMesh


@pytest.mark.parametrize("bands", [3, 4, 12])
@pytest.mark.parametrize("scale", [1e-8, 1.0, 1e8])
@pytest.mark.parametrize("complex_basis", [False, True])
@pytest.mark.parametrize("tolerance", [0.0, 1e-14, 1e-3])
def test_rank_one_constant_keeps_flat_bands_uncertain(
    bands, scale, complex_basis, tolerance
):
    phases = np.resize([1.0, 1j, -1.0, -1j], bands) if complex_basis else np.ones(bands)
    h = scale * np.outer(phases, phases.conj())
    # The entries share one exactly represented scale and phases in {+/-1,+/-i}.
    # Spectrum: bands-1 exact zeros and one positive eigenvalue bands*scale.
    exact_charge = (bands - 1) / 2
    mesh = SpectralMesh({(0,): h}, root_level=0, tolerance=tolerance)
    enclosure = mesh.occupation_enclosures(mu=0)[0]
    charge = mesh.integrate_charge(mu=0, target_error=bands, max_refinements=0)
    assert enclosure.charge_lower == 0
    assert enclosure.charge_upper == bands - 1
    assert enclosure.charge_lower <= exact_charge <= enclosure.charge_upper
    assert not enclosure.fixed_occupation
    assert not enclosure.remainder_is_sampled
    assert enclosure.active_dimension == bands - 1
    assert abs(charge.value - exact_charge) <= charge.stopping_error
    assert charge.error_stats.hamiltonian_evaluations == 0
    # Allow relative summation roundoff in the independently calculated norm.
    expected_allowance = 64 * np.finfo(float).eps * bands**2 * scale
    assert enclosure.model_error == pytest.approx(expected_allowance, rel=1e-14, abs=0)
    surface = mesh.fermi_surface(mu=0, min_feature_size=0.01, max_evaluations=0)
    assert surface.completed
    assert not surface.coverage_certified
    assert len(mesh.simplices) == 1
    assert surface.stats.terminal_inconclusive_simplices == 1


@pytest.mark.parametrize("shift,exact_charge", [(-1e-14, 3), (1e-14, 0)])
def test_roundoff_uncertainty_does_not_assume_half_occupation(shift, exact_charge):
    # Nearby strictly gapped matrices also lie within the numerical allowance.
    # Collapsing every uncertain band to half occupation would exclude these.
    h = np.ones((4, 4)) + shift * np.eye(4)
    enclosure = SpectralMesh({(0,): h}, root_level=0).occupation_enclosures(mu=0)[0]
    assert enclosure.charge_lower <= exact_charge <= enclosure.charge_upper
    assert enclosure.charge_lower == 0 and enclosure.charge_upper == 3
    assert not enclosure.fixed_occupation


@pytest.mark.parametrize("dimension", [1, 2])
@pytest.mark.parametrize("h", [np.zeros((4, 4)), np.diag([0, 0, 0, 4])])
def test_structurally_diagonal_flat_bands_still_have_exact_charge(dimension, h):
    exact_charge = np.count_nonzero(np.diag(h) == 0) / 2
    mesh = SpectralMesh({(0,) * dimension: h}, root_level=0)
    enclosures = mesh.occupation_enclosures(mu=0)
    assert sum(e.charge_lower for e in enclosures) == exact_charge
    assert sum(e.charge_upper for e in enclosures) == exact_charge
    assert all(e.model_error == 0 and not e.fixed_occupation for e in enclosures)
    result = mesh.integrate_charge(mu=0, target_error=0, max_refinements=0)
    assert result.value == exact_charge
    assert result.stopping_error == result.density_cut_error == 0


@pytest.mark.parametrize("origin", [0, 100])
def test_resolved_complex_constant_gap_still_has_exact_occupation(origin):
    h = np.array([[-2, 0.3j], [-0.3j, 3]]) + origin * np.eye(2)
    mesh = SpectralMesh({(0,): h}, root_level=0)
    enclosure = mesh.occupation_enclosures(mu=origin)[0]
    assert enclosure.model_error > 0
    assert enclosure.fixed_occupation
    assert enclosure.charge_lower == enclosure.charge_upper == 1
    result = mesh.integrate_charge(mu=origin, target_error=0, max_refinements=0)
    assert result.value == 1 and result.stopping_error == 0


def test_exact_diagonal_spectrum_retains_small_energies_and_band_vectors():
    diagonal = np.array([1e200, 0.0, -1e-200])
    mesh = SpectralMesh({(0,): np.diag(diagonal)}, root_level=0, tolerance=0)
    result = mesh.integrate_charge(mu=0, target_error=0, max_refinements=0)
    np.testing.assert_array_equal(mesh.eigenvalues[0], np.sort(diagonal))
    assert result.value == 1.5 and result.stopping_error == 0
    density = mesh.integrate_density_components_p(
        mu=0,
        lattice_vectors=[(0,)],
        components=[(0, i, i) for i in range(3)],
        target_error=1e-12,
        max_refinements=0,
    )
    np.testing.assert_allclose(density.values, [0, 0.5, 1], rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "h,tolerance", [(np.ones((4, 4)), 1e-14), (np.array([[5e-4]]), 1e-3)]
)
def test_constant_charge_does_not_refine_an_irreducible_error(h, tolerance):
    mesh = SpectralMesh({(0,): h}, root_level=0, tolerance=tolerance)
    simplices = mesh.simplices.copy()
    # A finite cap makes a regression fail promptly instead of refining forever.
    with pytest.raises(RuntimeError, match="did not converge"):
        mesh.integrate_charge(mu=0, target_error=1e-6, max_refinements=5)
    np.testing.assert_array_equal(mesh.simplices, simplices)
    assert mesh.cached_vertices == 2
