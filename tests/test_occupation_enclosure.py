"""Uniform polynomial cases and the explicit limits of sampled remainders."""

import numpy as np
import pytest

from fermisimplex import CertificateStatus, SpectralMesh, certify_simplex


def scalar(function):
    return lambda x: np.array([[function(x)]], complex)


def interval(enclosures):
    return (
        sum(e.charge_lower for e in enclosures),
        sum(e.charge_upper for e in enclosures),
    )


def test_quadratic_pocket_reopens_false_vertex_certificate():
    hamiltonian = scalar(lambda x: (x - 0.37) ** 2 - 0.04**2)
    values, vectors = np.linalg.eigh(np.array([hamiltonian(0), hamiltonian(1)]))
    old = certify_simplex(values, vectors, linearization_error_bound=0)
    assert old.status is CertificateStatus.CertifiedGapped
    mesh = SpectralMesh(hamiltonian, root_level=0)
    enclosures = mesh.occupation_enclosures(mu=0, depth=10, interpolation_error_bound=0)
    low, high = interval(enclosures)
    assert low <= 0.08 <= high
    assert high - low < 1e-4
    assert all(not e.fixed_occupation for e in enclosures)


def test_polynomial_subdivision_needs_no_extra_hamiltonian_samples():
    results = []
    for depth in (0, 2, 5):
        mesh = SpectralMesh(scalar(lambda x: (x - 0.37) ** 2 - 0.04**2), root_level=0)
        results.append(
            mesh.integrate_charge(
                mu=0,
                target_error=2,
                max_refinements=0,
                error_depth=depth,
                method="quadratic",
            )
        )
    assert len({r.error_stats.hamiltonian_evaluations for r in results}) == 1
    assert results[-1].stopping_error < results[0].stopping_error


def test_two_dimensional_pocket_charge_interval():
    def hamiltonian(x, y):
        return np.array([[(x - 0.33) ** 2 + (y - 0.47) ** 2 - 0.12**2]], complex)

    mesh = SpectralMesh(hamiltonian, root_level=0)
    low, high = interval(
        mesh.occupation_enclosures(mu=0, depth=6, interpolation_error_bound=0)
    )
    exact = np.pi * 0.12**2
    assert low <= exact <= high
    assert high - low < 3e-3


@pytest.mark.parametrize(
    "function,charge",
    [
        (lambda x: x - 0.37, 0.37),
        (lambda x: (x - 0.37) ** 2, 0.0),
        (lambda x: 0.0, 0.5),
    ],
)
def test_contacts_and_flat_bands_are_not_strict_gaps(function, charge):
    mesh = SpectralMesh(scalar(function), root_level=0)
    enclosures = mesh.occupation_enclosures(mu=0, depth=6, interpolation_error_bound=0)
    low, high = interval(enclosures)
    assert low - 1e-14 <= charge <= high + 1e-14
    assert any(not e.fixed_occupation for e in enclosures)


def test_known_remainder_protects_an_unsampled_pocket():
    def bump(x):
        t = (x - 0.11) / 0.01
        return 0.1 - (np.exp(1 - 1 / (1 - t * t)) if abs(t) < 1 else 0.0)

    mesh = SpectralMesh(scalar(bump), root_level=0)
    sampled = mesh.occupation_enclosures(mu=0)
    # Deliberate limitation: all interpolation and validation probes miss it.
    assert all(e.fixed_occupation and e.remainder_is_sampled for e in sampled)
    bounded = mesh.occupation_enclosures(mu=0, interpolation_error_bound=1.0)
    assert all(not e.fixed_occupation and not e.remainder_is_sampled for e in bounded)


def test_adaptive_charge_meets_analytic_cosine_reference():
    model = {(1,): np.array([[0.5]]), (-1,): np.array([[0.5]])}
    result = SpectralMesh(model, root_level=2).integrate_charge(
        mu=0.23, target_error=1e-5, max_refinements=300, method="quadratic"
    )
    exact = 1 - np.arccos(0.23) / np.pi
    assert result.stats.target_reached
    assert abs(result.value - exact) <= result.stopping_error + 1e-13
    assert abs(result.value - exact) < 1e-5


@pytest.mark.parametrize("bound", [-1.0, np.nan, np.inf])
def test_invalid_bound_rejected(bound):
    with pytest.raises((ValueError, RuntimeError)):
        SpectralMesh(scalar(lambda x: x)).occupation_enclosures(
            mu=0, interpolation_error_bound=bound
        )


def test_constant_complex_gapped_system_has_exact_charge():
    h = np.array([[-2, 0.3j], [-0.3j, 3]], complex)
    mesh = SpectralMesh(lambda x: h)
    result = mesh.integrate_charge(
        mu=0, target_error=0, max_refinements=0, method="quadratic"
    )
    assert result.value == 1
    assert result.stopping_error == 0
    assert result.density_cut_error == 0


def test_unknown_method_rejected():
    with pytest.raises(ValueError, match="method"):
        SpectralMesh(scalar(lambda x: x)).integrate_charge(
            mu=0, target_error=0.1, method="unknown"
        )


def test_quadratic_charge_reports_visible_crossings():
    result = SpectralMesh(scalar(lambda x: x - 0.37), root_level=0).integrate_charge(
        mu=0, target_error=1, max_refinements=0, method="quadratic"
    )
    assert result.visible_gapless_simplices == 1
    assert result.inconclusive_simplices == 0


def test_nearly_coincident_multiband_cuts():
    size = 12
    onsite = np.diag(np.random.default_rng(11).uniform(-0.35, 0.35, size))
    onsite -= 0.35 * (np.eye(size, k=1) + np.eye(size, k=-1))
    model = {(0,): onsite, (1,): -np.eye(size), (-1,): -np.eye(size)}
    mu = -0.1
    reference = np.sum(
        np.arccos(np.clip((np.linalg.eigvalsh(onsite) - mu) / 2, -1, 1)) / np.pi
    )
    result = SpectralMesh(model, root_level=2).integrate_charge(
        mu=mu, target_error=1e-3, max_refinements=500, method="quadratic"
    )
    assert abs(result.value - reference) <= result.stopping_error + 1e-12


def test_cut_disagreement_survives_total_charge_cancellation():
    mesh = SpectralMesh(
        scalar(lambda x: (x - 0.2) * (x - 0.5) * (x - 0.8)), root_level=0
    )
    result = mesh.integrate_charge(
        mu=0, target_error=2, max_refinements=0, method="quadratic"
    )
    # Both reported and exact charges are 1/2, but their occupied sets differ
    # on (.2,.5) and (.5,.8), with total measure .6.
    assert abs(result.value - 0.5) < 1e-14
    assert result.density_cut_error >= 0.6 - 1e-12


def test_three_dimensional_spherical_pocket():
    def hamiltonian(x, y, z):
        return np.array([[(x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2 - 0.18**2]])

    mesh = SpectralMesh(hamiltonian, root_level=1)
    low, high = interval(
        mesh.occupation_enclosures(mu=0, depth=3, interpolation_error_bound=0)
    )
    exact = 4 * np.pi * 0.18**3 / 3
    assert low <= exact <= high
    assert high - low < 1e-2
