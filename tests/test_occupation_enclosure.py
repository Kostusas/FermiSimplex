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
        mu=0.23, target_error=1e-5, max_refinements=300
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
    result = mesh.integrate_charge(mu=0, target_error=0, max_refinements=0)
    assert result.value == 1
    assert result.stopping_error == 0
    assert result.density_cut_error == 0


def test_quadratic_charge_reports_visible_crossings():
    result = SpectralMesh(scalar(lambda x: x - 0.37), root_level=0).integrate_charge(
        mu=0, target_error=1, max_refinements=0
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
        mu=mu, target_error=1e-3, max_refinements=500
    )
    assert abs(result.value - reference) <= result.stopping_error + 1e-12


def test_cut_disagreement_survives_total_charge_cancellation():
    mesh = SpectralMesh(
        scalar(lambda x: (x - 0.2) * (x - 0.5) * (x - 0.8)), root_level=0
    )
    result = mesh.integrate_charge(mu=0, target_error=2, max_refinements=0)
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


@pytest.mark.parametrize("degree", [3, 4])
@pytest.mark.parametrize("scale", [1.0, 1e-4, 1e-8])
def test_cubic_and_quartic_pockets_with_exact_occupied_length(degree, scale):
    def polynomial(x):
        factor = (
            1 + 0.6 * (x - 0.5) if degree == 3 else 1 + 0.6 * (x - 0.5) + (x - 0.5) ** 2
        )
        return scale * ((x - 0.37) ** 2 - 0.04**2) * factor

    # The extra factor is strictly positive on [0,1]; the occupied length is .08.
    mesh = SpectralMesh(scalar(polynomial), root_level=0)
    enclosures = mesh.occupation_enclosures(mu=0, depth=6)
    lower, upper = interval(enclosures)
    assert lower - 1e-12 <= 0.08 <= upper + 1e-12
    assert not enclosures[0].fixed_occupation


def test_quartic_interior_bubble_is_not_a_gap():
    def hamiltonian(x, y):
        return np.array([[0.001 + y * (1 - x) * (x - y) * (1 - 2 * x + y)]], complex)

    # On the first root triangle this equals .001 at every edge sample and
    # at the centroid. A negative interior witness proves an actual crossing.
    assert hamiltonian(0.8, 0.3)[0, 0].real == pytest.approx(-0.008)
    mesh = SpectralMesh(hamiltonian, root_level=0)
    enclosure = mesh.occupation_enclosures(mu=0, depth=6)[0]
    assert not enclosure.fixed_occupation
    # Exact maximum absolute quartic bubble on a triangle.
    assert enclosure.interpolation_error >= 9 / (512 * np.sqrt(3))
    surface = SpectralMesh(hamiltonian, root_level=0).fermi_surface(
        mu=0, min_feature_size=0.03, max_evaluations=5000
    )
    assert surface.completed
    assert len(surface.points) > 0
    # The formerly missed triangle contains extracted surface points.
    assert np.any(surface.points[:, 0] > surface.points[:, 1])


def test_exact_flat_tight_binding_charge_converges_without_claiming_a_gap():
    mesh = SpectralMesh({(0,): np.diag([0.0, 1.0])}, root_level=0)
    result = mesh.integrate_charge(mu=0, target_error=0, max_refinements=0)
    assert result.value == 0.5
    assert result.stopping_error == result.density_cut_error == 0
    enclosure = mesh.occupation_enclosures(mu=0)[0]
    assert enclosure.charge_lower == enclosure.charge_upper == 0.5
    assert not enclosure.fixed_occupation
    assert not enclosure.remainder_is_sampled


@pytest.mark.parametrize("dimension", [2, 3])
def test_quartic_lattice_requires_dimension_dependent_remainder(dimension):
    from itertools import product
    from math import factorial

    nodes = [
        a
        for a in product(range(5), repeat=dimension + 1)
        if sum(a) == 4 and any(n % 2 for n in a)
    ]
    witness = np.array([1 - dimension / 8] + [1 / 8] * dimension)

    def cardinals(weights):
        return np.array(
            [
                np.prod(
                    [
                        np.prod([4 * w - j for j in range(degree)]) / factorial(degree)
                        for w, degree in zip(weights, alpha)
                    ]
                )
                for alpha in nodes
            ]
        )

    signs = np.sign(cardinals(witness))
    offset = 2.2 if dimension == 2 else 3.2

    def residual(weights):
        return offset - signs @ cardinals(weights)

    assert residual(witness) < 0
    # At every interpolation node H=offset; each probe residual is +/-1.
    # A factor of two would falsely certify the constant interpolant positive.
    if dimension == 2:

        def model(x, y):
            return np.array([[residual([1 - x, x - y, y])]], complex)
    else:

        def model(x, y, z):
            return np.array([[residual([1 - x, x - y, y - z, z])]], complex)

    e = SpectralMesh(model, root_level=0).occupation_enclosures(mu=0)[0]
    assert not e.fixed_occupation
    assert e.interpolation_error >= 2**dimension - 1e-12
