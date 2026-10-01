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


def test_root_center_frame_is_shared_by_sign_proof_and_charge():
    def model(x):
        return np.array([[x - 0.2, 0.2 * x], [0.2 * x, 0.8 - x]])

    result = SpectralMesh(model, root_level=0).integrate_charge(
        mu=0, target_error=2, error_depth=0, max_refinements=0
    )
    # Both states stay active and their center is not diagonal in the anchor
    # frame. The determinant's two zeros delimit a positive-definite interval.
    exact = 1 - np.sqrt(1 - 4 * 1.04 * 0.16) / 1.04
    assert abs(result.value - exact) <= result.stopping_error + 1e-12
    assert result.error_stats.initial_active_dimension_sum == 2
    assert result.error_stats.reduced_eigensystems == 1
    assert result.error_stats.micro_simplices == 1


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


@pytest.mark.parametrize("tolerance,offset", [(1e-14, 5e-13), (1e-5, 5e-4)])
@pytest.mark.parametrize("sign", [-1, 1])
def test_constant_charge_uses_the_same_level_tolerance(tolerance, offset, sign):
    mesh = SpectralMesh({(0,): np.array([[100.0]])}, root_level=0, tolerance=tolerance)
    mu = 100 + sign * offset
    exact = float(sign > 0)
    assert mesh.estimate_charge_on_current_mesh(mu=mu).value == exact
    enclosure = mesh.occupation_enclosures(mu=mu)[0]
    assert enclosure.charge_lower == enclosure.charge_upper == exact
    assert enclosure.fixed_occupation
    result = mesh.integrate_charge(mu=mu, target_error=0, max_refinements=0)
    assert result.value == exact
    assert result.stopping_error == 0


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


@pytest.mark.parametrize("tolerance", [1e-6, 1e-3])
@pytest.mark.parametrize("depth", [0, 2, 6])
@pytest.mark.parametrize("dimension", [1, 2])
def test_cut_indicator_covers_level_tolerance_despite_charge_cancellation(
    tolerance, depth, dimension
):
    delta = tolerance / 2

    def model(x):
        return np.diag([x - delta, x - (1 - delta)])

    hamiltonian = model if dimension == 1 else lambda x, y: model(x)
    mesh = SpectralMesh(hamiltonian, root_level=0, tolerance=tolerance)
    charge = mesh.integrate_charge(
        mu=0, target_error=2 * tolerance, max_refinements=0, error_depth=depth
    )
    density = mesh.integrate_density_components_p(
        mu=0,
        lattice_vectors=[(0,) * dimension],
        components=[(0, 0, 0), (0, 1, 1)],
        target_error=1e-10,
        max_refinements=0,
    )
    exact_density = np.array([delta, 1 - delta])
    error = np.sum(np.abs(density.values - exact_density))
    assert charge.stats.target_reached and density.stats.target_reached
    assert charge.value == pytest.approx(1, abs=1e-12)
    if dimension == 1:
        assert charge.stopping_error < 1e-10
    assert error == pytest.approx(2 * delta, abs=1e-12)
    assert error <= charge.density_cut_error + 1e-12
    assert charge.density_cut_error < 2 * delta + 1e-10


@pytest.mark.parametrize("tight_binding", [False, True])
def test_cut_indicator_includes_tolerance_on_removed_safe_bands(tight_binding):
    # Both signs are strictly safe, but the reported cut is half occupied.
    h = np.diag([-5e-4, 5e-4])
    model = {(0,): h} if tight_binding else lambda x: h
    mesh = SpectralMesh(model, root_level=0, tolerance=1e-3)
    enclosure = mesh.occupation_enclosures(mu=0)[0]
    assert enclosure.active_dimension == 0
    assert enclosure.fixed_occupation
    assert enclosure.density_cut_error == 1
    assert mesh.estimate_charge_on_current_mesh(mu=0).value == 1


@pytest.mark.parametrize("offset", [-5e-4, 0, 5e-4])
@pytest.mark.parametrize("origin", [0, 100])
def test_constant_enclosure_keeps_physical_charge_despite_level_rounding(
    offset, origin
):
    mesh = SpectralMesh(
        {(0,): np.array([[origin + offset]])}, root_level=0, tolerance=1e-3
    )
    exact = 1.0 if offset < 0 else 0.0 if offset > 0 else 0.5
    enclosure = mesh.occupation_enclosures(mu=origin)[0]
    charge = mesh.integrate_charge(mu=origin, target_error=1, max_refinements=0)
    assert charge.value == 0.5
    assert enclosure.charge_lower == enclosure.charge_upper == exact
    assert enclosure.density_cut_error == charge.density_cut_error == abs(0.5 - exact)
    assert charge.stopping_error == abs(0.5 - exact)
    assert enclosure.fixed_occupation == (offset != 0)
    assert enclosure.active_dimension == (1 if offset == 0 else 0)
    assert not enclosure.remainder_is_sampled
    if offset != 0:
        with pytest.raises(RuntimeError, match="did not converge"):
            mesh.integrate_charge(mu=origin, target_error=0, max_refinements=0)
    else:
        assert mesh.integrate_charge(
            mu=origin, target_error=0, max_refinements=0
        ).stats.target_reached


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


@pytest.mark.parametrize("dimension", [1, 2])
@pytest.mark.parametrize(
    "scale,tolerance,expected_cut", [(1.0, 1e-14, 0.0), (1e-4, 1e-3, 1.0)]
)
def test_fixed_occupation_tightens_active_band_cut_indicator(
    dimension, scale, tolerance, expected_cut
):
    sx = np.array([[0, 1], [1, 0]], complex)
    sy = np.array([[0, -1j], [1j, 0]], complex)
    sz = np.diag([1, -1]).astype(complex)
    x = 0.5 * sz - 0.5j * sx
    model = {(0,) * dimension: 0.7 * sx + 1.5 * sz}
    model[(1,) + (0,) * (dimension - 1)] = x
    model[(-1,) + (0,) * (dimension - 1)] = x.conj().T
    if dimension == 2:
        y = 0.15 * sx + 0.125 * sz - 0.25j * sy
        model[(0, 1)] = y
        model[(0, -1)] = y.conj().T
    model = {key: scale * value for key, value in model.items()}
    mesh = SpectralMesh(model, root_level=2, tolerance=tolerance)
    result = mesh.integrate_charge(mu=0, target_error=1e-8, max_refinements=1000)
    enclosures = mesh.occupation_enclosures(mu=0)
    # dz >= scale * .25 in 2D (and scale * .5 in 1D), so exactly one
    # ordered band is occupied. Root block proofs can beat affine row bounds.
    assert all(e.fixed_occupation for e in enclosures)
    if dimension == 2:
        assert any(e.active_dimension > 0 for e in enclosures)
    assert result.value == pytest.approx(1, abs=1e-12)
    assert result.stopping_error == 0
    # With the larger level tolerance both reported bands are half occupied.
    # Equal total charge must not erase their unit occupation disagreement.
    assert result.density_cut_error == pytest.approx(expected_cut, abs=1e-12)
