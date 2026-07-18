from __future__ import annotations

import numpy as np
import pytest

from lineartetrahedron import CertificateStatus, certify_simplex

from .helpers import tb_k_matrix, winding_constant_gap_band


def _spectra(hoppings, points):
    values = []
    vectors = []
    for point in points:
        eigenvalues, eigenvectors = np.linalg.eigh(
            tb_k_matrix(hoppings, np.asarray([point]))
        )
        values.append(eigenvalues)
        vectors.append(eigenvectors)
    return np.asarray(values), np.asarray(vectors)


def test_certifies_a_gapped_simplex_with_an_explicit_linearization_error_bound():
    eigenvalues = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    eigenvectors = np.repeat(np.eye(2, dtype=complex)[None], 2, axis=0)

    certificate = certify_simplex(
        eigenvalues,
        eigenvectors,
        linearization_error_bound=0.25,
    )

    assert certificate.status is CertificateStatus.CertifiedGapped
    assert certificate.occupation_bounds.lower == 1
    assert certificate.occupation_bounds.upper == 1


def test_linearization_error_bound_is_required():
    eigenvalues = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    eigenvectors = np.repeat(np.eye(2, dtype=complex)[None], 2, axis=0)

    with pytest.raises(TypeError, match="linearization_error_bound"):
        certify_simplex(eigenvalues, eigenvectors)


def test_reports_a_visible_crossing():
    eigenvalues = np.array([[0.0, 1.0], [-1.0, 1.0]])
    eigenvectors = np.repeat(np.eye(2, dtype=complex)[None], 2, axis=0)

    certificate = certify_simplex(
        eigenvalues,
        eigenvectors,
        linearization_error_bound=0.0,
    )

    assert certificate.status is CertificateStatus.VisibleGapless
    assert certificate.occupation_bounds.lower == 0
    assert certificate.occupation_bounds.upper == 1


def test_reports_an_inconclusive_rotating_eigenspace():
    eigenvalues, eigenvectors = _spectra(
        winding_constant_gap_band(2),
        [0.0, 0.25],
    )

    certificate = certify_simplex(
        eigenvalues,
        eigenvectors,
        linearization_error_bound=0.0,
    )

    assert certificate.status is CertificateStatus.Inconclusive
    assert certificate.occupation_bounds.lower == 0
    assert certificate.occupation_bounds.upper == 2


def test_mu_interval_preserves_bounds_but_not_necessarily_status():
    eigenvalues = np.repeat(np.array([[-2.0, 0.0, 1.0]]), 2, axis=0)
    eigenvectors = np.repeat(np.eye(3, dtype=complex)[None], 2, axis=0)

    certificate = certify_simplex(
        eigenvalues,
        eigenvectors,
        linearization_error_bound=0.0,
    )
    shifted = certify_simplex(
        eigenvalues,
        eigenvectors,
        linearization_error_bound=0.0,
        mu=0.5,
    )

    assert certificate.status is CertificateStatus.VisibleGapless
    assert certificate.occupation_bounds_valid_at(0.5)
    assert shifted.status is CertificateStatus.CertifiedGapped
