from __future__ import annotations

import numpy as np

from ._native import (
    CertificateStatus,
    MuInterval,
    OccupationBounds,
    SimplexCertificate,
    certify_simplex as _certify_simplex,
)


def certify_simplex(
    eigenvalues,
    eigenvectors,
    *,
    linearization_error_bound: float,
    mu: float = 0.0,
    tolerance: float = 1e-14,
) -> SimplexCertificate:
    """Certify a simplex from trusted vertex eigensystems.

    ``eigenvalues`` must be finite and sorted in ascending order at every
    vertex. The columns of each ``eigenvectors`` matrix must form a finite
    orthonormal basis. These numerical preconditions are not checked.
    """
    values = np.ascontiguousarray(np.asarray(eigenvalues, dtype=np.float64))
    vectors = np.ascontiguousarray(np.asarray(eigenvectors, dtype=np.complex128))
    return _certify_simplex(
        values,
        vectors,
        float(linearization_error_bound),
        float(mu),
        float(tolerance),
    )


__all__ = [
    "CertificateStatus",
    "MuInterval",
    "OccupationBounds",
    "SimplexCertificate",
    "certify_simplex",
]
