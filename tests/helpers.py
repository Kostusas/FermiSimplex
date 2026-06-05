from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def dimerized_chain(delta: float = 0.2) -> dict[tuple[int, ...], np.ndarray]:
    return {
        (0,): np.array([[delta, 1.0], [1.0, -delta]], dtype=complex),
        (1,): np.array([[0.0, 0.4], [0.0, 0.0]], dtype=complex),
        (-1,): np.array([[0.0, 0.0], [0.4, 0.0]], dtype=complex),
    }


def qiwuzhang(mass: float = 0.5) -> dict[tuple[int, ...], np.ndarray]:
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sy = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    return {
        (0, 0): mass * sz,
        (1, 0): 0.5 * sz - 0.5j * sx,
        (-1, 0): 0.5 * sz + 0.5j * sx,
        (0, 1): 0.5 * sz - 0.5j * sy,
        (0, -1): 0.5 * sz + 0.5j * sy,
    }


def tb_k_matrix(tb: dict[tuple[int, ...], np.ndarray], point: np.ndarray) -> np.ndarray:
    return sum(
        matrix * np.exp(-1j * np.dot(point, np.asarray(key, dtype=float)))
        for key, matrix in tb.items()
    )


@dataclass(frozen=True)
class DenseReference:
    mu: float
    rho: dict[tuple[int, ...], np.ndarray]
    filling: float


def dense_reference(
    tb: dict[tuple[int, ...], np.ndarray],
    *,
    keys: list[tuple[int, ...]],
    mu: float | None = None,
    filling: float | None = None,
    nk: int = 1201,
) -> DenseReference:
    if (mu is None) == (filling is None):
        raise ValueError("Exactly one of mu or filling must be provided")
    ndim = len(next(iter(tb)))
    ndof = next(iter(tb.values())).shape[0]
    grids = [np.linspace(-np.pi, np.pi, nk, endpoint=False) for _ in range(ndim)]
    points = np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1).reshape(-1, ndim)

    eigvals = []
    eigvecs = []
    for point in points:
        values, vectors = np.linalg.eigh(tb_k_matrix(tb, point))
        eigvals.append(values)
        eigvecs.append(vectors)
    eigvals = np.asarray(eigvals)
    eigvecs = np.asarray(eigvecs)
    if filling is not None:
        flat = np.sort(eigvals.reshape(-1))
        index = int(np.clip(np.ceil(float(filling) * points.shape[0]) - 1, 0, flat.size - 1))
        mu = float(flat[index])
    assert mu is not None

    rho = {key: np.zeros((ndof, ndof), dtype=complex) for key in keys}
    resolved_filling = 0.0
    for point, values, vectors in zip(points, eigvals, eigvecs, strict=True):
        occupations = values <= mu
        resolved_filling += float(np.sum(occupations))
        density = vectors * occupations[np.newaxis, :] @ vectors.conj().T
        for key in keys:
            rho[key] += density * np.exp(1j * np.dot(point, np.asarray(key, dtype=float)))
    for key in keys:
        rho[key] /= points.shape[0]
    resolved_filling /= points.shape[0]
    return DenseReference(mu=float(mu), rho=rho, filling=resolved_filling)


def max_density_error(
    actual: dict[tuple[int, ...], np.ndarray],
    expected: dict[tuple[int, ...], np.ndarray],
) -> float:
    return max(float(np.max(np.abs(actual[key] - expected[key]))) for key in expected)
