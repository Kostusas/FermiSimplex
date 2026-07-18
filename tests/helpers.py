from __future__ import annotations

from dataclasses import dataclass

import numpy as np


TightBindingData = dict[tuple[int, ...], np.ndarray]


def constant_insulator(ndim: int) -> TightBindingData:
    return {(0,) * ndim: np.diag([-1.0, 1.0]).astype(complex)}


def axis_cosine_band(ndim: int) -> TightBindingData:
    positive = [0] * ndim
    positive[0] = 1
    negative = [0] * ndim
    negative[0] = -1
    return {
        tuple(positive): np.array([[0.5]], dtype=complex),
        tuple(negative): np.array([[0.5]], dtype=complex),
    }


def winding_constant_gap_band(winding: int) -> TightBindingData:
    sigma_z = np.diag([1.0, -1.0]).astype(complex)
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    return {
        (winding,): 0.5 * sigma_z + 0.5j * sigma_x,
        (-winding,): 0.5 * sigma_z - 0.5j * sigma_x,
    }


def dimerized_chain(delta: float = 0.2) -> TightBindingData:
    return {
        (0,): np.array([[delta, 1.0], [1.0, -delta]], dtype=complex),
        (1,): np.array([[0.0, 0.4], [0.0, 0.0]], dtype=complex),
        (-1,): np.array([[0.0, 0.0], [0.4, 0.0]], dtype=complex),
    }


def qiwuzhang(mass: float = 0.5) -> TightBindingData:
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


def tb_k_matrix(hoppings: TightBindingData, point: np.ndarray) -> np.ndarray:
    return sum(
        matrix * np.exp(-2j * np.pi * np.dot(point, np.asarray(key)))
        for key, matrix in hoppings.items()
    )


@dataclass(frozen=True)
class DenseReference:
    density_matrices: dict[tuple[int, ...], np.ndarray]
    filling: float


def dense_reference(
    hoppings: TightBindingData,
    *,
    mu: float,
    keys: list[tuple[int, ...]],
    nk: int = 1201,
) -> DenseReference:
    ndim = len(next(iter(hoppings)))
    ndof = next(iter(hoppings.values())).shape[0]
    grids = [np.linspace(0.0, 1.0, nk, endpoint=False) for _ in range(ndim)]
    points = np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1).reshape(-1, ndim)
    density_matrices = {
        key: np.zeros((ndof, ndof), dtype=complex)
        for key in keys
    }
    filling = 0.0

    for point in points:
        eigenvalues, eigenvectors = np.linalg.eigh(tb_k_matrix(hoppings, point))
        occupations = eigenvalues <= mu
        filling += float(np.sum(occupations))
        density = eigenvectors * occupations[np.newaxis, :] @ eigenvectors.conj().T
        for key in keys:
            phase = np.exp(2j * np.pi * np.dot(point, np.asarray(key)))
            density_matrices[key] += phase * density

    for matrix in density_matrices.values():
        matrix /= len(points)
    return DenseReference(
        density_matrices=density_matrices,
        filling=filling / len(points),
    )
