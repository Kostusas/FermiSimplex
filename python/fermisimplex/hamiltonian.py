from __future__ import annotations

import inspect
import operator
from collections.abc import Callable, Mapping

import numpy as np

from ._native import SpectralMesh as _NativeSpectralMesh


Matrix = np.ndarray
Point = np.ndarray


def _coordinate_count(function: Callable[..., Matrix]) -> int:
    try:
        parameters = tuple(inspect.signature(function).parameters.values())
    except (TypeError, ValueError) as exc:
        raise TypeError("function must have an inspectable signature") from exc

    positional = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    if not parameters or any(
        parameter.kind not in positional
        or parameter.default is not inspect.Parameter.empty
        for parameter in parameters
    ):
        raise TypeError(
            "function must accept one or more required positional coordinates"
        )
    return len(parameters)


def _coordinates_array(coordinates, ndim: int) -> Point:
    result = np.ascontiguousarray(np.asarray(coordinates, dtype=np.float64))
    if result.shape != (ndim,):
        raise ValueError(f"expected {ndim} coordinates")
    if not np.all(np.isfinite(result)):
        raise ValueError("coordinates must be finite")
    return result


def _callable_dimensions(function: Callable[..., Matrix]) -> tuple[int, int]:
    ndim = _coordinate_count(function)
    matrix = np.asarray(function(*np.zeros(ndim)), dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not matrix.shape[0]:
        raise ValueError("Hamiltonian must return a non-empty square matrix")
    return ndim, int(matrix.shape[0])


def _tight_binding_arrays(
    hoppings: Mapping[tuple[int, ...], Matrix],
) -> tuple[np.ndarray, np.ndarray]:
    if not hoppings:
        raise ValueError("hoppings must not be empty")

    try:
        lattice_vectors = [
            tuple(operator.index(component) for component in vector)
            for vector in hoppings
        ]
    except TypeError as exc:
        raise TypeError("hopping lattice vectors must contain only integers") from exc
    ndim = len(lattice_vectors[0])
    if ndim < 1 or any(len(vector) != ndim for vector in lattice_vectors):
        raise ValueError("lattice vectors must have one common positive dimension")

    matrices = [np.asarray(matrix) for matrix in hoppings.values()]
    shape = matrices[0].shape
    if len(shape) != 2 or shape[0] != shape[1] or shape[0] < 1:
        raise ValueError("hopping matrices must be non-empty and square")
    if any(matrix.shape != shape for matrix in matrices):
        raise ValueError("hopping matrices must all have the same shape")

    return (
        np.ascontiguousarray(np.asarray(lattice_vectors, dtype=np.int64)),
        np.ascontiguousarray(np.asarray(matrices, dtype=np.complex128)),
    )


def _create_spectral_mesh(
    hamiltonian,
    tolerance: float,
    root_level: int,
) -> _NativeSpectralMesh:
    if isinstance(hamiltonian, Mapping):
        lattice_vectors, matrices = _tight_binding_arrays(hamiltonian)
        return _NativeSpectralMesh(
            lattice_vectors,
            matrices,
            tolerance,
            root_level,
        )
    if callable(hamiltonian):
        ndim, ndof = _callable_dimensions(hamiltonian)
        return _NativeSpectralMesh(
            hamiltonian,
            ndim,
            ndof,
            tolerance,
            root_level,
        )
    raise TypeError(
        "hamiltonian must be a callable or a tight-binding mapping"
    )
