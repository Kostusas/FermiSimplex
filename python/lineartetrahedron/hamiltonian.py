from __future__ import annotations

import operator
from collections.abc import Callable, Mapping

import numpy as np

from ._native import SpectralMesh as _NativeSpectralMesh
from ._native import TightBindingModel


Matrix = np.ndarray
Point = np.ndarray


def _positive_integer(value: int, name: str) -> int:
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return int(result)


def _point_array(point, ndim: int) -> Point:
    result = np.ascontiguousarray(np.asarray(point, dtype=np.float64))
    if result.shape != (ndim,):
        raise ValueError(f"point must have shape ({ndim},)")
    if not np.all(np.isfinite(result)):
        raise ValueError("point coordinates must be finite")
    return result


class _HamiltonianModel:
    def _create_spectral_mesh(
        self,
        tolerance: float,
        root_level: int,
    ) -> _NativeSpectralMesh:
        raise NotImplementedError


class Hamiltonian(_HamiltonianModel):
    """Dense Hamiltonian defined by a Python callable."""

    def __init__(
        self,
        function: Callable[[Point], Matrix],
        *,
        ndim: int,
        ndof: int,
    ) -> None:
        if not callable(function):
            raise TypeError("function must be callable")
        self._function = function
        self._ndim = _positive_integer(ndim, "ndim")
        self._ndof = _positive_integer(ndof, "ndof")

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def ndof(self) -> int:
        return self._ndof

    def __call__(self, point) -> Matrix:
        matrix = np.ascontiguousarray(
            np.asarray(self._function(_point_array(point, self.ndim)), dtype=np.complex128)
        )
        expected_shape = (self.ndof, self.ndof)
        if matrix.shape != expected_shape:
            raise ValueError(f"Hamiltonian must return a matrix with shape {expected_shape}")
        return matrix

    def _create_spectral_mesh(
        self,
        tolerance: float,
        root_level: int,
    ) -> _NativeSpectralMesh:
        return _NativeSpectralMesh(
            self,
            self.ndim,
            self.ndof,
            tolerance,
            root_level,
        )


def _tight_binding_arrays(
    hoppings: Mapping[tuple[int, ...], Matrix],
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(hoppings, Mapping):
        raise TypeError("hoppings must be a mapping")
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


class TightBinding(_HamiltonianModel):
    """Dense translation-invariant Hamiltonian in reduced coordinates."""

    def __init__(self, hoppings: Mapping[tuple[int, ...], Matrix]) -> None:
        lattice_vectors, matrices = _tight_binding_arrays(hoppings)
        self._native = TightBindingModel(lattice_vectors, matrices)

    @property
    def ndim(self) -> int:
        return int(self._native.ndim)

    @property
    def ndof(self) -> int:
        return int(self._native.ndof)

    def __call__(self, point) -> Matrix:
        return np.asarray(self._native.evaluate(_point_array(point, self.ndim)))

    def _create_spectral_mesh(
        self,
        tolerance: float,
        root_level: int,
    ) -> _NativeSpectralMesh:
        return _NativeSpectralMesh(self._native, tolerance, root_level)


__all__ = ["Hamiltonian", "TightBinding"]
