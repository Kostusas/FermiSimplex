from __future__ import annotations

import inspect
import operator
from collections.abc import Callable, Mapping

import numpy as np

from ._native import SpectralMesh as _NativeSpectralMesh
from ._native import TightBindingModel


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


def _matrix_at(function: Callable[..., Matrix], point: Point) -> Matrix:
    return np.ascontiguousarray(
        np.asarray(function(*point.tolist()), dtype=np.complex128)
    )


def _coordinates_array(coordinates, ndim: int) -> Point:
    result = np.ascontiguousarray(np.asarray(coordinates, dtype=np.float64))
    if result.shape != (ndim,):
        raise ValueError(f"expected {ndim} coordinates")
    if not np.all(np.isfinite(result)):
        raise ValueError("coordinates must be finite")
    return result


class _HamiltonianModel:
    def _create_spectral_mesh(
        self,
        tolerance: float,
        root_level: int,
    ) -> _NativeSpectralMesh:
        raise NotImplementedError


class Hamiltonian(_HamiltonianModel):
    """Dense Hamiltonian defined by a function of explicit coordinates."""

    def __init__(self, function: Callable[..., Matrix]) -> None:
        if not callable(function):
            raise TypeError("function must be callable")
        self._function = function
        self._ndim = _coordinate_count(function)

        matrix = _matrix_at(function, np.zeros(self.ndim))
        if (
            matrix.ndim != 2
            or matrix.shape[0] != matrix.shape[1]
            or not matrix.shape[0]
        ):
            raise ValueError("Hamiltonian must return a non-empty square matrix")
        self._ndof = matrix.shape[0]

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def ndof(self) -> int:
        return self._ndof

    def __call__(self, *coordinates) -> Matrix:
        return self._evaluate(_coordinates_array(coordinates, self.ndim))

    def _evaluate(self, point: Point) -> Matrix:
        matrix = _matrix_at(self._function, point)
        expected_shape = (self.ndof, self.ndof)
        if matrix.shape != expected_shape:
            raise ValueError(
                f"Hamiltonian must return a matrix with shape {expected_shape}"
            )
        return matrix

    def _create_spectral_mesh(
        self,
        tolerance: float,
        root_level: int,
    ) -> _NativeSpectralMesh:
        return _NativeSpectralMesh(
            self._evaluate,
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

    def __call__(self, *coordinates) -> Matrix:
        point = _coordinates_array(coordinates, self.ndim)
        return np.asarray(self._native.evaluate(point))

    def _create_spectral_mesh(
        self,
        tolerance: float,
        root_level: int,
    ) -> _NativeSpectralMesh:
        return _NativeSpectralMesh(self._native, tolerance, root_level)


__all__ = ["Hamiltonian", "TightBinding"]
