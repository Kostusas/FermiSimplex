from __future__ import annotations

import math
import operator

import numpy as np

from ._native import (
    AdaptiveOptions,
    ChargeResult,
    DensityMatrixResult,
    FermiSurfaceResult,
    FermiSurfaceStats,
    IntegrationStats,
)
from .hamiltonian import Hamiltonian, TightBinding, _HamiltonianModel


def _finite_float(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative_float(value: float, name: str) -> float:
    result = _finite_float(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive_float(value: float, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _curvature_bound(value: float | None) -> float:
    return 0.0 if value is None else _nonnegative_float(value, "curvature_bound")


def _nonnegative_integer(value: int, name: str) -> int:
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return int(result)


def _root_level(value: int) -> int:
    result = _nonnegative_integer(value, "root_level")
    if result >= 31:
        raise ValueError("root_level must be in [0, 31)")
    return result


def _lattice_vector_array(lattice_vectors, ndim: int) -> np.ndarray:
    try:
        vectors = tuple(
            tuple(operator.index(component) for component in vector)
            for vector in lattice_vectors
        )
    except TypeError as exc:
        raise TypeError("lattice_vectors must contain only integers") from exc
    result = np.ascontiguousarray(np.asarray(vectors, dtype=np.int64))
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] != ndim:
        raise ValueError(
            f"lattice_vectors must have shape (n, {ndim}) with n > 0"
        )
    return result


class SpectralMesh:
    """Adaptive simplex mesh and its shared Hamiltonian spectrum cache."""

    def __init__(
        self,
        model: Hamiltonian | TightBinding,
        *,
        tolerance: float = 1e-14,
        root_level: int = 1,
    ) -> None:
        if not isinstance(model, _HamiltonianModel):
            raise TypeError("model must be a Hamiltonian or TightBinding")
        self._model = model
        self._tolerance = _nonnegative_float(tolerance, "tolerance")
        self._root_level = _root_level(root_level)
        self._native = model._create_spectral_mesh(
            self._tolerance,
            self._root_level,
        )

    @property
    def model(self) -> Hamiltonian | TightBinding:
        return self._model

    @property
    def ndim(self) -> int:
        return int(self._native.ndim)

    @property
    def ndof(self) -> int:
        return int(self._native.ndof)

    @property
    def tolerance(self) -> float:
        return self._tolerance

    @property
    def root_level(self) -> int:
        return self._root_level

    @property
    def cached_vertices(self) -> int:
        return int(self._native.cached_vertices)

    @property
    def active_simplices(self) -> int:
        return int(self._native.active_simplices)

    @property
    def active_vertices(self) -> int:
        return int(self._native.active_vertices)

    def integrate_charge(
        self,
        mu: float,
        options: AdaptiveOptions,
        *,
        curvature_bound: float | None = None,
    ) -> ChargeResult:
        return self._native.integrate_charge(
            _finite_float(mu, "mu"),
            options,
            _curvature_bound(curvature_bound),
        )

    def estimate_charge_on_current_mesh(
        self,
        mu: float,
        target_error: float,
        *,
        preview_depth: int = 1,
        curvature_bound: float | None = None,
    ) -> ChargeResult:
        depth = _nonnegative_integer(preview_depth, "preview_depth")
        if depth == 0:
            raise ValueError("preview_depth must be positive")
        return self._native.estimate_charge_on_current_mesh(
            _finite_float(mu, "mu"),
            _nonnegative_float(target_error, "target_error"),
            depth,
            _curvature_bound(curvature_bound),
        )

    def integrate_density_matrix(
        self,
        mu: float,
        lattice_vectors,
        options: AdaptiveOptions,
    ) -> DensityMatrixResult:
        return self._native.integrate_density_matrix(
            _finite_float(mu, "mu"),
            _lattice_vector_array(lattice_vectors, self.ndim),
            options,
        )

    def fermi_surface(
        self,
        mu: float,
        min_feature_size: float,
        max_evaluations: int | None = None,
        *,
        curvature_bound: float | None = None,
    ) -> FermiSurfaceResult:
        evaluation_budget = (
            None
            if max_evaluations is None
            else _nonnegative_integer(max_evaluations, "max_evaluations")
        )
        return self._native.fermi_surface(
            _finite_float(mu, "mu"),
            _positive_float(min_feature_size, "min_feature_size"),
            evaluation_budget,
            _curvature_bound(curvature_bound),
        )


__all__ = [
    "AdaptiveOptions",
    "ChargeResult",
    "DensityMatrixResult",
    "FermiSurfaceResult",
    "FermiSurfaceStats",
    "IntegrationStats",
    "SpectralMesh",
]
