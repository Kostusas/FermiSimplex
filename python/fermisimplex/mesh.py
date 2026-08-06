from __future__ import annotations

import math
import operator
from collections.abc import Callable, Mapping

import numpy as np

from ._native import (
    ChargeErrorStats,
    ChargeResult,
    DensityMatrixResult,
    FermiSurfaceResult,
    FermiSurfaceStats,
    IntegrationStats,
)
from .hamiltonian import _coordinates_array, _create_spectral_mesh


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


def _positive_integer(value: int, name: str) -> int:
    result = _nonnegative_integer(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


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


def _adaptive_parameters(
    target_error: float,
    max_refinements: int | None,
    preview_depth: int,
    min_refinement_batch_size: int,
    max_refinement_batch_size: int,
) -> tuple[float, int, int, int, int]:
    minimum_batch = _positive_integer(
        min_refinement_batch_size,
        "min_refinement_batch_size",
    )
    maximum_batch = _positive_integer(
        max_refinement_batch_size,
        "max_refinement_batch_size",
    )
    if maximum_batch < minimum_batch:
        raise ValueError(
            "max_refinement_batch_size must be at least "
            "min_refinement_batch_size"
        )
    refinement_limit = (
        -1
        if max_refinements is None
        else _nonnegative_integer(max_refinements, "max_refinements")
    )
    return (
        _nonnegative_float(target_error, "target_error"),
        refinement_limit,
        _nonnegative_integer(preview_depth, "preview_depth"),
        minimum_batch,
        maximum_batch,
    )


class SpectralMesh:
    """Adaptive simplex mesh and shared Hamiltonian spectrum cache.

    Parameters
    ----------
    hamiltonian
        Either a callable ``hamiltonian(kx, ky, ...) -> matrix`` or a
        tight-binding mapping ``{lattice_vector: hopping_matrix}``. Callable
        dimensions are inferred from the required positional arguments and
        from the matrix returned at the origin. Tight-binding mappings must
        contain opposite lattice vectors satisfying
        ``H[-R] == H[R].conj().T``.
    tolerance
        Non-negative numerical tolerance used when comparing energies with
        the chemical potential.
    root_level
        Initial uniform dyadic refinement level in reduced coordinates.

    Notes
    -----
    Callable models must return a finite Hermitian matrix of the inferred shape
    at every point. Values are trusted after the one-time shape inference.

    The mesh is stateful. Calculations refine its geometry and cache vertex
    eigensystems, so later calculations on the same instance reuse earlier
    work.
    """

    def __init__(
        self,
        hamiltonian: (
            Callable[..., np.ndarray]
            | Mapping[tuple[int, ...], np.ndarray]
        ),
        *,
        tolerance: float = 1e-14,
        root_level: int = 1,
    ) -> None:
        self._tolerance = _nonnegative_float(tolerance, "tolerance")
        self._root_level = _root_level(root_level)
        self._native = _create_spectral_mesh(
            hamiltonian,
            self._tolerance,
            self._root_level,
        )

    def evaluate(self, *coordinates) -> np.ndarray:
        """Evaluate the Hamiltonian at separate reduced coordinates."""
        point = _coordinates_array(coordinates, self.ndim)
        return np.asarray(self._native.evaluate(point))

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
        *,
        mu: float,
        target_error: float,
        max_refinements: int | None = None,
        error_depth: int = 1,
        min_refinement_batch_size: int = 1,
        max_refinement_batch_size: int = 100,
    ) -> ChargeResult:
        """Adaptively integrate the zero-temperature charge.

        Parameters
        ----------
        mu
            Chemical potential.
        target_error
            Target for the sampled adaptive stopping estimate.
        max_refinements
            Maximum number of simplex refinements, or ``None`` for no limit.
        error_depth
            Maximum number of complete temporary microsimplex subdivisions on
            each unresolved branch. One subdivision produces ``2**ndim``
            children; certified branches stop early. Temporary samples do not
            refine the persistent mesh. The estimate is not a rigorous bound
            and can miss structure between sampled microvertices. The frozen
            safe-block approximation can also lose accuracy when its variation
            is comparable with the anchor gap or its inertia changes.
        min_refinement_batch_size, max_refinement_batch_size
            Bounds on the number of simplices refined in one adaptive step.

        Returns
        -------
        ChargeResult
            Charge, sampled stopping-error estimate, derivative with respect
            to ``mu``, and integration and estimator statistics.
        """
        adaptive = _adaptive_parameters(
            target_error,
            max_refinements,
            0,
            min_refinement_batch_size,
            max_refinement_batch_size,
        )
        (
            target,
            refinement_limit,
            _,
            minimum_batch,
            maximum_batch,
        ) = adaptive
        return self._native.integrate_charge(
            _finite_float(mu, "mu"),
            target,
            refinement_limit,
            _nonnegative_integer(error_depth, "error_depth"),
            minimum_batch,
            maximum_batch,
        )

    def estimate_charge_on_current_mesh(
        self,
        *,
        mu: float,
        target_error: float,
        error_depth: int = 1,
    ) -> ChargeResult:
        """Estimate charge without committing further mesh refinement.

        The recursive estimator evaluates the Hamiltonian at temporary
        microvertices but does not add those points to the persistent mesh.
        ``error_depth=1`` permits one complete subdivision on each unresolved
        branch; certified branches stop early. The returned stopping error
        is sampled rather than rigorous. The corrected frozen-safe-block
        approximation can lose accuracy when variation is comparable with the
        anchor gap or safe-block inertia changes.
        """
        return self._native.estimate_charge_on_current_mesh(
            _finite_float(mu, "mu"),
            _nonnegative_float(target_error, "target_error"),
            _nonnegative_integer(error_depth, "error_depth"),
        )

    def integrate_density_matrix(
        self,
        *,
        mu: float,
        lattice_vectors,
        target_error: float,
        max_refinements: int | None = None,
        preview_depth: int = 1,
        min_refinement_batch_size: int = 1,
        max_refinement_batch_size: int = 100,
    ) -> DensityMatrixResult:
        """Adaptively integrate real-space density-matrix components.

        Parameters
        ----------
        mu
            Chemical potential.
        lattice_vectors
            Integer lattice vectors with shape ``(count, ndim)``.
        target_error
            Target for the adaptive quadrature estimate.
        max_refinements
            Maximum number of simplex refinements, or ``None`` for no limit.
        preview_depth
            Refinement depth used to estimate each simplex correction.
        min_refinement_batch_size, max_refinement_batch_size
            Bounds on the number of simplices refined in one adaptive step.

        Returns
        -------
        DensityMatrixResult
            Matrices with shape ``(count, ndof, ndof)``, their adaptive error
            estimate, and integration statistics. Density matrices are not
            currently certified.
        """
        adaptive = _adaptive_parameters(
            target_error,
            max_refinements,
            _positive_integer(preview_depth, "preview_depth"),
            min_refinement_batch_size,
            max_refinement_batch_size,
        )
        return self._native.integrate_density_matrix(
            _finite_float(mu, "mu"),
            _lattice_vector_array(lattice_vectors, self.ndim),
            *adaptive,
        )

    def fermi_surface(
        self,
        *,
        mu: float,
        min_feature_size: float,
        max_evaluations: int | None = None,
        curvature_bound: float | None = None,
    ) -> FermiSurfaceResult:
        """Find band-labelled Fermi-surface cells in reduced coordinates.

        Parameters
        ----------
        mu
            Chemical potential defining the surface.
        min_feature_size
            Refine unresolved simplices until their diameter is no larger
            than this value.
        max_evaluations
            Maximum number of new vertex diagonalizations, or ``None`` for no
            limit.
        curvature_bound
            Uniform bound on directional second derivatives of the
            Hamiltonian. ``None`` and ``0.0`` both assert zero curvature.

        Returns
        -------
        FermiSurfaceResult
            Surface points, cells, band labels, completion state, coverage
            certificate, and evaluation statistics.
        """
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
    "ChargeErrorStats",
    "ChargeResult",
    "DensityMatrixResult",
    "FermiSurfaceResult",
    "FermiSurfaceStats",
    "IntegrationStats",
    "SpectralMesh",
]
