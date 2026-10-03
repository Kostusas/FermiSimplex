from __future__ import annotations

import math
import operator
from collections.abc import Callable, Mapping

import numpy as np

from ._native import (
    ChargeErrorStats,
    ChargeResult,
    CurrentMeshChargeResult,
    DensityComponentsResult,
    DensityMatrixResult,
    FermiSurfaceResult,
    FermiSurfaceStats,
    IntegrationStats,
    OccupationEnclosure,
)
from .hamiltonian import _coordinates_array, _create_spectral_mesh
from .snapshot import EvaluatedSnapshot


def _finite_float(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _readonly(array) -> np.ndarray:
    result = np.asarray(array)
    result.flags.writeable = False
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
        raise ValueError(f"lattice_vectors must have shape (n, {ndim}) with n > 0")
    return result


def _density_component_array(components) -> np.ndarray:
    result = np.asarray(components)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] != 3:
        raise ValueError("components must have shape (n, 3) with n > 0")
    if not np.issubdtype(result.dtype, np.integer):
        raise TypeError("components must contain only integers")
    return np.ascontiguousarray(result, dtype=np.int64)


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
            "max_refinement_batch_size must be at least min_refinement_batch_size"
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


def _density_parameters(target_error, max_refinements, max_degree, max_h_refinements):
    degree = _positive_integer(max_degree, "max_degree")
    if degree < 3 or degree > 21 or degree % 2 == 0:
        raise ValueError("max_degree must be an odd integer in [3, 21]")
    return (
        _nonnegative_float(target_error, "target_error"),
        -1
        if max_refinements is None
        else _nonnegative_integer(max_refinements, "max_refinements"),
        degree,
        -1
        if max_h_refinements is None
        else _nonnegative_integer(max_h_refinements, "max_h_refinements"),
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
        hamiltonian: (Callable[..., np.ndarray] | Mapping[tuple[int, ...], np.ndarray]),
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

    @property
    def points(self) -> np.ndarray:
        """Active mesh vertices in reduced coordinates, as a read-only array."""
        return _readonly(self._native.points())

    @property
    def simplices(self) -> np.ndarray:
        """Active simplex indices into points, as a read-only array."""
        return _readonly(self._native.simplices())

    @property
    def eigenvalues(self) -> np.ndarray:
        """Cached active-mesh eigenvalues, as a read-only array."""
        return _readonly(self._native.eigenvalues())

    @property
    def eigenvectors(self) -> np.ndarray:
        """Cached active-mesh eigenvector matrices, as a read-only array.

        Each matrix uses the numpy.linalg.eigh convention: columns are
        normalized eigenvectors and the last axis is the band index.
        """
        return _readonly(self._native.eigenvectors())

    def evaluated_snapshot(
        self, *, include_eigenvectors: bool = True
    ) -> EvaluatedSnapshot:
        """Copy all retained spectra and the finest evaluated partition.

        Includes density-preview points and complete existing subdivisions.
        Does not evaluate, diagonalize, refine, or mutate the mesh. Raises
        RuntimeError if no complete evaluated covering exists. Eigenvectors
        can be omitted to reduce copying. Temporary charge-error data is
        excluded. See EvaluatedSnapshot for indexing and exact coordinate keys.
        """
        return EvaluatedSnapshot._from_native(
            self._native.evaluated_snapshot(include_eigenvectors)
        )

    def integrate_charge(
        self,
        *,
        mu: float,
        target_error: float,
        max_refinements: int | None = None,
        error_depth: int = 2,
        min_refinement_batch_size: int = 1,
        max_refinement_batch_size: int = 100,
    ) -> ChargeResult:
        """Adaptively integrate the zero-temperature charge.

        Structurally constant tight-binding matrices do not refine. If their
        eigenvalue uncertainty or level-rounding error exceeds the target,
        integration raises a nonconvergence error on the current mesh.

        Parameters
        ----------
        mu
            Chemical potential.
        target_error
            Target for the sampled adaptive stopping estimate.
        max_refinements
            Maximum number of simplex refinements, or ``None`` for no limit.
        error_depth
            Depth of temporary polynomial subdivision (each level gives
            ``2**ndim`` children). It refines a shared quadratic Schur
            occupation enclosure without further Hamiltonian samples. The
            interpolation remainder is estimated from samples, so the error
            indicator can still miss structure between those probes.
        min_refinement_batch_size, max_refinement_batch_size
            Bounds on the number of simplices refined in one adaptive step.

        Returns
        -------
        ChargeResult
            Charge, sampled stopping-error estimate, derivative with respect
            to ``mu``, sampled ``density_cut_error`` for the reported
            occupation cut, and integration and estimator statistics. The cut
            indicator does not enter the charge stopping test.
            The derivative holds tolerance-snapped vertices on the level;
            it is undefined at thresholds where that classification changes.
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

    def occupation_enclosures(
        self,
        *,
        mu: float,
        depth: int = 2,
        interpolation_error_bound: float | None = None,
    ) -> list[OccupationEnclosure]:
        """Inspect quadratic enclosures in the order of ``simplices``.

        Does not refine. With no bound the interpolation remainder is sampled;
        a fixed occupation is then a conditional sign claim. A supplied bound
        must uniformly bound the quadratic Hamiltonian interpolation error on
        every current simplex. It is the caller's responsibility to establish it.
        Returned charge endpoints include polynomial integration uncertainty.
        Constant tight-binding matrices also retain eigensolver roundoff near
        ``mu``; constancy alone does not prove an exact flat-band occupation.
        """
        bound = (
            None
            if interpolation_error_bound is None
            else _nonnegative_float(
                interpolation_error_bound, "interpolation_error_bound"
            )
        )
        return self._native.occupation_enclosures(
            _finite_float(mu, "mu"), _nonnegative_integer(depth, "depth"), bound
        )

    def estimate_charge_on_current_mesh(
        self,
        *,
        mu: float,
    ) -> CurrentMeshChargeResult:
        """Integrate charge directly on the current mesh.

        This evaluates missing eigensystems at existing vertices and applies
        the linear-simplex charge rule. It performs no certification, error
        estimation, temporary subdivision, or persistent refinement.
        The returned slope differentiates the reported charge with near-level
        vertices held on the cut by the mesh tolerance.
        """
        return self._native.estimate_charge_on_current_mesh(
            _finite_float(mu, "mu"),
        )

    def occupied_weights(self, mu: float) -> np.ndarray:
        """Return occupied cut-simplex weights on the active mesh.

        The result has shape (active_vertices, ndof) and its rows match
        points, eigenvalues, and eigenvectors. It accumulates analytical
        barycentric moments for every active simplex and band. All active
        eigensystems must already be cached; this method never evaluates the
        Hamiltonian or refines the mesh.

        Consequently, weights.sum() is the current-mesh particle number and
        np.sum(weights * mesh.eigenvalues) is the occupied band energy in the
        same normalization as the other FermiSimplex integrals.
        """
        return np.asarray(
            self._native.occupied_weights(
                _finite_float(mu, "mu"),
            )
        )

    def estimate_density_on_current_mesh(
        self,
        *,
        mu: float,
        lattice_vectors,
        components,
    ) -> DensityComponentsResult:
        """Evaluate affine density moments on the current mesh.

        Evaluates missing vertex spectra without refining or sampling interior
        points. No quadrature error is estimated; ``stopping_error`` is zero.
        ``components`` lists (lattice-vector index, row, column) entries.
        """
        return self._native.estimate_density_on_current_mesh(
            _finite_float(mu, "mu"),
            _lattice_vector_array(lattice_vectors, self.ndim),
            _density_component_array(components),
        )

    def integrate_density_matrix(
        self,
        *,
        mu: float,
        lattice_vectors,
        target_error: float,
        max_refinements: int | None = None,
        max_degree: int = 7,
        max_h_refinements: int | None = None,
    ) -> DensityMatrixResult:
        """Integrate full real-space density matrices on the resolved charge mesh.

        First call :meth:`integrate_charge` at the requested ``mu``. Density
        uses the same adaptive cubature as :meth:`integrate_density_components`:
        raise polynomial degree, then bisect unresolved density cells. Charge
        geometry and occupation remain fixed. The error estimate covers smooth
        quadrature, not the remaining occupation-cut error.

        Returns matrices with shape ``(len(lattice_vectors), ndof, ndof)``.
        Limits and stopping behavior match :meth:`integrate_density_components`.
        """
        return self._native.integrate_density_matrix(
            _finite_float(mu, "mu"),
            _lattice_vector_array(lattice_vectors, self.ndim),
            *_density_parameters(
                target_error, max_refinements, max_degree, max_h_refinements
            ),
        )

    def integrate_density_components(
        self,
        *,
        mu: float,
        lattice_vectors,
        components,
        target_error: float,
        max_refinements: int | None = None,
        max_degree: int = 7,
        max_h_refinements: int | None = None,
    ) -> DensityComponentsResult:
        """Integrate selected entries on the resolved charge mesh.

        First call :meth:`integrate_charge` at the requested ``mu``. Components
        are (lattice-vector index, row, column); request order and duplicates
        are preserved. Only requested entries are retained at interior samples.

        Compare degree three with the vertex average, then raise the degree
        through 5, 7, ..., ``max_degree`` (odd, in [3, 21]). Nested samples are
        reused within this call. Cells that exhaust the degree cap bisect on a
        private density tree and restart at degree three. Their occupations
        restrict the original charge-simplex cuts, preserving total charge.
        Vertex moments correct the leading occupation/projector correlation.

        ``max_refinements`` limits degree promotions and ``max_h_refinements``
        limits bisections. Both are unlimited by default; zero forbids the
        corresponding work. Budget exhaustion is reported by
        ``stats.target_reached == False``. The stopping estimate combines
        incoherent and coherent changes between rules with a roundoff floor.
        Higher-order cut errors remain outside this empirical estimate.

        Charge geometry and its retained spectra remain unchanged. When
        bisections are disabled, OpenMP can parallelize promotion batches for
        at least 32 orbitals. Worker exceptions propagate to the caller.
        """
        return self._native.integrate_density_components(
            _finite_float(mu, "mu"),
            _lattice_vector_array(lattice_vectors, self.ndim),
            _density_component_array(components),
            *_density_parameters(
                target_error, max_refinements, max_degree, max_h_refinements
            ),
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
            Hamiltonian. A positive bound gives a uniform allowance for the
            quadratic interpolant; ``None`` and ``0.0`` use sampled remainders.
            Surface classification and charge use the same enclosure.

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
    "CurrentMeshChargeResult",
    "DensityComponentsResult",
    "DensityMatrixResult",
    "FermiSurfaceResult",
    "FermiSurfaceStats",
    "IntegrationStats",
    "OccupationEnclosure",
    "SpectralMesh",
]
