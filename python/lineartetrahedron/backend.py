from __future__ import annotations

import inspect
import operator
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from ._native import (
        AdaptiveOptions,
        InconclusiveChargeErrorMode,
        IntegrationRuntime,
        TightBindingHessianBound,
        TightBindingModel,
    )
    from ._native import _fermi_surface_stats as _native_fermi_surface_stats
    from ._native import _reset_fermi_surface_stats
    from ._native import certify_simplex as _native_certify_simplex
    from ._native import fermi_surface as _native_fermi_surface
    from ._native import fermi_surface_callable as _native_fermi_surface_callable

    NATIVE_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when extension is unavailable
    AdaptiveOptions = None
    InconclusiveChargeErrorMode = None
    IntegrationRuntime = None
    TightBindingModel = None
    TightBindingHessianBound = None
    _native_certify_simplex = None
    _native_fermi_surface_stats = None
    _reset_fermi_surface_stats = None
    _native_fermi_surface = None
    _native_fermi_surface_callable = None
    NATIVE_AVAILABLE = False

_tb_type = dict[tuple[int, ...], np.ndarray]
_GEOM_TOL = 1e-14


@dataclass(frozen=True)
class IntegrationStats:
    n_evaluator_evals: int
    n_cached_nodes: int
    n_active_simplices: int
    n_active_vertices: int
    subdivisions: int
    converged: bool


@dataclass(frozen=True)
class DensityMatrixComponentsResult:
    values: np.ndarray
    errors: np.ndarray
    keys: tuple[tuple[int, ...], ...]
    components: tuple[tuple[int, int], ...]
    stats: IntegrationStats


@dataclass(frozen=True)
class OccupationBounds:
    lower: int
    upper: int


@dataclass(frozen=True)
class MuInterval:
    lower: float
    upper: float

    @property
    def is_valid(self) -> bool:
        return self.lower <= self.upper


@dataclass(frozen=True)
class SimplexCertificate:
    status: str
    occupation_bounds: OccupationBounds
    mu_interval: MuInterval
    energy_bound: float

    @property
    def has_mu_interval(self) -> bool:
        return self.mu_interval.is_valid

    @property
    def occupation_width(self) -> int:
        return self.occupation_bounds.upper - self.occupation_bounds.lower

    def reusable_at(self, mu: float) -> bool:
        return self.mu_interval.lower <= float(mu) <= self.mu_interval.upper


@dataclass(frozen=True)
class FermiSurfaceStats:
    evaluated_vertices: int
    cut_simplices: int
    feature_size_simplices: int
    unresolved_simplices: int


@dataclass(frozen=True)
class FermiSurfaceParameters:
    mu: float
    min_feature_size: float
    hessian_bound: float | None
    hessian_bound_is_callable: bool
    anharmonicity_bound: float
    max_diagonalizations: int | None
    ndim: int
    ndof: int


@dataclass(frozen=True)
class FermiSurfaceStates:
    band_indices: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray


@dataclass(frozen=True)
class FermiSurface:
    points: np.ndarray
    cells: np.ndarray
    converged: bool
    stats: FermiSurfaceStats
    parameters: FermiSurfaceParameters
    states: FermiSurfaceStates | None = None


def _is_sparse_like(matrix: Any) -> bool:
    return hasattr(matrix, "toarray") and hasattr(matrix, "tocoo")


def _require_native_extension() -> None:
    if not NATIVE_AVAILABLE or TightBindingModel is None or IntegrationRuntime is None:
        raise RuntimeError(
            "Zero-temperature integration requires the compiled "
            "lineartetrahedron._native extension"
        )


def _validate_dense_tb(tb: _tb_type) -> None:
    if not tb:
        raise ValueError("Tight-binding Hamiltonian cannot be empty")
    ndim = len(next(iter(tb)))
    if ndim < 1:
        raise ValueError("Tight-binding Hamiltonian dimension must be positive")
    ndof: int | None = None
    for key, matrix in tb.items():
        if len(tuple(key)) != ndim:
            raise ValueError("All tight-binding keys must have the same dimension")
        if _is_sparse_like(matrix):
            raise TypeError("LinearTetrahedron supports dense complex128 matrices only")
        array = np.asarray(matrix)
        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise ValueError("Tight-binding values must be square matrices")
        if ndof is None:
            ndof = int(array.shape[0])
        elif array.shape != (ndof, ndof):
            raise ValueError("All tight-binding matrices must have the same shape")


def _tb_to_tight_binding_model(tb: _tb_type):
    _require_native_extension()
    keys, matrices = _tb_arrays(tb)
    return TightBindingModel(keys, matrices)


def _tb_arrays(tb: _tb_type):
    _validate_dense_tb(tb)
    keys = np.asarray([tuple(key) for key in tb.keys()], dtype=np.int64, order="C")
    matrices = np.asarray(
        [np.asarray(value) for value in tb.values()],
        dtype=np.complex128,
        order="C",
    )
    return keys, matrices


def tight_binding_hamiltonian(tb: _tb_type):
    _validate_dense_tb(tb)
    ndim = len(next(iter(tb)))
    terms = tuple(
        (
            tuple(int(part) for part in key),
            np.asarray(matrix, dtype=np.complex128),
        )
        for key, matrix in tb.items()
    )

    class TightBindingHamiltonian:
        __signature__ = inspect.Signature(
            [
                inspect.Parameter(
                    f"k{axis}",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
                for axis in range(ndim)
            ]
        )

        def __call__(self, *coords: float) -> np.ndarray:
            if len(coords) != ndim:
                raise TypeError(f"expected {ndim} coordinates, got {len(coords)}")
            k = np.asarray(coords, dtype=float)
            result = np.zeros_like(terms[0][1], dtype=np.complex128)
            for key, matrix in terms:
                phase = np.exp(-2j * np.pi * float(np.dot(k, np.asarray(key, dtype=float))))
                result += phase * matrix
            return result

    return TightBindingHamiltonian()


def tight_binding_hessian_bound(tb: _tb_type):
    _require_native_extension()
    if TightBindingHessianBound is None:
        raise RuntimeError("Tight-binding Hessian bounds require the compiled native extension")
    keys, matrices = _tb_arrays(tb)
    native = TightBindingHessianBound(keys, matrices)
    ndim = int(native.ndim)

    class TightBindingHessianBoundCallable:
        __signature__ = inspect.Signature(
            [
                inspect.Parameter(
                    f"k{axis}",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
                for axis in range(ndim)
            ]
        )

        def __call__(self, *coords: float) -> float:
            if len(coords) != ndim:
                raise TypeError(f"expected {ndim} coordinates, got {len(coords)}")
            point = np.ascontiguousarray(np.asarray(coords, dtype=np.float64))
            return float(native.evaluate_point(point))

    return TightBindingHessianBoundCallable()


def _callable_ndim(hamiltonian: Any) -> int:
    try:
        signature = inspect.signature(hamiltonian)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "Hamiltonian callables must have an inspectable signature with explicit "
            "required scalar coordinate arguments"
        ) from exc

    parameters = list(signature.parameters.values())
    if not parameters:
        raise TypeError("Hamiltonian callable must accept at least one scalar coordinate")

    for parameter in parameters:
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise TypeError("Hamiltonian callable must not use *args or **kwargs")
        if parameter.kind == inspect.Parameter.KEYWORD_ONLY:
            raise TypeError("Hamiltonian coordinates must be positional scalar arguments")
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise TypeError("Hamiltonian coordinates must be positional scalar arguments")
        if parameter.default is not inspect.Parameter.empty:
            raise TypeError("Hamiltonian coordinate arguments must not have defaults")
    return len(parameters)


def _prepare_callable_hamiltonian(hamiltonian: Any):
    ndim = _callable_ndim(hamiltonian)

    def wrapped(*coords: float) -> np.ndarray:
        return np.ascontiguousarray(
            np.asarray(hamiltonian(*coords), dtype=np.complex128)
        )

    try:
        probe = wrapped(*((0.0,) * ndim))
    except TypeError as exc:
        raise TypeError(
            "Hamiltonian callable must accept explicit scalar coordinate arguments, "
            "for example H(kx, ky)"
        ) from exc

    if probe.ndim != 2 or probe.shape[0] != probe.shape[1]:
        raise ValueError("Hamiltonian callable must return a square dense matrix")
    ndof = int(probe.shape[0])
    if ndof < 1:
        raise ValueError("Hamiltonian callable matrix must be non-empty")
    return wrapped, ndim, ndof


def _normalize_max_diagonalizations(max_diagonalizations: int | None) -> int:
    if max_diagonalizations is None:
        return -1
    try:
        value = operator.index(max_diagonalizations)
    except TypeError as exc:
        raise TypeError("max_diagonalizations must be an integer or None") from exc
    if value < 0:
        raise ValueError("max_diagonalizations must be non-negative or None")
    return int(value)


def _normalize_max_refinements(max_refinements: int | None) -> int:
    if max_refinements is None:
        return -1
    try:
        value = operator.index(max_refinements)
    except TypeError as exc:
        raise TypeError("max_refinements must be an integer or None") from exc
    if value < 0:
        raise ValueError("max_refinements must be non-negative or None")
    return int(value)


def _adaptive_options(
    *,
    target_error: float | None,
    options,
    max_refinements: int | None,
    preview_depth: int,
    min_refinement_batch_size: int,
    max_refinement_batch_size: int,
):
    if options is not None:
        if (
            target_error is not None
            or max_refinements is not None
            or preview_depth != 1
            or min_refinement_batch_size != 1
            or max_refinement_batch_size != 100
        ):
            raise ValueError(
                "target_error/refinement keyword arguments cannot be combined with options"
            )
        return options
    if target_error is None:
        raise TypeError("target_error is required when options is not provided")
    return AdaptiveOptions(
        target_error=float(target_error),
        max_refinements=_normalize_max_refinements(max_refinements),
        preview_depth=operator.index(preview_depth),
        min_refinement_batch_size=operator.index(min_refinement_batch_size),
        max_refinement_batch_size=operator.index(max_refinement_batch_size),
    )


def _normalize_hessian_bound(hessian_bound: Any):
    if callable(hessian_bound):
        return hessian_bound
    hessian = float(hessian_bound)
    if not np.isfinite(hessian) or hessian < 0.0:
        raise ValueError("hessian_bound must be finite and non-negative")
    return hessian


def _normalize_keys(keys, ndim: int) -> tuple[tuple[int, ...], ...]:
    if keys is None:
        raise TypeError("keys are required")
    normalized = tuple(tuple(int(part) for part in key) for key in keys)
    if not normalized:
        raise ValueError("keys must be non-empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError("keys must be unique")
    if any(len(key) != ndim for key in normalized):
        raise ValueError("All keys must match the Hamiltonian dimension")
    return normalized


def _normalize_components(
    components,
    ndof: int,
) -> tuple[tuple[int, int], ...]:
    if components is None:
        raise TypeError("components are required")
    normalized: list[tuple[int, int]] = []
    seen = set()
    for component in components:
        if len(component) != 2:
            raise ValueError("components must be (row, col) pairs")
        row, col = (int(component[0]), int(component[1]))
        if row < 0 or col < 0 or row >= ndof or col >= ndof:
            raise ValueError("density component row/column out of bounds")
        pair = (row, col)
        if pair in seen:
            raise ValueError(f"duplicate density component {pair!r}")
        seen.add(pair)
        normalized.append(pair)
    if not normalized:
        raise ValueError("components must be non-empty")
    return tuple(normalized)


def _density_selection_arrays(
    keys: tuple[tuple[int, ...], ...],
    components: tuple[tuple[int, int], ...],
):
    rows: list[int] = []
    cols: list[int] = []
    key_indices: list[int] = []
    for key_index in range(len(keys)):
        for row, col in components:
            rows.append(row)
            cols.append(col)
            key_indices.append(key_index)
    return (
        np.ascontiguousarray(np.asarray(keys, dtype=np.int64)),
        np.ascontiguousarray(np.asarray(rows, dtype=np.int64)),
        np.ascontiguousarray(np.asarray(cols, dtype=np.int64)),
        np.ascontiguousarray(np.asarray(key_indices, dtype=np.int64)),
    )


def _integration_stats(result, mesh: SpectralMesh) -> IntegrationStats:
    return IntegrationStats(
        n_evaluator_evals=int(result.work),
        n_cached_nodes=int(mesh.n_cached_nodes),
        n_active_simplices=int(result.n_active_simplices),
        n_active_vertices=int(result.n_active_vertices),
        subdivisions=int(result.refinements),
        converged=bool(result.converged),
    )


class SpectralMesh:
    def __init__(self, hamiltonian: Any, *, tol: float = _GEOM_TOL) -> None:
        _require_native_extension()
        self._tol = float(tol)
        if not np.isfinite(self._tol) or self._tol < 0.0:
            raise ValueError("tol must be finite and non-negative")

        self._callable = None
        self._wrapped_callable = None
        self._ndim: int
        self._ndof: int

        if callable(hamiltonian):
            wrapped, ndim, ndof = _prepare_callable_hamiltonian(hamiltonian)
            self._callable = hamiltonian
            self._wrapped_callable = wrapped
            self._ndim = int(ndim)
            self._ndof = int(ndof)
            self._native = IntegrationRuntime(wrapped, self._ndim, self._ndof, self._tol)
        else:
            raise TypeError("hamiltonian must be callable")

    @property
    def ndim(self) -> int:
        return int(self._native.ndim)

    @property
    def ndof(self) -> int:
        return int(self._native.ndof)

    @property
    def n_cached_nodes(self) -> int:
        return int(self._native.n_cached_nodes)

    @property
    def n_active_simplices(self) -> int:
        return int(self._native.n_active_simplices)

    @property
    def n_active_vertices(self) -> int:
        return int(self._native.n_active_vertices)


def _require_mesh(mesh: SpectralMesh) -> SpectralMesh:
    if not isinstance(mesh, SpectralMesh):
        raise TypeError("expected a SpectralMesh")
    return mesh


def charge(
    mesh: SpectralMesh,
    *,
    mu: float = 0.0,
    target_error: float | None = None,
    options=None,
    refine: bool = True,
    certify: bool = True,
    hessian_bound: Any = 0.0,
    anharmonicity_bound: float = 0.0,
    inconclusive_error_mode: str = "projected",
    max_refinements: int | None = None,
    preview_depth: int = 1,
    min_refinement_batch_size: int = 1,
    max_refinement_batch_size: int = 100,
):
    mesh = _require_mesh(mesh)
    hessian = _normalize_hessian_bound(hessian_bound)
    anharmonicity = float(anharmonicity_bound)
    if not np.isfinite(anharmonicity) or anharmonicity < 0.0:
        raise ValueError("anharmonicity_bound must be finite and non-negative")
    if not isinstance(inconclusive_error_mode, str):
        raise TypeError("inconclusive_error_mode must be a string")
    normalized_error_mode = inconclusive_error_mode.lower()
    if normalized_error_mode not in {"projected", "conservative"}:
        raise ValueError(
            "inconclusive_error_mode must be 'projected' or 'conservative'"
        )
    native_error_mode = (
        InconclusiveChargeErrorMode.Projected
        if normalized_error_mode == "projected"
        else InconclusiveChargeErrorMode.Conservative
    )
    adaptive_options = _adaptive_options(
        target_error=target_error,
        options=options,
        max_refinements=max_refinements,
        preview_depth=preview_depth,
        min_refinement_batch_size=min_refinement_batch_size,
        max_refinement_batch_size=max_refinement_batch_size,
    )
    return mesh._native.integrate_charge(
        float(mu),
        adaptive_options,
        bool(refine),
        bool(certify),
        hessian,
        anharmonicity,
        native_error_mode,
    )


def density_matrix_components(
    mesh: SpectralMesh,
    *,
    mu: float = 0.0,
    keys,
    components,
    target_error: float | None = None,
    options=None,
    refine: bool = True,
    max_refinements: int | None = None,
    preview_depth: int = 1,
    min_refinement_batch_size: int = 1,
    max_refinement_batch_size: int = 100,
) -> DensityMatrixComponentsResult:
    mesh = _require_mesh(mesh)
    normalized_keys = _normalize_keys(keys, mesh.ndim)
    normalized_components = _normalize_components(components, mesh.ndof)
    adaptive_options = _adaptive_options(
        target_error=target_error,
        options=options,
        max_refinements=max_refinements,
        preview_depth=preview_depth,
        min_refinement_batch_size=min_refinement_batch_size,
        max_refinement_batch_size=max_refinement_batch_size,
    )
    key_array, rows, cols, key_indices = _density_selection_arrays(
        normalized_keys,
        normalized_components,
    )
    result = mesh._native.integrate_density(
        float(mu),
        adaptive_options,
        key_array,
        rows,
        cols,
        key_indices,
        bool(refine),
    )
    shape = (len(normalized_keys), len(normalized_components))
    return DensityMatrixComponentsResult(
        values=np.asarray(result.estimate_array()).reshape(shape),
        errors=np.asarray(result.error_vector_array()).reshape(shape),
        keys=normalized_keys,
        components=normalized_components,
        stats=_integration_stats(result, mesh),
    )


def _certificate_from_native(certificate) -> SimplexCertificate:
    return SimplexCertificate(
        status=str(certificate.status),
        occupation_bounds=OccupationBounds(
            lower=int(certificate.occupation_bounds.lower),
            upper=int(certificate.occupation_bounds.upper),
        ),
        mu_interval=MuInterval(
            lower=float(certificate.mu_interval.lower),
            upper=float(certificate.mu_interval.upper),
        ),
        energy_bound=float(certificate.energy_bound),
    )


def certify_simplex(
    eigenvalues,
    eigenvectors,
    mu: float = 0.0,
    margin: float = 0.0,
    tol: float = _GEOM_TOL,
    estimate_occupation_bounds: bool = True,
) -> SimplexCertificate:
    _require_native_extension()
    if _native_certify_simplex is None:
        raise RuntimeError("Simplex certification requires the compiled native extension")
    values = np.ascontiguousarray(np.asarray(eigenvalues, dtype=np.float64))
    vectors = np.ascontiguousarray(np.asarray(eigenvectors, dtype=np.complex128))
    if values.ndim != 2:
        raise ValueError("eigenvalues must have shape (nvertices, ndof)")
    if vectors.ndim != 3:
        raise ValueError("eigenvectors must have shape (nvertices, ndof, ndof)")
    if vectors.shape != (values.shape[0], values.shape[1], values.shape[1]):
        raise ValueError("eigenvectors must have shape (nvertices, ndof, ndof)")
    certificate = _native_certify_simplex(
        values,
        vectors,
        float(mu),
        float(margin),
        float(tol),
        bool(estimate_occupation_bounds),
    )
    return _certificate_from_native(certificate)


def _fermi_stats_from_native(result, stats: dict[str, Any]) -> FermiSurfaceStats:
    return FermiSurfaceStats(
        evaluated_vertices=int(stats["evaluated_vertices"]),
        cut_simplices=int(result.n_cut_simplices),
        feature_size_simplices=int(result.n_feature_size_simplices),
        unresolved_simplices=int(result.n_unresolved_simplices),
    )


def _fermi_surface_from_native_result(
    result,
    *,
    stats: dict[str, Any],
    mu: float,
    min_feature_size: float,
    hessian_bound: float | None,
    hessian_bound_is_callable: bool,
    anharmonicity_bound: float,
    max_diagonalizations: int,
    ndim: int,
    ndof: int,
) -> FermiSurface:
    states = None
    if bool(result.has_states):
        states = FermiSurfaceStates(
            band_indices=np.asarray(result.state_band_indices_array()),
            eigenvalues=np.asarray(result.state_eigenvalues_array()),
            eigenvectors=np.asarray(result.state_eigenvectors_array()),
        )
    return FermiSurface(
        points=np.asarray(result.points_array()),
        cells=np.asarray(result.cells_array()),
        converged=bool(result.converged),
        stats=_fermi_stats_from_native(result, stats),
        parameters=FermiSurfaceParameters(
            mu=float(mu),
            min_feature_size=float(min_feature_size),
            hessian_bound=None if hessian_bound is None else float(hessian_bound),
            hessian_bound_is_callable=bool(hessian_bound_is_callable),
            anharmonicity_bound=float(anharmonicity_bound),
            max_diagonalizations=None if max_diagonalizations < 0 else max_diagonalizations,
            ndim=int(ndim),
            ndof=int(ndof),
        ),
        states=states,
    )


def fermi_surface(
    hamiltonian: SpectralMesh | Any,
    *,
    mu: float = 0.0,
    min_feature_size: float,
    max_diagonalizations: int | None = None,
    hessian_bound: Any = 0.0,
    anharmonicity_bound: float = 0.0,
    return_states: bool = False,
) -> FermiSurface:
    _require_native_extension()
    if (
        _native_fermi_surface is None
        or _native_fermi_surface_callable is None
        or _reset_fermi_surface_stats is None
        or _native_fermi_surface_stats is None
    ):
        raise RuntimeError("Fermi surface extraction requires the compiled native extension")
    feature_size = float(min_feature_size)
    if not np.isfinite(feature_size) or feature_size <= 0.0:
        raise ValueError("min_feature_size must be positive")
    hessian = _normalize_hessian_bound(hessian_bound)
    hessian_is_callable = callable(hessian)
    hessian_parameter = None if hessian_is_callable else float(hessian)
    anharmonicity = float(anharmonicity_bound)
    if not np.isfinite(anharmonicity) or anharmonicity < 0.0:
        raise ValueError("anharmonicity_bound must be finite and non-negative")
    max_diag_native = _normalize_max_diagonalizations(max_diagonalizations)
    if not isinstance(return_states, bool):
        raise TypeError("return_states must be a bool")

    if isinstance(hamiltonian, SpectralMesh):
        ndim = hamiltonian.ndim
        ndof = hamiltonian.ndof
        _reset_fermi_surface_stats()
        result = _native_fermi_surface_callable(
            hamiltonian._wrapped_callable,
            ndim,
            ndof,
            float(mu),
            feature_size,
            max_diag_native,
            hessian,
            anharmonicity,
            hamiltonian._tol,
            return_states,
        )
    elif callable(hamiltonian):
        wrapped, ndim, ndof = _prepare_callable_hamiltonian(hamiltonian)
        _reset_fermi_surface_stats()
        result = _native_fermi_surface_callable(
            wrapped,
            ndim,
            ndof,
            float(mu),
            feature_size,
            max_diag_native,
            hessian,
            anharmonicity,
            _GEOM_TOL,
            return_states,
        )
    else:
        raise TypeError("hamiltonian must be a SpectralMesh or callable")

    return _fermi_surface_from_native_result(
        result,
        stats=_native_fermi_surface_stats(),
        mu=float(mu),
        min_feature_size=feature_size,
        hessian_bound=hessian_parameter,
        hessian_bound_is_callable=hessian_is_callable,
        anharmonicity_bound=anharmonicity,
        max_diagonalizations=max_diag_native,
        ndim=ndim,
        ndof=ndof,
    )
