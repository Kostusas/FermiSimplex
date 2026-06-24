from __future__ import annotations

import inspect
import operator
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from ._native import AdaptiveOptions, IntegrationRuntime, TightBindingModel
    from ._native import _fermi_surface_stats as _native_fermi_surface_stats
    from ._native import _reset_fermi_surface_stats
    from ._native import fermi_surface as _native_fermi_surface
    from ._native import fermi_surface_callable as _native_fermi_surface_callable

    NATIVE_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when extension is unavailable
    AdaptiveOptions = None
    IntegrationRuntime = None
    TightBindingModel = None
    _native_fermi_surface_stats = None
    _reset_fermi_surface_stats = None
    _native_fermi_surface = None
    _native_fermi_surface_callable = None
    NATIVE_AVAILABLE = False

_tb_type = dict[tuple[int, ...], np.ndarray]
_GEOM_TOL = 1e-14


@dataclass(frozen=True)
class DensityIntegrationInfo:
    n_evaluator_evals: int
    n_cached_nodes: int
    n_active_simplices: int
    n_active_vertices: int
    subdivisions: int


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
    margin: float
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
    _validate_dense_tb(tb)
    keys = np.asarray([tuple(key) for key in tb.keys()], dtype=np.int64, order="C")
    matrices = np.asarray([np.asarray(value) for value in tb.values()], dtype=np.complex128, order="C")
    return TightBindingModel(keys, matrices)


def _build_native_runtime(
    tb: _tb_type,
    *,
    keys: list[tuple[int, ...]],
    component_rows: np.ndarray,
    component_cols: np.ndarray,
    component_key_indices: np.ndarray,
    tol: float,
):
    _require_native_extension()
    model = _tb_to_tight_binding_model(tb)
    return IntegrationRuntime(
        model,
        np.ascontiguousarray(np.asarray(keys, dtype=np.int64)),
        np.ascontiguousarray(np.asarray(component_rows, dtype=np.int64)),
        np.ascontiguousarray(np.asarray(component_cols, dtype=np.int64)),
        np.ascontiguousarray(np.asarray(component_key_indices, dtype=np.int64)),
        float(tol),
    )


def _full_density_components(
    keys: list[tuple[int, ...]],
    *,
    size: int,
) -> list[tuple[int, int, tuple[int, ...]]]:
    grid = np.arange(int(size), dtype=np.int64)
    rows, cols = np.meshgrid(grid, grid, indexing="ij")
    components: list[tuple[int, int, tuple[int, ...]]] = []
    for key in keys:
        normalized_key = tuple(int(part) for part in key)
        components.extend(
            (int(row), int(col), normalized_key)
            for row, col in zip(rows.reshape(-1), cols.reshape(-1), strict=True)
        )
    return components


class _PreparedDensityComponents:
    def __init__(
        self,
        *,
        size: int,
        keys: list[tuple[int, ...]],
        rows: np.ndarray,
        cols: np.ndarray,
        key_indices: np.ndarray,
    ) -> None:
        self.size = int(size)
        self.keys = tuple(keys)
        self.rows = np.asarray(rows, dtype=np.int64)
        self.cols = np.asarray(cols, dtype=np.int64)
        self.key_indices = np.asarray(key_indices, dtype=np.int64)

    @property
    def value_count(self) -> int:
        return int(self.rows.size)

    def values_to_tb(self, values: np.ndarray) -> _tb_type:
        values = np.asarray(values)
        rho = {
            key: np.zeros((self.size, self.size), dtype=complex) for key in self.keys
        }
        for index, value in enumerate(values):
            key = self.keys[int(self.key_indices[index])]
            rho[key][int(self.rows[index]), int(self.cols[index])] = value
        return rho

    def values_and_errors_to_tb(
        self,
        values: np.ndarray,
        errors: np.ndarray,
    ) -> tuple[_tb_type, _tb_type]:
        rho = self.values_to_tb(values)
        errors = np.asarray(errors)
        error_dtype = errors.dtype if errors.size else float
        rho_error = {
            key: np.zeros((self.size, self.size), dtype=error_dtype)
            for key in self.keys
        }
        for index, error in enumerate(errors):
            key = self.keys[int(self.key_indices[index])]
            rho_error[key][int(self.rows[index]), int(self.cols[index])] = error
        return rho, rho_error


def _prepare_density_components(
    h: _tb_type,
    keys: list[tuple[int, ...]],
    density_components,
) -> _PreparedDensityComponents:
    size = int(next(iter(h.values())).shape[0])
    normalized_keys = [tuple(int(part) for part in key) for key in keys]
    if not normalized_keys:
        raise ValueError("keys must be non-empty")
    if len(set(normalized_keys)) != len(normalized_keys):
        raise ValueError("keys must be unique")
    ndim = len(normalized_keys[0])
    if any(len(key) != ndim for key in normalized_keys):
        raise ValueError("All keys must have the same dimension")
    if density_components is None:
        density_components = _full_density_components(normalized_keys, size=size)
    key_index_by_key = {key: index for index, key in enumerate(normalized_keys)}

    rows: list[int] = []
    cols: list[int] = []
    key_indices: list[int] = []
    seen = set()
    for component in density_components:
        if len(component) != 3:
            raise ValueError("density components must be (row, col, key) triples")
        row, col, key = component
        row = int(row)
        col = int(col)
        key = tuple(int(part) for part in key)
        if len(key) != ndim:
            raise ValueError("density component key dimension does not match keys")
        if key not in key_index_by_key:
            raise ValueError(f"density component key {key!r} is not in keys")
        if row < 0 or col < 0 or row >= size or col >= size:
            raise ValueError("density component row/column out of bounds")
        dedupe_key = (row, col, key)
        if dedupe_key in seen:
            raise ValueError(f"duplicate density component {dedupe_key!r}")
        seen.add(dedupe_key)
        rows.append(row)
        cols.append(col)
        key_indices.append(key_index_by_key[key])

    return _PreparedDensityComponents(
        size=size,
        keys=normalized_keys,
        rows=np.asarray(rows, dtype=np.int64),
        cols=np.asarray(cols, dtype=np.int64),
        key_indices=np.asarray(key_indices, dtype=np.int64),
    )


def _density_info(result, runtime) -> DensityIntegrationInfo:
    return DensityIntegrationInfo(
        n_evaluator_evals=int(result.work),
        n_cached_nodes=int(runtime.n_cached_nodes),
        n_active_simplices=int(result.n_active_simplices),
        n_active_vertices=int(result.n_active_vertices),
        subdivisions=int(result.refinements),
    )


class Runtime:
    def __init__(
        self,
        h: _tb_type,
        *,
        keys: list[tuple[int, ...]],
        density_components=None,
        tol: float = _GEOM_TOL,
    ) -> None:
        _validate_dense_tb(h)
        self._prepared = _prepare_density_components(h, keys, density_components)
        self._native = _build_native_runtime(
            h,
            keys=list(self._prepared.keys),
            component_rows=self._prepared.rows,
            component_cols=self._prepared.cols,
            component_key_indices=self._prepared.key_indices,
            tol=tol,
        )

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

    def integrate_charge(self, mu: float, options):
        return self._native.integrate_charge(float(mu), options)

    def evaluate_charge(self, mu: float, options, *, certify: bool = True):
        return self._native.evaluate_charge(float(mu), options, bool(certify))

    def integrate_density(self, mu: float, options):
        result = self._native.integrate_density(float(mu), options)
        rho, error = self._prepared.values_and_errors_to_tb(
            result.estimate_array(),
            result.error_vector_array(),
        )
        return rho, error, _density_info(result, self._native)


def density_matrix_at_mu_zero_temp(
    h: _tb_type,
    *,
    mu: float,
    keys: list[tuple[int, ...]],
    density_atol: float,
    density_components=None,
    max_subdivisions: int | None = None,
    adaptive_options=None,
):
    runtime = Runtime(h, keys=keys, density_components=density_components)
    options = adaptive_options
    if options is None:
        options = AdaptiveOptions(
            target_error=float(density_atol),
            max_refinements=-1 if max_subdivisions is None else int(max_subdivisions),
        )
    return runtime.integrate_density(float(mu), options)


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


def _fermi_stats_from_native(result, stats: dict[str, Any]) -> FermiSurfaceStats:
    return FermiSurfaceStats(
        evaluated_vertices=int(stats["evaluated_vertices"]),
        cut_simplices=int(result.n_cut_simplices),
        feature_size_simplices=int(result.n_feature_size_simplices),
        unresolved_simplices=int(result.n_unresolved_simplices),
    )


def fermi_surface(
    hamiltonian: _tb_type | Any,
    *,
    mu: float = 0.0,
    min_feature_size: float,
    max_diagonalizations: int | None = None,
    margin: float | None = 0.0,
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
    certificate_margin = 0.0 if margin is None else float(margin)
    if not np.isfinite(certificate_margin) or certificate_margin < 0.0:
        raise ValueError("margin must be finite and non-negative")
    max_diag_native = _normalize_max_diagonalizations(max_diagonalizations)
    if not isinstance(return_states, bool):
        raise TypeError("return_states must be a bool")

    if isinstance(hamiltonian, dict):
        model = _tb_to_tight_binding_model(hamiltonian)
        ndim = int(model.ndim)
        ndof = int(model.ndof)
        _reset_fermi_surface_stats()
        result = _native_fermi_surface(
            model,
            float(mu),
            feature_size,
            max_diag_native,
            certificate_margin,
            _GEOM_TOL,
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
            certificate_margin,
            _GEOM_TOL,
            return_states,
        )
    else:
        raise TypeError("hamiltonian must be a tight-binding dict or a callable")

    native_stats = _native_fermi_surface_stats()
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
        stats=_fermi_stats_from_native(result, native_stats),
        parameters=FermiSurfaceParameters(
            mu=float(mu),
            min_feature_size=feature_size,
            margin=certificate_margin,
            max_diagonalizations=None if max_diag_native < 0 else max_diag_native,
            ndim=ndim,
            ndof=ndof,
        ),
        states=states,
    )
