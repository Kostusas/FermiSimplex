from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from ._native import AdaptiveOptions, IntegrationRuntime, TightBindingModel
    from ._native import fermi_surface as _native_fermi_surface

    NATIVE_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when extension is unavailable
    AdaptiveOptions = None
    IntegrationRuntime = None
    TightBindingModel = None
    _native_fermi_surface = None
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
class FermiSurface:
    points: np.ndarray
    cells: np.ndarray
    converged: bool
    refinements: int
    n_active_simplices: int
    n_active_vertices: int
    n_safe_simplices: int
    n_cut_simplices: int
    n_feature_size_simplices: int
    n_unresolved_simplices: int
    min_feature_size: float


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


def _tb_to_general_tight_binding_model(tb: _tb_type):
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

    def evaluate_charge(self, mu: float, options):
        return self._native.evaluate_charge(float(mu), options)

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


def fermi_surface(
    h: _tb_type,
    *,
    mu: float,
    min_feature_size: float,
    max_refinements: int | None = None,
    tol: float = _GEOM_TOL,
) -> FermiSurface:
    _require_native_extension()
    if _native_fermi_surface is None:
        raise RuntimeError("Fermi surface extraction requires the compiled native extension")
    model = _tb_to_general_tight_binding_model(h)
    result = _native_fermi_surface(
        model,
        float(mu),
        float(min_feature_size),
        -1 if max_refinements is None else int(max_refinements),
        float(tol),
    )
    return FermiSurface(
        points=np.asarray(result.points_array()),
        cells=np.asarray(result.cells_array()),
        converged=bool(result.converged),
        refinements=int(result.refinements),
        n_active_simplices=int(result.n_active_simplices),
        n_active_vertices=int(result.n_active_vertices),
        n_safe_simplices=int(result.n_safe_simplices),
        n_cut_simplices=int(result.n_cut_simplices),
        n_feature_size_simplices=int(result.n_feature_size_simplices),
        n_unresolved_simplices=int(result.n_unresolved_simplices),
        min_feature_size=float(result.min_feature_size),
    )
