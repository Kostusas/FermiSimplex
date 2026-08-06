from __future__ import annotations

import argparse
import json
import math
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fermisimplex import SpectralMesh


Array = np.ndarray
Hoppings = dict[tuple[int, ...], Array]

CHARGE_ERROR_STATS = (
    "root_simplices",
    "hamiltonian_evaluations",
    "full_eigensystems",
    "reduced_eigensystems",
    "norm_eigensystems",
    "safe_block_solves",
    "schur_reductions",
    "micro_simplices",
    "terminal_simplices",
    "conservative_fallbacks",
    "singular_schur_failures",
    "initial_active_dimension_sum",
    "terminal_active_dimension_sum",
    "minimum_active_dimension",
)

INTEGRATION_STATS = (
    "evaluations",
    "simplex_visits",
    "refinements",
    "cached_vertices",
    "active_simplices",
    "active_vertices",
    "target_reached",
)

GRID_SHIFT = (math.sqrt(2.0) - 1.0, math.sqrt(3.0) - 1.0)


@dataclass(frozen=True)
class Model:
    name: str
    family: str
    hoppings: Hoppings
    active_hoppings: Hoppings
    fixed_occupied: int
    reference_key: str
    mu: float = 0.0

    @property
    def ndim(self) -> int:
        return len(next(iter(self.hoppings)))

    @property
    def ndof(self) -> int:
        return int(next(iter(self.hoppings.values())).shape[0])

    @property
    def active_dimension(self) -> int:
        return int(next(iter(self.active_hoppings.values())).shape[0])


def add_pair(hoppings: Hoppings, vector: tuple[int, ...], matrix: Array) -> None:
    matrix = np.asarray(matrix, dtype=complex)
    opposite = tuple(-value for value in vector)
    hoppings[vector] = hoppings.get(vector, np.zeros_like(matrix)) + matrix
    hoppings[opposite] = (
        hoppings.get(opposite, np.zeros_like(matrix)) + matrix.conj().T
    )


def add_cosine(hoppings: Hoppings, vector: tuple[int, ...], matrix: Array) -> None:
    add_pair(hoppings, vector, 0.5 * np.asarray(matrix, dtype=complex))


def embed(
    name: str,
    family: str,
    active_hoppings: Hoppings,
    ndof: int,
    reference_key: str,
) -> Model:
    active_dimension = next(iter(active_hoppings.values())).shape[0]
    if ndof < active_dimension:
        raise ValueError("embedding dimension is smaller than the active block")
    fixed_occupied = (ndof - active_dimension) // 2
    active_slice = slice(fixed_occupied, fixed_occupied + active_dimension)
    zero = (0,) * len(next(iter(active_hoppings)))

    onsite = np.zeros((ndof, ndof), dtype=complex)
    onsite[np.arange(fixed_occupied), np.arange(fixed_occupied)] = -4.0
    empty_start = fixed_occupied + active_dimension
    onsite[np.arange(empty_start, ndof), np.arange(empty_start, ndof)] = 4.0
    hoppings: Hoppings = {zero: onsite}
    for vector, block in active_hoppings.items():
        target = hoppings.setdefault(vector, np.zeros((ndof, ndof), dtype=complex))
        target[active_slice, active_slice] += block
    return Model(
        name=f"{name}_N{ndof}",
        family=family,
        hoppings=hoppings,
        active_hoppings=active_hoppings,
        fixed_occupied=fixed_occupied,
        reference_key=reference_key,
    )


def scalar_model() -> Model:
    hoppings: Hoppings = {(0,): np.array([[0.08]], dtype=complex)}
    add_cosine(hoppings, (1,), np.array([[0.55]]))
    phase = 0.37
    add_pair(
        hoppings,
        (2,),
        np.array([[0.06 * np.exp(1j * phase)]], dtype=complex),
    )
    return embed("smooth_scalar", "scalar", hoppings, 1, "smooth_scalar")


def avoided_active() -> Hoppings:
    identity = np.eye(2, dtype=complex)
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sigma_z = np.diag([1.0, -1.0]).astype(complex)
    hoppings: Hoppings = {(0, 0): 0.05 * identity + 0.06 * sigma_x}
    add_cosine(hoppings, (1, 0), 0.32 * sigma_z)
    add_cosine(hoppings, (0, 1), 0.46 * identity)
    return hoppings


def cluster_active() -> Hoppings:
    onsite = np.diag([-0.14, -0.04, 0.06, 0.16]).astype(complex)
    for index in range(3):
        onsite[index, index + 1] = 0.035
        onsite[index + 1, index] = 0.035
    hoppings: Hoppings = {(0, 0): onsite}
    add_cosine(hoppings, (0, 1), 0.48 * np.eye(4))
    add_cosine(hoppings, (1, 0), np.diag([0.14, -0.12, 0.10, -0.08]))
    mixing = np.zeros((4, 4), dtype=complex)
    mixing[0, 2] = mixing[2, 0] = 0.03
    mixing[1, 3] = mixing[3, 1] = -0.025
    add_cosine(hoppings, (1, 1), mixing)
    return hoppings


def coupled_core() -> Hoppings:
    """Two Fermi-level states coupled to four separated safe states."""
    onsite = np.diag([0.02, -0.04, -1.80, -1.20, 1.20, 1.80]).astype(complex)
    onsite[0, 1] = onsite[1, 0] = 0.035
    hoppings: Hoppings = {(0, 0): onsite}

    along_x = np.diag([0.42, -0.35, 0.05, -0.04, 0.03, -0.02]).astype(complex)
    along_x[0, 2] = along_x[2, 0] = 0.10
    along_x[1, 4] = along_x[4, 1] = 0.08
    add_cosine(hoppings, (1, 0), along_x)

    along_y = np.diag([0.36, 0.31, -0.03, 0.04, -0.04, 0.03]).astype(complex)
    along_y[0, 3] = along_y[3, 0] = -0.09
    along_y[1, 5] = along_y[5, 1] = 0.07
    add_cosine(hoppings, (0, 1), along_y)

    diagonal = np.zeros((6, 6), dtype=complex)
    diagonal[0, 4] = diagonal[4, 0] = 0.055
    diagonal[1, 2] = diagonal[2, 1] = -0.045
    add_pair(hoppings, (1, 1), -0.5j * diagonal)
    return hoppings


def gate_model(active_dimension: int, ndof: int, label: str) -> Model:
    offsets = np.linspace(-0.08, 0.08, active_dimension)
    active: Hoppings = {(0,): np.diag(offsets.astype(complex))}
    add_cosine(active, (1,), 0.5 * np.eye(active_dimension))
    return embed(label, "gate", active, ndof, f"gate_q{active_dimension}")


def alias_model(harmonic: int, label: str) -> Model:
    active: Hoppings = {(0,): np.array([[0.10]], dtype=complex)}
    add_cosine(active, (harmonic,), np.array([[0.30]]))
    return embed(label, "alias", active, 1, label)


def active_occupation(model: Model, grid_size: int, chunk_size: int = 65536) -> float:
    ndim = model.ndim
    total_points = grid_size**ndim
    occupied = 0
    powers = [grid_size**axis for axis in range(ndim)]
    shifts = GRID_SHIFT[:ndim]
    for start in range(0, total_points, chunk_size):
        flat = np.arange(start, min(start + chunk_size, total_points))
        points = np.empty((flat.size, ndim), dtype=float)
        for axis, power in enumerate(powers):
            index = (flat // power) % grid_size
            points[:, axis] = (index + shifts[axis]) / grid_size

        matrices = np.zeros(
            (flat.size, model.active_dimension, model.active_dimension),
            dtype=complex,
        )
        for vector, hopping in model.active_hoppings.items():
            phase = np.exp(2j * np.pi * (points @ np.asarray(vector)))
            matrices += phase[:, None, None] * hopping
        values = np.linalg.eigvalsh(matrices)
        occupied += int(np.count_nonzero(values <= model.mu))
    return occupied / total_points


def reference_charge(
    model: Model,
    coarse_grid: int,
    fine_grid: int,
) -> tuple[float, float]:
    coarse = active_occupation(model, coarse_grid)
    fine = active_occupation(model, fine_grid)
    return model.fixed_occupied + fine, abs(fine - coarse)


def public_stats(value: object, names: tuple[str, ...]) -> dict[str, int | bool]:
    return {name: getattr(value, name) for name in names}


def run_case(
    model: Model,
    suite: str,
    root_level: int,
    error_depth: int,
    reference: float,
    reference_delta: float,
    expectation: str | None = None,
) -> dict[str, object]:
    mesh = SpectralMesh(model.hoppings, root_level=root_level)
    started = time.perf_counter()
    result = mesh.estimate_charge_on_current_mesh(
        mu=model.mu,
        target_error=0.0,
        error_depth=error_depth,
    )
    elapsed = time.perf_counter() - started

    value = float(result.value)
    eta = float(result.stopping_error)
    signed_error = value - reference
    true_error = abs(signed_error)
    effectivity = (
        eta / true_error
        if true_error > max(1e-14, reference_delta)
        else None
    )
    stats = public_stats(result.error_stats, CHARGE_ERROR_STATS)
    roots = int(stats["root_simplices"])
    initial_q_mean = (
        float(stats["initial_active_dimension_sum"]) / roots if roots else 0.0
    )
    terminals = int(stats["terminal_simplices"])
    terminal_q_mean = (
        float(stats["terminal_active_dimension_sum"]) / terminals
        if terminals
        else 0.0
    )
    return {
        "suite": suite,
        "model": model.name,
        "family": model.family,
        "ndim": model.ndim,
        "ndof": model.ndof,
        "nominal_active_dimension": model.active_dimension,
        "fixed_occupied": model.fixed_occupied,
        "mu": model.mu,
        "root_level": root_level,
        "h": 2.0**(-root_level),
        "error_depth": error_depth,
        "expectation": expectation,
        "wall_seconds": elapsed,
        "reference_charge": reference,
        "reference_delta": reference_delta,
        "linear_charge": value,
        "signed_error": signed_error,
        "true_error": true_error,
        "eta": eta,
        "effectivity": effectivity,
        "compatible": true_error <= eta + 5e-12,
        "compatible_with_reference_delta": (
            true_error <= eta + reference_delta + 5e-12
        ),
        "visible_gapless_simplices": int(result.visible_gapless_simplices),
        "inconclusive_simplices": int(result.inconclusive_simplices),
        "initial_q_mean": initial_q_mean,
        "terminal_q_mean": terminal_q_mean,
        "charge_error_stats": stats,
        "integration_stats": public_stats(result.stats, INTEGRATION_STATS),
    }


def warm_up() -> None:
    mesh = SpectralMesh({(0,): np.array([[-1.0]])}, root_level=0)
    mesh.estimate_charge_on_current_mesh(
        mu=0.0,
        target_error=0.0,
        error_depth=0,
    )


def benchmark(preset: str, only: str) -> dict[str, object]:
    benchmark_started = time.perf_counter()
    quick = preset == "quick"
    reference_grids = {
        1: (32771, 65537) if quick else (131071, 524287),
        2: (113, 181) if quick else (257, 509),
    }
    cases: list[dict[str, object]] = []
    references: dict[str, tuple[float, float]] = {}

    def reference(model: Model) -> tuple[float, float]:
        if model.reference_key not in references:
            coarse, fine = reference_grids[model.ndim]
            active_reference, delta = reference_charge(model, coarse, fine)
            references[model.reference_key] = (
                active_reference - model.fixed_occupied,
                delta,
            )
        active_reference, delta = references[model.reference_key]
        return model.fixed_occupied + active_reference, delta

    if only in {"all", "scalar"}:
        model = scalar_model()
        ref, delta = reference(model)
        levels = range(2, 6) if quick else range(2, 8)
        depths = (0, 1) if quick else (0, 1, 2)
        for root_level in levels:
            for depth in depths:
                cases.append(
                    run_case(model, "scalar_scaling", root_level, depth, ref, delta)
                )

    if only in {"all", "embedded"}:
        depths = (1,) if quick else (0, 1, 2)
        for active_name, active in (
            ("avoided", avoided_active()),
            ("cluster", cluster_active()),
        ):
            for ndof in (16, 64, 128):
                model = embed(
                    active_name,
                    "embedded",
                    active,
                    ndof,
                    active_name,
                )
                ref, delta = reference(model)
                for depth in depths:
                    cases.append(
                        run_case(
                            model,
                            "embedded_scaling",
                            1,
                            depth,
                            ref,
                            delta,
                        )
                    )

    if only in {"all", "coupled"}:
        depths = (1,) if quick else (0, 1, 2)
        core = coupled_core()
        for ndof in (16, 64, 128):
            model = embed("coupled", "coupled", core, ndof, "coupled")
            ref, delta = reference(model)
            for depth in depths:
                cases.append(
                    run_case(
                        model,
                        "coupled_scaling",
                        1,
                        depth,
                        ref,
                        delta,
                        "nonzero active-safe Schur correction",
                    )
                )

    if only in {"all", "gate"}:
        gate_cases = (
            (gate_model(2, 2, "small_full"), "q=2 always recurses"),
            (gate_model(3, 8, "below_half"), "2q<N recurses"),
            (gate_model(4, 8, "half_gate"), "q>2 and 2q>=N falls back"),
            (gate_model(3, 5, "ceil_half_gate"), "integer half gate falls back"),
        )
        for model, expectation in gate_cases:
            ref, delta = reference(model)
            cases.append(
                run_case(model, "root_gate", 1, 1, ref, delta, expectation)
            )

    if only in {"all", "alias"}:
        alias_cases = (
            (alias_model(2, "visible_alias"), "edge midpoint exposes pocket"),
            (alias_model(4, "exact_dyadic_alias"), "all sampled dyadic points alias"),
        )
        depths = (1,) if quick else (0, 1, 2)
        for model, expectation in alias_cases:
            ref, delta = reference(model)
            for depth in depths:
                cases.append(
                    run_case(model, "aliasing", 1, depth, ref, delta, expectation)
                )

    compatible = sum(bool(case["compatible"]) for case in cases)
    finite_effectivities = [
        float(case["effectivity"])
        for case in cases
        if case["effectivity"] is not None
    ]
    estimator_wall_seconds = sum(
        float(case["wall_seconds"]) for case in cases
    )
    total_wall_seconds = time.perf_counter() - benchmark_started
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "preset": preset,
        "only": only,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "reference_grids": {
            f"{dimension}d": {"coarse": grids[0], "fine": grids[1]}
            for dimension, grids in reference_grids.items()
        },
        "summary": {
            "cases": len(cases),
            "estimator_wall_seconds": estimator_wall_seconds,
            "total_wall_seconds": total_wall_seconds,
            "compatible_cases": compatible,
            "underestimated_cases": len(cases) - compatible,
            "median_effectivity": (
                float(np.median(finite_effectivities))
                if finite_effectivities
                else None
            ),
            "minimum_effectivity": (
                min(finite_effectivities) if finite_effectivities else None
            ),
        },
        "cases": cases,
    }


def plot_scalar(summary: dict[str, object], output: Path) -> None:
    import matplotlib.pyplot as plt

    cases = [
        case
        for case in summary["cases"]
        if case["suite"] == "scalar_scaling"
    ]
    if not cases:
        raise RuntimeError("the selected benchmark has no scalar scaling cases")
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    for depth in sorted({int(case["error_depth"]) for case in cases}):
        selected = sorted(
            (case for case in cases if case["error_depth"] == depth),
            key=lambda case: case["h"],
        )
        h = [case["h"] for case in selected]
        axes[0].loglog(
            h,
            [case["true_error"] for case in selected],
            "o-",
            label=f"true d={depth}",
        )
        axes[0].loglog(
            h,
            [case["eta"] for case in selected],
            "s--",
            label=f"eta d={depth}",
        )
        axes[1].semilogx(
            h,
            [case["effectivity"] for case in selected],
            "o-",
            label=f"d={depth}",
        )
    axes[0].set(xlabel="h", ylabel="charge error")
    axes[1].set(xlabel="h", ylabel="eta / true error")
    axes[0].legend(fontsize=7)
    axes[1].axhline(1.0, color="black", linewidth=0.8)
    axes[1].legend(fontsize=7)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the public recursive C++ charge-error estimator."
    )
    parser.add_argument("--preset", choices=("quick", "full"), default="quick")
    parser.add_argument(
        "--only",
        choices=("all", "scalar", "embedded", "coupled", "gate", "alias"),
        default="all",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    if args.plot is not None and args.only not in {"all", "scalar"}:
        parser.error("--plot requires --only all or scalar")

    warm_up()
    summary = benchmark(args.preset, args.only)
    if args.plot is not None:
        plot_scalar(summary, args.plot)
    rendered = json.dumps(summary, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
