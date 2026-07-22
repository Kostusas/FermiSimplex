from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from fermisimplex import SpectralMesh


LOW_HARMONICS = (
    ((1, 0), 0.24),
    ((0, 1), 0.23),
    ((1, 1), 0.17),
    ((1, -1), 0.15),
    ((2, 1), 0.09),
    ((1, 2), 0.08),
)


@dataclass(frozen=True)
class BenchmarkResult:
    seconds: float
    points: int
    segments: int
    evaluations: int
    completed: bool
    coverage_certified: bool
    terminal_visible_simplices: int
    terminal_inconclusive_simplices: int


def add_hopping_pair(
    hoppings: dict[tuple[int, int], np.ndarray],
    lattice_vector: tuple[int, int],
    matrix: np.ndarray,
) -> None:
    opposite = tuple(-component for component in lattice_vector)
    hoppings[lattice_vector] = hoppings.get(
        lattice_vector, np.zeros_like(matrix)
    ) + matrix
    hoppings[opposite] = hoppings.get(
        opposite, np.zeros_like(matrix)
    ) + matrix.conj().T


def add_diagonal_pair(
    hoppings: dict[tuple[int, int], np.ndarray],
    lattice_vector: tuple[int, int],
    diagonal: np.ndarray,
) -> None:
    add_hopping_pair(hoppings, lattice_vector, np.diag(diagonal.astype(complex)))


def random_mixing(
    rng: np.random.Generator,
    ndof: int,
    strength: float,
) -> np.ndarray:
    matrix = rng.standard_normal((ndof, ndof))
    matrix = matrix + 1j * rng.standard_normal((ndof, ndof))
    np.fill_diagonal(matrix, 0.0)
    return strength * matrix / math.sqrt(ndof)


def stress_model(
    ndof: int = 60,
    seed: int = 4,
    pocket_bands: int = 10,
    mixing_strength: float = 0.035,
) -> dict[tuple[int, int], np.ndarray]:
    rng = np.random.default_rng(seed)
    offsets = np.linspace(-1.0, 1.0, ndof) + 0.04 * rng.standard_normal(ndof)
    hoppings = {(0, 0): np.diag(offsets.astype(complex))}

    for lattice_vector, scale in LOW_HARMONICS:
        amplitudes = scale * (0.35 + rng.random(ndof))
        amplitudes[:pocket_bands] = 0.0
        phases = rng.uniform(0.0, 2.0 * math.pi, ndof)
        add_diagonal_pair(
            hoppings,
            lattice_vector,
            0.5 * amplitudes * np.exp(1j * phases),
        )

    for band in range(pocket_bands):
        hoppings[(0, 0)][band, band] = 0.28 + 0.018 * band
        for lattice_vector, amplitude in (
            ((8, 0), 0.24 + 0.008 * band),
            ((0, 8), 0.22 + 0.006 * band),
            ((8, 8), 0.06),
        ):
            diagonal = np.zeros(ndof, dtype=complex)
            diagonal[band] = 0.5 * amplitude
            add_diagonal_pair(hoppings, lattice_vector, diagonal)

    mixing_rng = np.random.default_rng(seed + 1000)
    for lattice_vector, _ in LOW_HARMONICS:
        add_hopping_pair(
            hoppings,
            lattice_vector,
            random_mixing(mixing_rng, ndof, mixing_strength),
        )
    return hoppings


def conservative_curvature_bound(
    hoppings: dict[tuple[int, int], np.ndarray],
) -> float:
    return (2.0 * math.pi) ** 2 * sum(
        np.linalg.norm(matrix, ord="fro") * np.dot(vector, vector)
        for vector, matrix in hoppings.items()
    )


def run_benchmark(
    hoppings: dict[tuple[int, int], np.ndarray],
    mu: float,
    min_feature_size: float,
    max_evaluations: int | None,
    curvature_bound: float,
) -> BenchmarkResult:
    mesh = SpectralMesh(hoppings)
    started = time.perf_counter()
    surface = mesh.fermi_surface(
        mu=mu,
        min_feature_size=min_feature_size,
        max_evaluations=max_evaluations,
        curvature_bound=curvature_bound,
    )
    elapsed = time.perf_counter() - started
    return BenchmarkResult(
        seconds=elapsed,
        points=int(surface.points.shape[0]),
        segments=int(surface.cells.shape[0]),
        evaluations=int(surface.stats.evaluations),
        completed=bool(surface.completed),
        coverage_certified=bool(surface.coverage_certified),
        terminal_visible_simplices=int(surface.stats.terminal_visible_simplices),
        terminal_inconclusive_simplices=int(
            surface.stats.terminal_inconclusive_simplices
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--mixing-strength", type=float, default=0.035)
    parser.add_argument("--min-feature-size", type=float, default=0.01)
    parser.add_argument("--max-evaluations", type=int)
    parser.add_argument(
        "--curvature-bound",
        type=float,
        help="user bound M; defaults to a conservative Frobenius-norm bound",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    hoppings = stress_model(seed=args.seed, mixing_strength=args.mixing_strength)
    curvature_bound = args.curvature_bound
    if curvature_bound is None:
        curvature_bound = conservative_curvature_bound(hoppings)
    result = run_benchmark(
        hoppings,
        args.mu,
        args.min_feature_size,
        args.max_evaluations,
        curvature_bound,
    )
    summary = {
        "model": {
            "ndim": 2,
            "ndof": 60,
            "hopping_terms": len(hoppings),
            "seed": args.seed,
            "mixing_strength": args.mixing_strength,
        },
        "parameters": {
            "mu": args.mu,
            "min_feature_size": args.min_feature_size,
            "max_evaluations": args.max_evaluations,
            "curvature_bound": curvature_bound,
        },
        "result": asdict(result),
    }
    rendered = json.dumps(summary, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
