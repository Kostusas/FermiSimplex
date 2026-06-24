from __future__ import annotations

import argparse
import html
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from lineartetrahedron import FermiSurface, fermi_surface

_DEFAULT_NORMALIZED_FEATURE_SIZE = 0.01
_DEFAULT_OUTPUT_DIR = Path("/private/tmp/lineartetrahedron_benchmarks")
_LOW_HARMONICS = [
    ((1, 0), 0.24),
    ((0, 1), 0.23),
    ((1, 1), 0.17),
    ((1, -1), 0.15),
    ((2, 1), 0.09),
    ((1, 2), 0.08),
]


@dataclass(frozen=True)
class CaseResult:
    n_points: int
    n_segments: int
    converged: bool
    n_cut_simplices: int
    n_feature_size_simplices: int
    n_unresolved_simplices: int
    evaluated_vertices: int


def _add_hopping_pair(
    tb: dict[tuple[int, int], np.ndarray],
    key: tuple[int, int],
    matrix: np.ndarray,
) -> None:
    opposite = (-key[0], -key[1])
    tb[key] = tb.get(key, np.zeros_like(matrix)) + matrix
    tb[opposite] = tb.get(opposite, np.zeros_like(matrix)) + matrix.conj().T


def _add_diagonal_hopping_pair(
    tb: dict[tuple[int, int], np.ndarray],
    key: tuple[int, int],
    diagonal: np.ndarray,
) -> None:
    _add_hopping_pair(tb, key, np.diag(np.asarray(diagonal, dtype=np.complex128)))


def _dense_offdiagonal(
    rng: np.random.Generator,
    *,
    ndof: int,
    strength: float,
) -> np.ndarray:
    matrix = rng.standard_normal((ndof, ndof)) + 1j * rng.standard_normal((ndof, ndof))
    np.fill_diagonal(matrix, 0.0)
    return (strength / math.sqrt(ndof)) * matrix.astype(np.complex128)


def make_stress_hamiltonian(
    *,
    ndof: int = 60,
    seed: int = 4,
    pocket_bands: int = 10,
    mixing_strength: float = 0.035,
) -> dict[tuple[int, int], np.ndarray]:
    """Build the maintained dense-mixed 60x60 Fermi-surface stress model."""
    rng = np.random.default_rng(seed)
    offsets = np.linspace(-1.0, 1.0, ndof) + 0.04 * rng.standard_normal(ndof)
    tb: dict[tuple[int, int], np.ndarray] = {
        (0, 0): np.diag(offsets.astype(np.complex128))
    }

    for key, scale in _LOW_HARMONICS:
        amplitudes = scale * (0.35 + rng.random(ndof))
        amplitudes[:pocket_bands] = 0.0
        phases = rng.uniform(0.0, 2.0 * math.pi, ndof)
        _add_diagonal_hopping_pair(tb, key, 0.5 * amplitudes * np.exp(1j * phases))

    for band in range(pocket_bands):
        tb[(0, 0)][band, band] = 0.28 + 0.018 * band
        for key, amplitude in [
            ((8, 0), 0.24 + 0.008 * band),
            ((0, 8), 0.22 + 0.006 * band),
            ((8, 8), 0.06),
        ]:
            diagonal = np.zeros(ndof, dtype=np.complex128)
            diagonal[band] = 0.5 * amplitude
            _add_diagonal_hopping_pair(tb, key, diagonal)

    mixing_rng = np.random.default_rng(seed + 1000)
    for key, _scale in _LOW_HARMONICS:
        _add_hopping_pair(
            tb,
            key,
            _dense_offdiagonal(mixing_rng, ndof=ndof, strength=mixing_strength),
        )

    return tb


def run_case(
    hamiltonian: dict[tuple[int, int], np.ndarray],
    *,
    mu: float,
    min_feature_size_normalized: float,
    max_diagonalizations: int | None,
) -> tuple[FermiSurface, CaseResult]:
    surface = fermi_surface(
        hamiltonian,
        mu=mu,
        min_feature_size=min_feature_size_normalized,
        max_diagonalizations=max_diagonalizations,
    )
    stats = surface.stats
    return surface, CaseResult(
        n_points=int(surface.points.shape[0]),
        n_segments=int(surface.cells.shape[0]),
        converged=bool(surface.converged),
        n_cut_simplices=int(stats.cut_simplices),
        n_feature_size_simplices=int(stats.feature_size_simplices),
        n_unresolved_simplices=int(stats.unresolved_simplices),
        evaluated_vertices=int(stats.evaluated_vertices),
    )


def _svg_line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str,
    width: float,
    opacity: float = 1.0,
) -> str:
    return (
        f'<line x1="{x1:.4f}" y1="{y1:.4f}" x2="{x2:.4f}" y2="{y2:.4f}" '
        f'stroke="{color}" stroke-width="{width:.3f}" '
        f'stroke-opacity="{opacity:.3f}" stroke-linecap="round" />'
    )


def _svg_text(
    value: str,
    x: float,
    y: float,
    *,
    size: int = 13,
    weight: int = 400,
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="#111" '
        f'text-anchor="{anchor}">{html.escape(value)}</text>'
    )


def write_svg(
    output: Path,
    *,
    surface: FermiSurface,
    result: CaseResult,
    mu: float,
    min_feature_size_normalized: float,
    max_diagonalizations: int | None,
    seed: int,
    mixing_strength: float,
) -> None:
    width = 900
    height = 860
    left = 92
    top = 110
    plot_size = 690
    points = np.asarray(surface.points)
    cells = np.asarray(surface.cells)

    def map_point(kx: float, ky: float) -> tuple[float, float]:
        x_physical = 2.0 * math.pi * kx - math.pi
        y_physical = 2.0 * math.pi * ky - math.pi
        x = left + (x_physical + math.pi) / (2.0 * math.pi) * plot_size
        y = top + (math.pi - y_physical) / (2.0 * math.pi) * plot_size
        return x, y

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff" />',
        _svg_text("Adaptive Fermi surface: dense-mixed random 60x60", 38, 42, size=24, weight=700),
        _svg_text(
            f"seed={seed}, mixing_strength={mixing_strength:g}, mu={mu:g}, "
            f"normalized feature={min_feature_size_normalized:g}, "
            f"max_diagonalizations={max_diagonalizations}",
            38,
            70,
            size=14,
        ),
        _svg_text(
            f"segments={result.n_segments}, evaluated vertices={result.evaluated_vertices}, "
            f"converged={result.converged}",
            38,
            92,
            size=14,
        ),
        f'<rect x="{left}" y="{top}" width="{plot_size}" height="{plot_size}" '
        'fill="none" stroke="#222" stroke-width="1" />',
        _svg_line(
            left + plot_size / 2,
            top,
            left + plot_size / 2,
            top + plot_size,
            color="#d6d6d6",
            width=0.7,
        ),
        _svg_line(
            left,
            top + plot_size / 2,
            left + plot_size,
            top + plot_size / 2,
            color="#d6d6d6",
            width=0.7,
        ),
    ]

    for start, end in cells:
        x1, y1 = map_point(float(points[start, 0]), float(points[start, 1]))
        x2, y2 = map_point(float(points[end, 0]), float(points[end, 1]))
        parts.append(_svg_line(x1, y1, x2, y2, color="#c91f1f", width=0.48, opacity=0.92))

    parts.extend([
        _svg_text("-pi", left - 10, top + plot_size + 24, size=12),
        _svg_text("0", left + plot_size / 2, top + plot_size + 24, size=12, anchor="middle"),
        _svg_text("pi", left + plot_size - 8, top + plot_size + 24, size=12),
        _svg_text("-pi", left - 38, top + plot_size + 4, size=12),
        _svg_text("0", left - 24, top + plot_size / 2 + 4, size=12),
        _svg_text("pi", left - 28, top + 4, size=12),
        _svg_text("kx", left + plot_size / 2, top + plot_size + 52, size=13, anchor="middle"),
        _svg_text("ky", left - 54, top + plot_size / 2, size=13, anchor="middle"),
        _svg_text(
            f"cut={result.n_cut_simplices}; "
            f"feature-size={result.n_feature_size_simplices}; unresolved={result.n_unresolved_simplices}",
            38,
            height - 24,
            size=13,
        ),
        "</svg>",
    ])
    output.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--mixing-strength", type=float, default=0.035)
    parser.add_argument("--min-feature-size", type=float, default=_DEFAULT_NORMALIZED_FEATURE_SIZE)
    parser.add_argument("--max-diagonalizations", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    hamiltonian = make_stress_hamiltonian(
        seed=args.seed,
        mixing_strength=args.mixing_strength,
    )
    surface, result = run_case(
        hamiltonian,
        mu=args.mu,
        min_feature_size_normalized=args.min_feature_size,
        max_diagonalizations=args.max_diagonalizations,
    )

    stem = "fermi_surface_60_dense_mixed_rotated_inertia"
    svg_path = args.output_dir / f"{stem}.svg"
    summary_path = args.output_dir / f"{stem}_summary.json"
    write_svg(
        svg_path,
        surface=surface,
        result=result,
        mu=args.mu,
        min_feature_size_normalized=args.min_feature_size,
        max_diagonalizations=args.max_diagonalizations,
        seed=args.seed,
        mixing_strength=args.mixing_strength,
    )

    summary = {
        "model": {
            "kind": "60x60 stress model plus dense off-diagonal low-harmonic hoppings",
            "ndim": 2,
            "ndof": 60,
            "n_hopping_terms": len(hamiltonian),
            "pocket_bands": 10,
            "seed": args.seed,
            "mixing_strength": args.mixing_strength,
        },
        "parameters": {
            "mu": args.mu,
            "min_feature_size_normalized": args.min_feature_size,
            "max_diagonalizations": args.max_diagonalizations,
        },
        "result": asdict(result),
        "svg": str(svg_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
