from __future__ import annotations

import argparse
import html
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from lineartetrahedron import FermiSurface, fermi_surface


@dataclass(frozen=True)
class CaseResult:
    use_weyl_bounds: bool
    seconds: float
    n_points: int
    n_segments: int
    converged: bool
    refinements: int
    n_active_simplices: int
    n_active_vertices: int
    n_unresolved_simplices: int


def _add_diagonal_hopping_pair(
    tb: dict[tuple[int, int], np.ndarray],
    key: tuple[int, int],
    diagonal: np.ndarray,
) -> None:
    matrix = np.diag(np.asarray(diagonal, dtype=np.complex128))
    opposite = (-key[0], -key[1])
    tb[key] = tb.get(key, np.zeros_like(matrix)) + matrix
    tb[opposite] = tb.get(opposite, np.zeros_like(matrix)) + matrix.conj().T


def make_stress_hamiltonian(
    *,
    ndof: int = 60,
    seed: int = 4,
    pocket_bands: int = 10,
) -> dict[tuple[int, int], np.ndarray]:
    """Build a deterministic 2D 60x60 TB model with ordinary and aliased pockets."""
    rng = np.random.default_rng(seed)
    offsets = np.linspace(-1.0, 1.0, ndof) + 0.04 * rng.standard_normal(ndof)
    tb: dict[tuple[int, int], np.ndarray] = {
        (0, 0): np.diag(offsets.astype(np.complex128))
    }

    low_harmonics = [
        ((1, 0), 0.24),
        ((0, 1), 0.23),
        ((1, 1), 0.17),
        ((1, -1), 0.15),
        ((2, 1), 0.09),
        ((1, 2), 0.08),
    ]
    for key, scale in low_harmonics:
        amplitudes = scale * (0.35 + rng.random(ndof))
        amplitudes[:pocket_bands] = 0.0
        phases = rng.uniform(0.0, 2.0 * math.pi, ndof)
        _add_diagonal_hopping_pair(tb, key, 0.5 * amplitudes * np.exp(1j * phases))

    # These bands are deliberately undersampled by the initial root mesh. Without
    # Weyl marking they are invisible unless another band happens to refine there.
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

    return tb


def run_case(
    hamiltonian: dict[tuple[int, int], np.ndarray],
    *,
    mu: float,
    min_feature_size: float,
    max_refinements: int,
    use_weyl_bounds: bool,
) -> tuple[FermiSurface, CaseResult]:
    start = time.perf_counter()
    surface = fermi_surface(
        hamiltonian,
        mu=mu,
        min_feature_size=min_feature_size,
        max_refinements=max_refinements,
        use_weyl_bounds=use_weyl_bounds,
    )
    seconds = time.perf_counter() - start
    return surface, CaseResult(
        use_weyl_bounds=use_weyl_bounds,
        seconds=seconds,
        n_points=int(surface.points.shape[0]),
        n_segments=int(surface.cells.shape[0]),
        converged=bool(surface.converged),
        refinements=int(surface.refinements),
        n_active_simplices=int(surface.n_active_simplices),
        n_active_vertices=int(surface.n_active_vertices),
        n_unresolved_simplices=int(surface.n_unresolved_simplices),
    )


def _segment_lines(surface: FermiSurface) -> list[tuple[float, float, float, float]]:
    points = np.asarray(surface.points)
    cells = np.asarray(surface.cells)
    return [
        (
            float(points[cell[0], 0]),
            float(points[cell[0], 1]),
            float(points[cell[1], 0]),
            float(points[cell[1], 1]),
        )
        for cell in cells
    ]


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
        f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" '
        f'stroke="{color}" stroke-width="{width:.3f}" '
        f'stroke-opacity="{opacity:.3f}" stroke-linecap="round" />'
    )


def write_svg(
    output: Path,
    *,
    no_weyl: FermiSurface,
    with_weyl: FermiSurface,
    no_weyl_result: CaseResult,
    with_weyl_result: CaseResult,
    mu: float,
    min_feature_size: float,
    max_refinements: int,
) -> None:
    width = 1120
    height = 600
    panel = 460
    margin = 70
    gap = 70
    top = 95
    plot_size = 390
    left_x = margin
    right_x = margin + panel + gap

    def map_point(kx: float, ky: float, origin_x: float) -> tuple[float, float]:
        x = origin_x + (kx + math.pi) / (2.0 * math.pi) * plot_size
        y = top + (math.pi - ky) / (2.0 * math.pi) * plot_size
        return x, y

    def panel_lines(surface: FermiSurface, origin_x: float, color: str) -> list[str]:
        lines: list[str] = []
        for kx1, ky1, kx2, ky2 in _segment_lines(surface):
            x1, y1 = map_point(kx1, ky1, origin_x)
            x2, y2 = map_point(kx2, ky2, origin_x)
            lines.append(_svg_line(x1, y1, x2, y2, color=color, width=1.15, opacity=0.82))
        return lines

    def frame(origin_x: float) -> list[str]:
        x0 = origin_x
        y0 = top
        x1 = origin_x + plot_size
        y1 = top + plot_size
        center_x = origin_x + 0.5 * plot_size
        center_y = top + 0.5 * plot_size
        return [
            f'<rect x="{x0:.3f}" y="{y0:.3f}" width="{plot_size:.3f}" '
            f'height="{plot_size:.3f}" fill="none" stroke="#222" stroke-width="1.2" />',
            _svg_line(center_x, y0, center_x, y1, color="#d0d0d0", width=0.8),
            _svg_line(x0, center_y, x1, center_y, color="#d0d0d0", width=0.8),
        ]

    def label(text: str, x: float, y: float, size: int = 16, weight: int = 400) -> str:
        return (
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
            f'font-size="{size}" font-weight="{weight}" fill="#111">'
            f'{html.escape(text)}</text>'
        )

    def stats(result: CaseResult) -> str:
        return (
            f'{result.seconds:.3f}s, segments={result.n_segments}, '
            f'refined={result.refinements}, unresolved={result.n_unresolved_simplices}'
        )

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff" />',
        label("60x60 adaptive Fermi surface extraction", 42, 42, size=24, weight=700),
        label(
            f"mu={mu:g}, min_feature_size={min_feature_size:g}, "
            f"max_refinements={max_refinements}",
            42,
            68,
            size=14,
        ),
        label("without Weyl marking", left_x, 82, size=18, weight=700),
        label(stats(no_weyl_result), left_x, 525, size=13),
        label("with Hessian-Weyl marking", right_x, 82, size=18, weight=700),
        label(stats(with_weyl_result), right_x, 525, size=13),
        *frame(left_x),
        *frame(right_x),
        *panel_lines(no_weyl, left_x, "#303030"),
        *panel_lines(with_weyl, right_x, "#b72222"),
        label("-pi", left_x - 9, top + plot_size + 24, size=12),
        label("pi", left_x + plot_size - 8, top + plot_size + 24, size=12),
        label("-pi", right_x - 9, top + plot_size + 24, size=12),
        label("pi", right_x + plot_size - 8, top + plot_size + 24, size=12),
        label("physical kx", left_x + 160, top + plot_size + 48, size=13),
        label("physical kx", right_x + 160, top + plot_size + 48, size=13),
        label("ky", left_x - 35, top + 202, size=13),
        label("ky", right_x - 35, top + 202, size=13),
        '</svg>',
    ]
    output.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--min-feature-size", type=float, default=1.8)
    parser.add_argument("--max-refinements", type=int, default=800)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    hamiltonian = make_stress_hamiltonian()
    no_weyl, no_weyl_result = run_case(
        hamiltonian,
        mu=args.mu,
        min_feature_size=args.min_feature_size,
        max_refinements=args.max_refinements,
        use_weyl_bounds=False,
    )
    with_weyl, with_weyl_result = run_case(
        hamiltonian,
        mu=args.mu,
        min_feature_size=args.min_feature_size,
        max_refinements=args.max_refinements,
        use_weyl_bounds=True,
    )

    svg_path = args.output_dir / "fermi_surface_60.svg"
    summary_path = args.output_dir / "fermi_surface_60_summary.json"
    write_svg(
        svg_path,
        no_weyl=no_weyl,
        with_weyl=with_weyl,
        no_weyl_result=no_weyl_result,
        with_weyl_result=with_weyl_result,
        mu=args.mu,
        min_feature_size=args.min_feature_size,
        max_refinements=args.max_refinements,
    )
    summary = {
        "model": {
            "ndim": 2,
            "ndof": 60,
            "n_hopping_terms": len(hamiltonian),
            "pocket_bands": 10,
            "seed": 4,
        },
        "parameters": {
            "mu": args.mu,
            "min_feature_size": args.min_feature_size,
            "max_refinements": args.max_refinements,
        },
        "without_weyl": asdict(no_weyl_result),
        "with_weyl": asdict(with_weyl_result),
        "svg": str(svg_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
