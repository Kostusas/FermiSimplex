"""Generate the minimal, looping public-API gallery for docs/showcase.md."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(ROOT / "docs" / "generated" / "matplotlib"),
)

import matplotlib.tri as mtri
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

from fermisimplex import SpectralMesh


ASSETS = ROOT / "docs" / "assets"
BACKGROUND = "#07111f"
FOREGROUND = "#e6edf7"
MUTED = "#8fa3bd"
GRID = "#26364b"
AMBER = "#ffb000"
MAGENTA = "#ff5dce"
GREEN = "#72e0a3"
BAND_COLORS = (AMBER, MAGENTA, GREEN)


@dataclass(frozen=True)
class Model2D:
    title: str
    hamiltonian: Callable[[float, float], np.ndarray]
    mu: float
    curvature_bound: float
    feature_sizes: tuple[float, ...]


@dataclass(frozen=True)
class Model3D:
    title: str
    hamiltonian: Callable[[float, float, float], np.ndarray]
    mu: float
    curvature_bound: float
    feature_sizes: tuple[float, ...]


@dataclass(frozen=True)
class SurfaceSnapshot:
    feature_size: float
    samples: np.ndarray
    points: np.ndarray
    cells: np.ndarray
    bands: np.ndarray


class RecordingHamiltonian2D:
    """Record real public-API callback locations in evaluation order."""

    def __init__(self, hamiltonian: Callable[[float, float], np.ndarray]) -> None:
        self.hamiltonian = hamiltonian
        self.points: list[tuple[float, float]] = []

    def __call__(self, kx: float, ky: float) -> np.ndarray:
        self.points.append((float(kx), float(ky)))
        return self.hamiltonian(kx, ky)


class RecordingHamiltonian3D:
    """Record real public-API callback locations in evaluation order."""

    def __init__(
        self,
        hamiltonian: Callable[[float, float, float], np.ndarray],
    ) -> None:
        self.hamiltonian = hamiltonian
        self.points: list[tuple[float, float, float]] = []

    def __call__(self, kx: float, ky: float, kz: float) -> np.ndarray:
        self.points.append((float(kx), float(ky), float(kz)))
        return self.hamiltonian(kx, ky, kz)


def reconstructed_density_wave(kx: float, ky: float) -> np.ndarray:
    x, y = 2.0 * np.pi * np.array([kx, ky])

    def dispersion(u: float, v: float) -> float:
        return (
            -1.75 * np.cos(u)
            - 1.15 * np.cos(v)
            + 0.88 * np.cos(u) * np.cos(v)
            - 0.32 * np.cos(2.0 * u)
            - 0.18 * np.cos(2.0 * v)
            + 0.20 * np.cos(2.0 * u - v)
            + 0.22 * np.sin(u + 2.0 * v)
            - 0.12 * np.sin(2.0 * u + v)
            + 0.13 * np.cos(3.0 * u - v + 0.6)
        )

    energy = dispersion(x, y)
    folded_energy = dispersion(x + np.pi, y + 0.68 * np.pi) + 0.18
    coupling_real = (
        0.22
        + 0.11 * np.cos(x - y)
        + 0.07 * np.cos(2.0 * y)
        + 0.06 * np.sin(2.0 * x + y)
    )
    coupling_imag = (
        0.10 * np.sin(x + 2.0 * y)
        + 0.07 * np.sin(2.0 * x - y)
    )
    coupling = coupling_real - 1.0j * coupling_imag
    return np.array(
        [
            [energy, coupling],
            [coupling.conjugate(), folded_energy],
        ],
        dtype=complex,
    )


def gapped_graphene(kx: float, ky: float) -> np.ndarray:
    x, y = 2.0 * np.pi * np.array([kx, ky])
    hopping = 1.0 + np.exp(1.0j * x) + np.exp(1.0j * y)
    return np.array(
        [
            [0.18, -hopping],
            [-hopping.conjugate(), -0.18],
        ],
        dtype=complex,
    )


def rashba_metal(kx: float, ky: float) -> np.ndarray:
    x, y = 2.0 * np.pi * np.array([kx, ky])
    scalar = -2.0 * (np.cos(x) + np.cos(y))
    dx = 0.85 * np.sin(y)
    dy = -0.85 * np.sin(x)
    dz = 0.22 + 0.18 * (np.cos(x) - np.cos(y))
    return np.array(
        [
            [scalar + dz, dx - 1.0j * dy],
            [dx + 1.0j * dy, scalar - dz],
        ],
        dtype=complex,
    )


def kagome_metal(kx: float, ky: float) -> np.ndarray:
    a = np.cos(np.pi * kx)
    b = np.cos(np.pi * ky)
    c = np.cos(np.pi * (kx - ky))
    return -2.0 * np.array(
        [
            [0.0, a, b],
            [a, 0.0, c],
            [b, c, 0.0],
        ],
        dtype=complex,
    )


def noble_metal_fcc(kx: float, ky: float, kz: float) -> np.ndarray:
    x, y, z = 2.0 * np.pi * (np.array([kx, ky, kz]) - 0.5)
    cx, cy, cz = np.cos(x), np.cos(y), np.cos(z)
    energy = -(cx * cy + cy * cz + cz * cx) - 0.55 * (cx + cy + cz)
    return np.array([[energy]], dtype=complex)


RECONSTRUCTED_METAL = Model2D(
    title="Chiral density-wave metal",
    hamiltonian=reconstructed_density_wave,
    mu=-0.40,
    curvature_bound=12.31 * (2.0 * np.pi) ** 2,
    feature_sizes=(0.30, 0.20, 0.13, 0.08, 0.05, 0.03),
)

RASHBA = Model2D(
    title="Spin-split Rashba metal",
    hamiltonian=rashba_metal,
    mu=-2.25,
    curvature_bound=6.06 * (2.0 * np.pi) ** 2,
    feature_sizes=(0.30, 0.20, 0.13, 0.08, 0.05, 0.03),
)

KAGOME = Model2D(
    title="Kagome metal",
    hamiltonian=kagome_metal,
    mu=0.15,
    curvature_bound=8.0 * np.pi**2,
    feature_sizes=(0.30, 0.20, 0.13, 0.08, 0.05, 0.03),
)

NOBLE_METAL = Model3D(
    title="Noble-metal-inspired Fermi surface",
    hamiltonian=noble_metal_fcc,
    mu=0.24,
    # Fourier Hessian bound: 6 from pair terms and 3 * 0.55 from linear terms.
    curvature_bound=8.0 * (2.0 * np.pi) ** 2,
    feature_sizes=(0.40, 0.28, 0.20, 0.14, 0.10),
)


def capture_2d(model: Model2D) -> list[SurfaceSnapshot]:
    recording = RecordingHamiltonian2D(model.hamiltonian)
    mesh = SpectralMesh(recording, root_level=0)
    recording.points.clear()
    snapshots = []

    for feature_size in model.feature_sizes:
        surface = mesh.fermi_surface(
            mu=model.mu,
            min_feature_size=feature_size,
            curvature_bound=model.curvature_bound,
        )
        if not surface.completed:
            raise RuntimeError(f"{model.title} did not complete")
        if len(recording.points) != mesh.cached_vertices:
            raise RuntimeError("recorded samples no longer match cached vertices")
        snapshots.append(
            SurfaceSnapshot(
                feature_size=feature_size,
                samples=np.asarray(recording.points, dtype=float).copy(),
                points=surface.points.copy(),
                cells=surface.cells.copy(),
                bands=surface.cell_bands.copy(),
            )
        )
    return snapshots


def capture_3d(model: Model3D) -> list[SurfaceSnapshot]:
    recording = RecordingHamiltonian3D(model.hamiltonian)
    mesh = SpectralMesh(recording, root_level=0)
    recording.points.clear()
    snapshots = []

    for feature_size in model.feature_sizes:
        surface = mesh.fermi_surface(
            mu=model.mu,
            min_feature_size=feature_size,
            curvature_bound=model.curvature_bound,
        )
        if not surface.completed:
            raise RuntimeError(f"{model.title} did not complete")
        if len(recording.points) != mesh.cached_vertices:
            raise RuntimeError("recorded samples no longer match cached vertices")
        snapshots.append(
            SurfaceSnapshot(
                feature_size=feature_size,
                samples=np.asarray(recording.points, dtype=float).copy(),
                points=surface.points.copy(),
                cells=surface.cells.copy(),
                bands=surface.cell_bands.copy(),
            )
        )
    return snapshots


def figure_image(figure) -> Image.Image:
    figure.canvas.draw()
    rgba = np.asarray(figure.canvas.buffer_rgba())
    return Image.fromarray(rgba).convert("RGB")


def save_gif(frames: list[Image.Image], destination: Path) -> None:
    palette = frames[-1].quantize(
        colors=128,
        method=Image.Quantize.MEDIANCUT,
    )
    converted = [
        frame.quantize(palette=palette, dither=Image.Dither.NONE)
        for frame in frames
    ]
    converted[0].save(
        destination,
        save_all=True,
        append_images=converted[1:],
        duration=150,
        optimize=True,
        disposal=2,
        loop=0,
    )


def completed_pass_states(
    snapshots: list[SurfaceSnapshot],
    *,
    frames_per_stage: int,
    final_hold: int,
):
    for stage in range(len(snapshots)):
        for frame in range(frames_per_stage):
            denominator = max(1, frames_per_stage - 1)
            new_highlight = 1.0 - frame / denominator
            yield stage, new_highlight
    for _ in range(final_hold):
        yield len(snapshots) - 1, 0.0


def style_2d_axis(axis) -> None:
    axis.set_facecolor(BACKGROUND)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_aspect("equal")
    axis.set_xticks((0.0, 0.5, 1.0))
    axis.set_yticks((0.0, 0.5, 1.0))
    axis.set_xlabel(r"$k_1$", color=MUTED, labelpad=5)
    axis.set_ylabel(r"$k_2$", color=MUTED, labelpad=5)
    axis.tick_params(colors=MUTED, labelsize=8, length=0, pad=6)
    for spine in axis.spines.values():
        spine.set_color(GRID)
        spine.set_linewidth(0.65)


def write_graphene_quickstart() -> None:
    surface = SpectralMesh(gapped_graphene).fermi_surface(
        mu=0.90,
        min_feature_size=0.025,
        curvature_bound=(2.0 * np.pi) ** 2,
    )

    figure, axis = plt.subplots(
        figsize=(7.2, 5.4),
        dpi=120,
        facecolor=BACKGROUND,
        constrained_layout=True,
    )
    style_2d_axis(axis)
    axis.add_collection(
        LineCollection(
            surface.points[surface.cells],
            colors=AMBER,
            linewidths=1.9,
            capstyle="round",
        )
    )
    axis.set_title(
        "Gapped graphene",
        color=FOREGROUND,
        fontsize=14,
        loc="left",
        pad=12,
    )
    axis.text(
        1.0,
        1.02,
        r"$\mu = 0.90$",
        transform=axis.transAxes,
        color=MUTED,
        fontsize=9,
        ha="right",
        va="bottom",
    )
    figure.savefig(
        ASSETS / "showcase_graphene_2d.png",
        facecolor=figure.get_facecolor(),
        dpi=120,
    )
    plt.close(figure)


def add_sampling_grid(axis, samples: np.ndarray) -> None:
    if len(samples) < 3:
        return
    triangulation = mtri.Triangulation(samples[:, 0], samples[:, 1])
    segments = samples[triangulation.edges]
    axis.add_collection(
        LineCollection(
            segments,
            colors=GRID,
            linewidths=0.28,
            alpha=0.30,
            zorder=1,
        )
    )


def add_samples(
    axis,
    samples: np.ndarray,
    *,
    previous_count: int,
    new_highlight: float,
) -> None:
    new = samples[min(previous_count, len(samples)) :]
    if len(samples):
        axis.scatter(
            samples[:, 0],
            samples[:, 1],
            s=5,
            color=MUTED,
            alpha=0.50,
            edgecolors="none",
            zorder=3,
        )
    if len(new) and new_highlight > 0.0:
        axis.scatter(
            new[:, 0],
            new[:, 1],
            s=8,
            color=FOREGROUND,
            alpha=0.80 * new_highlight,
            edgecolors="none",
            linewidths=0.0,
            zorder=4,
        )


def surface_colors(snapshot: SurfaceSnapshot) -> np.ndarray:
    return np.asarray(
        [BAND_COLORS[int(band) % len(BAND_COLORS)] for band in snapshot.bands]
    )


def add_surface_2d(
    axis,
    snapshot: SurfaceSnapshot,
    *,
    alpha: float,
    linewidth: float,
) -> None:
    if not len(snapshot.cells) or alpha <= 0.0:
        return
    axis.add_collection(
        LineCollection(
            snapshot.points[snapshot.cells],
            colors=surface_colors(snapshot),
            linewidths=linewidth,
            alpha=alpha,
            zorder=5,
        )
    )


def add_2d_state(
    axis,
    snapshots: list[SurfaceSnapshot],
    *,
    stage: int,
    new_highlight: float,
) -> None:
    snapshot = snapshots[stage]
    previous_count = 0 if stage == 0 else len(snapshots[stage - 1].samples)
    add_sampling_grid(axis, snapshot.samples)
    add_samples(
        axis,
        snapshot.samples,
        previous_count=previous_count,
        new_highlight=new_highlight,
    )
    add_surface_2d(
        axis,
        snapshot,
        alpha=0.98,
        linewidth=1.8,
    )


def write_adaptive_2d_gif(
    model: Model2D,
    snapshots: list[SurfaceSnapshot],
) -> None:
    frames = []
    for stage, new_highlight in completed_pass_states(
        snapshots,
        frames_per_stage=5,
        final_hold=8,
    ):
        snapshot = snapshots[stage]
        visible = len(snapshot.samples)
        figure = plt.figure(figsize=(9.0, 6.75), dpi=100, facecolor=BACKGROUND)
        axis = figure.add_axes((0.18, 0.14, 0.64, 0.72))
        style_2d_axis(axis)
        add_2d_state(
            axis,
            snapshots,
            stage=stage,
            new_highlight=new_highlight,
        )

        figure.text(
            0.055,
            0.945,
            model.title,
            color=FOREGROUND,
            fontsize=17,
            ha="left",
            va="top",
        )
        figure.text(
            0.945,
            0.942,
            f"{visible:,} diagonalizations",
            color=MUTED,
            fontsize=10,
            ha="right",
            va="top",
        )
        figure.text(
            0.5,
            0.055,
            f"adaptive pass {stage + 1} / {len(snapshots)}",
            color=MUTED,
            fontsize=9,
            ha="center",
            va="center",
        )
        frames.append(figure_image(figure))
        plt.close(figure)

    save_gif(frames, ASSETS / "showcase_adaptive_2d.gif")


def write_multiband_2d_gif(
    models: tuple[Model2D, Model2D],
    model_snapshots: tuple[list[SurfaceSnapshot], list[SurfaceSnapshot]],
) -> None:
    frames = []
    states = completed_pass_states(
        model_snapshots[0],
        frames_per_stage=5,
        final_hold=8,
    )
    for stage, new_highlight in states:
        figure = plt.figure(figsize=(11.0, 5.8), dpi=100, facecolor=BACKGROUND)
        axes = (
            figure.add_axes((0.055, 0.17, 0.40, 0.70)),
            figure.add_axes((0.545, 0.17, 0.40, 0.70)),
        )

        for axis, model, snapshots in zip(
            axes,
            models,
            model_snapshots,
            strict=True,
        ):
            visible = len(snapshots[stage].samples)
            style_2d_axis(axis)
            add_2d_state(
                axis,
                snapshots,
                stage=stage,
                new_highlight=new_highlight,
            )
            axis.set_title(
                model.title,
                color=FOREGROUND,
                fontsize=13,
                loc="left",
                pad=14,
            )
            axis.text(
                1.0,
                1.025,
                f"{visible:,} diagonalizations",
                transform=axis.transAxes,
                color=MUTED,
                fontsize=8,
                ha="right",
                va="bottom",
            )

        figure.text(
            0.5,
            0.065,
            f"adaptive pass {stage + 1} / {len(model_snapshots[0])}",
            color=MUTED,
            fontsize=9,
            ha="center",
            va="center",
        )
        frames.append(figure_image(figure))
        plt.close(figure)

    save_gif(frames, ASSETS / "showcase_multiband_2d.gif")


def style_3d_axis(axis) -> None:
    axis.set_facecolor(BACKGROUND)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_zlim(0.0, 1.0)
    axis.set_box_aspect((1.0, 1.0, 1.0), zoom=1.05)
    axis.set_axis_off()


def add_unit_cube(axis) -> None:
    corners = np.asarray(
        [
            (x, y, z)
            for z in (0.0, 1.0)
            for y in (0.0, 1.0)
            for x in (0.0, 1.0)
        ]
    )
    edges = (
        (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
        (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
    )
    for start, end in edges:
        segment = corners[[start, end]]
        axis.plot(
            segment[:, 0],
            segment[:, 1],
            segment[:, 2],
            color=GRID,
            linewidth=0.40,
            alpha=0.28,
        )


def smooth_surface_colors(snapshot: SurfaceSnapshot) -> np.ndarray:
    triangles = snapshot.points[snapshot.cells]
    face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    lengths = np.linalg.norm(face_normals, axis=1)
    face_normals /= np.maximum(lengths[:, None], np.finfo(float).eps)

    vertex_normals = np.zeros_like(snapshot.points)
    for corner in range(3):
        np.add.at(vertex_normals, snapshot.cells[:, corner], face_normals)
    lengths = np.linalg.norm(vertex_normals, axis=1)
    vertex_normals /= np.maximum(lengths[:, None], np.finfo(float).eps)

    averaged = vertex_normals[snapshot.cells].mean(axis=1)
    light = np.asarray((0.35, -0.45, 0.82))
    light /= np.linalg.norm(light)
    brightness = 0.68 + 0.32 * np.abs(averaged @ light)
    base = np.asarray(
        [
            to_rgb(BAND_COLORS[int(band) % len(BAND_COLORS)])
            for band in snapshot.bands
        ]
    )
    return base * brightness[:, None]


def add_surface_3d(
    axis,
    snapshot: SurfaceSnapshot,
    *,
    alpha: float,
) -> None:
    if not len(snapshot.cells) or alpha <= 0.0:
        return
    triangles = snapshot.points[snapshot.cells]
    colors = smooth_surface_colors(snapshot)
    axis.add_collection3d(
        Poly3DCollection(
            triangles,
            facecolors=colors,
            edgecolors=colors,
            linewidth=0.0,
            alpha=alpha,
            shade=False,
        )
    )


def write_noble_metal_3d_gif(
    model: Model3D,
    snapshots: list[SurfaceSnapshot],
) -> None:
    states: list[tuple[int, float, float, bool]] = []
    for stage in range(len(snapshots)):
        states.extend([(stage, -55.0, 24.0, False)] * 4)

    orbit_frames = 37
    for orbit_frame in range(orbit_frames):
        phase = orbit_frame / max(1, orbit_frames - 1)
        states.append(
            (
                len(snapshots) - 1,
                -55.0 + 360.0 * phase,
                24.0 + 3.0 * np.sin(2.0 * np.pi * phase),
                True,
            )
        )
    states.extend([(len(snapshots) - 1, 305.0, 24.0, True)] * 6)

    frames = []
    for stage, azimuth, elevation, rotating in states:
        snapshot = snapshots[stage]
        visible = len(snapshot.samples)

        figure = plt.figure(figsize=(9.0, 6.75), dpi=100, facecolor=BACKGROUND)
        axis = figure.add_axes((0.02, 0.025, 0.96, 0.91), projection="3d")
        style_3d_axis(axis)
        add_unit_cube(axis)
        axis.view_init(elev=elevation, azim=azimuth)
        add_surface_3d(axis, snapshot, alpha=0.96)

        figure.text(
            0.055,
            0.945,
            model.title,
            color=FOREGROUND,
            fontsize=17,
            ha="left",
            va="top",
        )
        label = (
            "final mesh"
            if rotating
            else f"adaptive pass {stage + 1} / {len(snapshots)}"
        )
        figure.text(
            0.945,
            0.942,
            f"{label}  ·  {visible:,} diagonalizations",
            color=MUTED,
            fontsize=10,
            ha="right",
            va="top",
        )
        frames.append(figure_image(figure))
        plt.close(figure)

    save_gif(frames, ASSETS / "showcase_noble_metal_3d.gif")


def report(name: str, snapshots: list[SurfaceSnapshot]) -> None:
    final = snapshots[-1]
    print(
        f"{name}: {len(final.samples):,} eigensolves, "
        f"{len(final.cells):,} cells, "
        f"feature size {final.feature_size:g}"
    )


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)

    write_graphene_quickstart()
    reconstructed = capture_2d(RECONSTRUCTED_METAL)
    rashba = capture_2d(RASHBA)
    kagome = capture_2d(KAGOME)
    noble_metal = capture_3d(NOBLE_METAL)

    write_adaptive_2d_gif(RECONSTRUCTED_METAL, reconstructed)
    write_multiband_2d_gif((RASHBA, KAGOME), (rashba, kagome))
    write_noble_metal_3d_gif(NOBLE_METAL, noble_metal)

    report(RECONSTRUCTED_METAL.title, reconstructed)
    report(RASHBA.title, rashba)
    report(KAGOME.title, kagome)
    report(NOBLE_METAL.title, noble_metal)


if __name__ == "__main__":
    main()
