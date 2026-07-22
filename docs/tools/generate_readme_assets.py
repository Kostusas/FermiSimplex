from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(ROOT / "docs" / "generated" / "matplotlib"),
)

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

from examples.fermi_surface import CURVATURE_BOUND, hamiltonian
from examples.quick_start import (
    CURVATURE_BOUND as SCHWARZ_P_CURVATURE_BOUND,
    hamiltonian as schwarz_p_hamiltonian,
)
from fermisimplex import SpectralMesh


ASSETS = ROOT / "docs" / "assets"
BACKGROUND = "#07111f"
FOREGROUND = "#e6edf7"
MUTED = "#8fa3bd"
AMBER = "#ffb000"
CYAN = "#4dd5ff"
GRID = "#26364b"


@dataclass(frozen=True)
class SurfaceSnapshot:
    points: np.ndarray
    cells: np.ndarray
    feature_size: float
    diagonalizations: int


def refinement_snapshots() -> list[SurfaceSnapshot]:
    mesh = SpectralMesh(schwarz_p_hamiltonian)
    snapshots = []
    for feature_size in (0.40, 0.28, 0.20, 0.14, 0.10, 0.07):
        surface = mesh.fermi_surface(
            mu=0.17,
            min_feature_size=feature_size,
            curvature_bound=SCHWARZ_P_CURVATURE_BOUND,
        )
        snapshots.append(
            SurfaceSnapshot(
                points=surface.points.copy(),
                cells=surface.cells.copy(),
                feature_size=feature_size,
                diagonalizations=mesh.cached_vertices,
            )
        )
    return snapshots


def style_3d_axis(axis) -> None:
    axis.set_facecolor(BACKGROUND)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_zlim(0.0, 1.0)
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.set_axis_off()


def add_surface(axis, snapshot: SurfaceSnapshot, *, edges: bool) -> None:
    triangles = snapshot.points[snapshot.cells]
    collection = Poly3DCollection(
        triangles,
        facecolors=AMBER,
        edgecolors=GRID if edges else AMBER,
        linewidth=0.12 if edges else 0.0,
        alpha=0.96,
        shade=True,
    )
    axis.add_collection3d(collection)


def figure_image(figure) -> Image.Image:
    figure.canvas.draw()
    rgba = np.asarray(figure.canvas.buffer_rgba())
    return Image.fromarray(rgba).convert("RGB")


def write_refinement_gif(snapshots: list[SurfaceSnapshot]) -> None:
    frames: list[Image.Image] = []
    states = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5, 5]
    angles = np.linspace(-58.0, -22.0, len(states))

    for state, angle in zip(states, angles, strict=True):
        snapshot = snapshots[state]
        figure = plt.figure(figsize=(9.6, 5.4), dpi=90, facecolor=BACKGROUND)
        axis = figure.add_axes((0.22, 0.04, 0.56, 0.88), projection="3d")
        style_3d_axis(axis)
        axis.view_init(elev=24.0, azim=float(angle))
        add_surface(axis, snapshot, edges=state < len(snapshots) - 1)

        figure.text(
            0.5,
            0.94,
            "Adaptive Fermi-surface extraction",
            color=FOREGROUND,
            fontsize=18,
            ha="center",
        )
        figure.text(
            0.5,
            0.06,
            f"feature size {snapshot.feature_size:.2f}   •   "
            f"{snapshot.diagonalizations:,} cached diagonalizations",
            color=MUTED,
            fontsize=10,
            ha="center",
        )
        frames.append(figure_image(figure))
        plt.close(figure)

    palette_frames = [
        frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=128)
        for frame in frames
    ]
    palette_frames[0].save(
        ASSETS / "fermi_surface_refinement.gif",
        save_all=True,
        append_images=palette_frames[1:],
        duration=260,
        loop=0,
        optimize=True,
        disposal=2,
    )


def write_surface_gallery(final_snapshot: SurfaceSnapshot) -> None:
    surface_2d = SpectralMesh(hamiltonian).fermi_surface(
        mu=0.0,
        min_feature_size=0.01,
        curvature_bound=CURVATURE_BOUND,
    )

    figure = plt.figure(figsize=(12.0, 5.4), dpi=140, facecolor=BACKGROUND)
    contour_axis = figure.add_axes((0.06, 0.17, 0.38, 0.68))
    contour_axis.set_facecolor(BACKGROUND)
    colors = np.where(surface_2d.cell_bands == 0, AMBER, CYAN)
    contour_axis.add_collection(
        LineCollection(
            surface_2d.points[surface_2d.cells],
            colors=colors,
            linewidths=1.15,
            alpha=0.96,
        )
    )
    contour_axis.set_xlim(0.0, 1.0)
    contour_axis.set_ylim(0.0, 1.0)
    contour_axis.set_aspect("equal")
    contour_axis.set_xlabel(r"$k_x$", color=MUTED)
    contour_axis.set_ylabel(r"$k_y$", color=MUTED)
    contour_axis.tick_params(colors=MUTED, labelsize=8)
    for spine in contour_axis.spines.values():
        spine.set_color(GRID)
    contour_axis.grid(color=GRID, linewidth=0.45, alpha=0.55)

    surface_axis = figure.add_axes((0.49, 0.03, 0.46, 0.91), projection="3d")
    style_3d_axis(surface_axis)
    surface_axis.view_init(elev=23.0, azim=-42.0)
    add_surface(surface_axis, final_snapshot, edges=False)

    figure.text(0.25, 0.91, "Two-band contours", color=FOREGROUND, ha="center", fontsize=14)
    figure.text(
        0.73,
        0.91,
        "Three-dimensional Fermi surface",
        color=FOREGROUND,
        ha="center",
        fontsize=14,
    )
    figure.text(
        0.25,
        0.035,
        f"{surface_2d.cells.shape[0]:,} segments   •   "
        f"{surface_2d.stats.evaluations:,} diagonalizations",
        color=MUTED,
        ha="center",
        fontsize=9,
    )
    figure.text(
        0.73,
        0.035,
        f"{final_snapshot.cells.shape[0]:,} triangles   •   "
        f"{final_snapshot.diagonalizations:,} diagonalizations",
        color=MUTED,
        ha="center",
        fontsize=9,
    )
    figure.savefig(
        ASSETS / "fermi_surface_gallery.png",
        facecolor=figure.get_facecolor(),
        dpi=140,
    )
    plt.close(figure)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    snapshots = refinement_snapshots()
    write_refinement_gif(snapshots)
    write_surface_gallery(snapshots[-1])


if __name__ == "__main__":
    main()
