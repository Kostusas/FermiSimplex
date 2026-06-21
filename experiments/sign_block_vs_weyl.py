"""Temporary experiment: vertex-anchor Weyl vs sign-block certificates in 2D.

The model is an affine 2D Hermitian Hamiltonian

    A(x, y) = H(x, y) - mu I = A0 + x Hx + y Hy,  mu = 0.

It is built to expose the difference between a global Weyl radius and the
sign-block test:

* x-motion is mostly harmless: remote occupied bands move down and remote empty
  bands move up, with additional occupied-empty mixing.
* y-motion is harmful: the closest occupied/empty pair moves toward zero.

For the right triangle [(0, 0), (h, 0), (0, h)], the sign-block certificate from
the (0, 0) vertex is exact for the harmful y-motion and should pass until h ~= 1.
Weyl sees the large harmless x-motion through one global spectral norm and fails
at a much smaller h.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class Model:
    """Linear Hermitian model in two variables."""

    a0: Array
    hx: Array
    hy: Array
    gaps: Array
    safe_rate: float
    mix_rate: float
    bad_rate: float

    def a(self, k: Array) -> Array:
        return self.a0 + k[0] * self.hx + k[1] * self.hy


@dataclass(frozen=True)
class CertResult:
    passed: bool
    best_anchor: int | None
    best_value: float
    values: tuple[float, ...]


def random_orthogonal(n: int, seed: int) -> Array:
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(n, n)))
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs


def normalized_rectangular(shape: tuple[int, int], seed: int) -> Array:
    rng = np.random.default_rng(seed)
    m = rng.normal(size=shape)
    norm = np.linalg.svd(m, compute_uv=False)[0]
    return m / norm


def make_gaps(bands: int, max_gap: float) -> Array:
    if bands < 4 or bands % 2 != 0:
        raise ValueError("--bands must be an even integer >= 4")

    half = bands // 2
    emp_max = 0.92 * max_gap
    occ = np.empty(half)
    emp = np.empty(half)

    # Keep the near-Fermi gaps close to the previous 8-band toy model so the
    # dangerous mode remains comparable across sizes, then distribute the added
    # remote bands smoothly through the same bandwidth.
    occ[0] = 1.0
    emp[0] = 1.0
    if half >= 2:
        occ[1] = min(1.6, max_gap)
        emp[1] = min(1.4, emp_max)
    if half >= 3:
        occ[2:] = np.geomspace(min(3.0, max_gap), max_gap, half - 2)
        emp[2:] = np.geomspace(min(2.8, emp_max), emp_max, half - 2)

    return np.concatenate([-occ, emp])


def make_model(
    bands: int = 8,
    seed: int = 4,
    safe_rate: float = 2.5,
    mix_rate: float = 1.5,
    max_gap: float = 6.0,
) -> Model:
    """Construct an even-band real-symmetric Hamiltonian.

    The first half of anchor eigenvalues are occupied and the rest are empty.
    The closest pair has gap 1.0; remote bands have larger gaps and dominate the
    Weyl norm through harmless x-motion.
    """

    gaps = make_gaps(bands, max_gap)
    half = bands // 2
    signs = np.sign(gaps)
    abs_gaps = np.abs(gaps)

    # Harmless x drift: away from zero, mostly on remote bands. The nearest pair
    # deliberately barely moves, so the Weyl anchor gap stays small.
    ramp = np.linspace(0.0, 1.0, half)
    safe_weights = np.concatenate([ramp**1.4, ramp**1.3])
    if half >= 2:
        safe_weights[1] = 0.10
        safe_weights[half + 1] = 0.12
    if half >= 3:
        safe_weights[2] = 0.45
        safe_weights[half + 2] = 0.50
    hx_diag = safe_rate * safe_weights * abs_gaps * signs

    hx_eigenbasis = np.diag(hx_diag)
    cross = normalized_rectangular((half, bands - half), seed + 100)
    cross_scale = np.sqrt(abs_gaps[:half, None] * abs_gaps[None, half:])
    hx_eigenbasis[:half, half:] = mix_rate * cross_scale * cross
    hx_eigenbasis[half:, :half] = hx_eigenbasis[:half, half:].T

    # Harmful y drift: the closest occupied band moves toward zero. In the
    # sign-normalized negative block this is exactly R = I - y for that band, so
    # the true occupation count changes once the triangle reaches y > 1.
    bad_rate = 1.0
    bad_weights = np.zeros(bands)
    bad_weights[0] = 1.0
    if half >= 2:
        bad_weights[1] = 0.25
    hy_diag = -bad_rate * bad_weights * abs_gaps * signs
    hy_eigenbasis = np.diag(hy_diag)

    q = random_orthogonal(len(gaps), seed)
    a0 = q @ np.diag(gaps) @ q.T
    hx = q @ hx_eigenbasis @ q.T
    hy = q @ hy_eigenbasis @ q.T
    return Model(a0=a0, hx=hx, hy=hy, gaps=gaps, safe_rate=safe_rate, mix_rate=mix_rate, bad_rate=bad_rate)


def triangle_vertices(h: float) -> tuple[Array, Array, Array]:
    return (
        np.array([0.0, 0.0]),
        np.array([h, 0.0]),
        np.array([0.0, h]),
    )


def vertex_matrices(model: Model, h: float) -> tuple[Array, Array, Array]:
    return tuple(model.a(v) for v in triangle_vertices(h))


def inertia_count(a: Array, tol: float = 1e-10) -> tuple[int, int, int]:
    evals = np.linalg.eigvalsh(a)
    return (
        int(np.sum(evals < -tol)),
        int(np.sum(np.abs(evals) <= tol)),
        int(np.sum(evals > tol)),
    )


def weyl_certificate(mats: tuple[Array, ...], tol: float = 1e-12) -> CertResult:
    margins: list[float] = []
    for anchor, ap in enumerate(mats):
        evals = np.linalg.eigvalsh(ap)
        gap = float(np.min(np.abs(evals)))
        if gap <= tol:
            margins.append(np.inf)
            continue
        radius = max(float(np.linalg.norm(ai - ap, ord=2)) for ai in mats)
        margins.append(radius / gap)

    best_anchor = int(np.argmin(margins))
    best = margins[best_anchor]
    return CertResult(passed=best < 1.0 - tol, best_anchor=best_anchor, best_value=best, values=tuple(margins))


def sign_block_anchor_margin(mats: tuple[Array, ...], anchor: int, tol: float = 1e-12) -> float:
    ap = mats[anchor]
    evals, u = np.linalg.eigh(ap)
    pos = evals > tol
    neg = evals < -tol
    if not np.all(pos | neg):
        return -np.inf

    min_block_eval = np.inf
    if np.any(pos):
        up = u[:, pos]
        inv_sqrt = np.diag(1.0 / np.sqrt(evals[pos]))
        for ai in mats:
            r = inv_sqrt @ (up.conj().T @ ai @ up) @ inv_sqrt
            min_block_eval = min(min_block_eval, float(np.linalg.eigvalsh(r)[0]))

    if np.any(neg):
        un = u[:, neg]
        inv_sqrt = np.diag(1.0 / np.sqrt(-evals[neg]))
        for ai in mats:
            # The leading minus sign is the sign normalization; it makes
            # R^- = I at the anchor.
            r = -inv_sqrt @ (un.conj().T @ ai @ un) @ inv_sqrt
            min_block_eval = min(min_block_eval, float(np.linalg.eigvalsh(r)[0]))

    return min_block_eval


def sign_block_certificate(mats: tuple[Array, ...], tol: float = 1e-12) -> CertResult:
    anchor_margins: list[float] = []

    for anchor, _ in enumerate(mats):
        anchor_margins.append(sign_block_anchor_margin(mats, anchor, tol=tol))

    best_anchor = int(np.argmax(anchor_margins))
    best = anchor_margins[best_anchor]
    return CertResult(passed=best > tol, best_anchor=best_anchor, best_value=best, values=tuple(anchor_margins))


def eta_bad_from_margin(sb_margin: float) -> float:
    return max(0.0, 1.0 - sb_margin)


def sign_block_norm_metric(mats: tuple[Array, ...], anchor: int) -> float:
    ap = mats[anchor]
    evals, u = np.linalg.eigh(ap)
    tol = 1e-12
    pos = evals > tol
    neg = evals < -tol
    worst = 0.0

    if np.any(pos):
        up = u[:, pos]
        inv_sqrt = np.diag(1.0 / np.sqrt(evals[pos]))
        eye = np.eye(int(np.sum(pos)))
        for ai in mats:
            r = inv_sqrt @ (up.conj().T @ ai @ up) @ inv_sqrt
            worst = max(worst, float(np.linalg.norm(r - eye, ord=2)))

    if np.any(neg):
        un = u[:, neg]
        inv_sqrt = np.diag(1.0 / np.sqrt(-evals[neg]))
        eye = np.eye(int(np.sum(neg)))
        for ai in mats:
            r = -inv_sqrt @ (un.conj().T @ ai @ un) @ inv_sqrt
            worst = max(worst, float(np.linalg.norm(r - eye, ord=2)))

    return worst


def sampled_safety(model: Model, h: float, samples_per_edge: int) -> tuple[bool, float]:
    reference = inertia_count(model.a(np.array([0.0, 0.0])))
    min_gap = np.inf
    for i in range(samples_per_edge):
        x = h * i / (samples_per_edge - 1)
        for j in range(samples_per_edge - i):
            y = h * j / (samples_per_edge - 1)
            evals = np.linalg.eigvalsh(model.a(np.array([x, y])))
            min_gap = min(min_gap, float(np.min(np.abs(evals))))
            if inertia_count(model.a(np.array([x, y]))) != reference:
                return False, min_gap
    return min_gap > 1e-8, min_gap


def find_threshold(predicate: Callable[[float], bool], hi: float = 1.0) -> float:
    lo = 0.0
    while predicate(hi):
        lo = hi
        hi *= 2.0
        if hi > 1024.0:
            raise RuntimeError("predicate did not fail before h=1024")

    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if predicate(mid):
            lo = mid
        else:
            hi = mid
    return lo


def print_summary(model: Model, h_weyl: float, h_sb: float) -> None:
    print("Model")
    print(f"  bands: {len(model.gaps)} ({np.sum(model.gaps < 0)} occupied, {np.sum(model.gaps > 0)} empty)")
    print(f"  anchor gap range: [{np.min(np.abs(model.gaps)):.3g}, {np.max(np.abs(model.gaps)):.3g}]")
    print(f"  first occupied gaps: {np.array2string(model.gaps[:min(5, len(model.gaps) // 2)], precision=3)}")
    print(f"  first empty gaps: {np.array2string(model.gaps[len(model.gaps) // 2:len(model.gaps) // 2 + min(5, len(model.gaps) // 2)], precision=3)}")
    print(f"  safe x drift rate: {model.safe_rate:g}")
    print(f"  occupied-empty x mixing rate: {model.mix_rate:g}")
    print(f"  harmful y drift rate: {model.bad_rate:g}")
    print()
    print("Vertex-anchor thresholds on triangle [(0,0), (h,0), (0,h)]")
    print(f"  Weyl max h:      {h_weyl:.8f}")
    print(f"  sign-block max h:{h_sb:.8f}")
    print(f"  side-length ratio h_SB / h_Weyl: {h_sb / h_weyl:.2f}x")
    print(f"  2D triangle-count ratio estimate: {(h_sb / h_weyl) ** 2:.1f}x fewer triangles")
    print()


def print_table(model: Model, hs: list[float], samples_per_edge: int) -> None:
    print("Diagnostics")
    print("  h           Weyl?  Weyl margin  SB?    SB best  SB@v0    eta_bad@v0  eta_norm@v0  sampled safe  sample gap")
    for h in hs:
        mats = vertex_matrices(model, h)
        wy = weyl_certificate(mats)
        sb = sign_block_certificate(mats)
        sb_v0 = sign_block_anchor_margin(mats, 0)
        eta_norm_v0 = sign_block_norm_metric(mats, 0)
        sampled_safe, sample_gap = sampled_safety(model, h, samples_per_edge=samples_per_edge)
        print(
            f"  {h:10.6f}  "
            f"{str(wy.passed):5s}  {wy.best_value:11.6f}  "
            f"{str(sb.passed):5s}  {sb.best_value:7.4f}  "
            f"{sb_v0:7.4f}  "
            f"{eta_bad_from_margin(sb_v0):10.4f}  "
            f"{eta_norm_v0:10.4f}  "
            f"{str(sampled_safe):12s}  "
            f"{sample_gap:10.4e}"
        )


def run_case(bands: int, args: argparse.Namespace) -> tuple[int, float, float, float]:
    model = make_model(
        bands=bands,
        seed=args.seed,
        safe_rate=args.safe_rate,
        mix_rate=args.mix_rate,
        max_gap=args.max_gap,
    )

    started = time.perf_counter()
    h_weyl = find_threshold(lambda h: weyl_certificate(vertex_matrices(model, h)).passed)
    h_sb = find_threshold(lambda h: sign_block_certificate(vertex_matrices(model, h)).passed)
    elapsed = time.perf_counter() - started

    print_summary(model, h_weyl, h_sb)
    print(f"  threshold search wall time: {elapsed:.3f}s")
    print()
    if args.table:
        print_table(
            model,
            [
                0.5 * h_weyl,
                0.99 * h_weyl,
                1.01 * h_weyl,
                0.5 * (h_weyl + h_sb),
                0.99 * h_sb,
                1.01 * h_sb,
            ],
            samples_per_edge=args.samples_per_edge,
        )
        print()

    return bands, h_weyl, h_sb, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bands", type=int, nargs="+", default=[8], help="even matrix sizes to run")
    parser.add_argument("--max-gap", type=float, default=6.0, help="largest absolute anchor eigenvalue")
    parser.add_argument("--safe-rate", type=float, default=2.5, help="harmless x drift strength")
    parser.add_argument("--mix-rate", type=float, default=1.5, help="occupied-empty x mixing strength")
    parser.add_argument("--seed", type=int, default=4, help="deterministic random seed")
    parser.add_argument("--samples-per-edge", type=int, default=81, help="sampled inertia grid resolution")
    parser.add_argument("--table", action=argparse.BooleanOptionalAction, default=True, help="print diagnostic rows")
    args = parser.parse_args()

    results = [run_case(bands, args) for bands in args.bands]
    if len(results) > 1:
        print("Summary")
        print("  bands  Weyl h      sign-block h  h ratio   area ratio  search time")
        for bands, h_weyl, h_sb, elapsed in results:
            ratio = h_sb / h_weyl
            print(f"  {bands:5d}  {h_weyl:10.8f}  {h_sb:12.8f}  {ratio:7.2f}x  {ratio**2:9.1f}x  {elapsed:9.3f}s")


if __name__ == "__main__":
    main()
