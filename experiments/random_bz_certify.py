"""Temporary whole-BZ scan for Weyl vs sign-block certificates.

This measures the vertex-anchor certificates on a unit-square BZ, triangulated
as an n x n grid of squares split into two triangles each.  The Hamiltonian is a
random real-symmetric Fourier model:

    A(k) = H(k) - mu I,  mu = 0,  k in [0, 1)^2.

The certificate comparison is applied to the affine/linear interpolation over
each triangle using the three vertex matrices.  That matches the linear
sign-block certificate directly.  It does not add a nonlinear Fourier
interpolation remainder bound; this script is intended as a rough scaling
diagnostic for the core idea.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from sign_block_vs_weyl import sign_block_certificate, weyl_certificate


Array = np.ndarray


@dataclass(frozen=True)
class FourierTerm:
    gx: int
    gy: int
    cos: Array
    sin: Array


@dataclass(frozen=True)
class FourierModel:
    base: Array
    terms: tuple[FourierTerm, ...]

    def a(self, k: tuple[float, float]) -> Array:
        x, y = k
        out = self.base.copy()
        for term in self.terms:
            phase = 2.0 * np.pi * (term.gx * x + term.gy * y)
            out += np.cos(phase) * term.cos + np.sin(phase) * term.sin
        return out


@dataclass(frozen=True)
class RotationTerm:
    i: int
    j: int
    gx: int
    gy: int
    phase: float
    amplitude: float


@dataclass(frozen=True)
class BandVariation:
    gx: Array
    gy: Array
    phase: Array
    weight: Array


@dataclass(frozen=True)
class RotationModel:
    gaps: Array
    rotations: tuple[RotationTerm, ...]
    variation: BandVariation
    variation_scale: float

    def a(self, k: tuple[float, float]) -> Array:
        x, y = k
        phases = 2.0 * np.pi * (self.variation.gx * x + self.variation.gy * y) + self.variation.phase
        factors = 1.0 + self.variation_scale * self.variation.weight * np.sin(phases)
        d = self.gaps * factors

        q = np.eye(len(self.gaps))
        for term in self.rotations:
            phase = 2.0 * np.pi * (term.gx * x + term.gy * y) + term.phase
            angle = term.amplitude * np.sin(phase)
            c = np.cos(angle)
            s = np.sin(angle)
            qi = q[:, term.i].copy()
            qj = q[:, term.j].copy()
            q[:, term.i] = c * qi + s * qj
            q[:, term.j] = -s * qi + c * qj

        return (q * d) @ q.T


@dataclass(frozen=True)
class HoppingTerm:
    rx: int
    ry: int
    matrix: Array


@dataclass(frozen=True)
class TightBindingModel:
    onsite: Array
    hoppings: tuple[HoppingTerm, ...]
    mu: float = 0.0

    def a(self, k: tuple[float, float]) -> Array:
        x, y = k
        out = self.onsite.copy()
        for hopping in self.hoppings:
            phase = np.exp(2j * np.pi * (hopping.rx * x + hopping.ry * y))
            out += phase * hopping.matrix + np.conj(phase) * hopping.matrix.conj().T
        return out - self.mu * np.eye(out.shape[0], dtype=out.dtype)


@dataclass(frozen=True)
class MeshStats:
    n: int
    triangles: int
    weyl_passed: int
    sb_passed: int
    both_passed: int
    min_weyl_slack: float
    min_sb_margin: float

    @property
    def weyl_all(self) -> bool:
        return self.weyl_passed == self.triangles

    @property
    def sb_all(self) -> bool:
        return self.sb_passed == self.triangles


@dataclass(frozen=True)
class ResolutionStats:
    n: int
    triangles: int
    cut_h: float
    weyl_safe: int
    weyl_cut: int
    weyl_unresolved: int
    sb_safe: int
    sb_cut: int
    sb_unresolved: int
    raw_cut: int
    min_weyl_slack: float
    min_sb_margin: float

    @property
    def weyl_resolved_all(self) -> bool:
        return self.weyl_unresolved == 0

    @property
    def sb_resolved_all(self) -> bool:
        return self.sb_unresolved == 0


def random_symmetric(n: int, rng: np.random.Generator) -> Array:
    a = rng.normal(size=(n, n))
    return 0.5 * (a + a.T)


def random_complex(rows: int, cols: int | np.random.Generator, rng: np.random.Generator | None = None) -> Array:
    if rng is None:
        rng = cols
        cols = rows
    assert isinstance(rng, np.random.Generator)
    assert isinstance(cols, int)
    return rng.normal(size=(rows, cols)) + 1j * rng.normal(size=(rows, cols))


def random_hermitian(n: int, rng: np.random.Generator) -> Array:
    a = random_complex(n, rng)
    return 0.5 * (a + a.conj().T)


def normalize_spectral(a: Array) -> Array:
    norm = np.linalg.norm(a, ord=2)
    if norm == 0.0:
        return a
    return a / norm


def make_base_gaps(bands: int, max_gap: float) -> Array:
    if bands < 4 or bands % 2 != 0:
        raise ValueError("--bands must be an even integer >= 4")
    half = bands // 2
    occupied = np.geomspace(1.0, max_gap, half)
    empty = np.geomspace(1.0, 0.9 * max_gap, half)
    return np.concatenate([-occupied, empty])


def random_orthogonal(n: int, rng: np.random.Generator) -> Array:
    q, r = np.linalg.qr(rng.normal(size=(n, n)))
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs


def random_unitary(n: int, rng: np.random.Generator) -> Array:
    q, r = np.linalg.qr(random_complex(n, rng))
    phases = np.diag(r)
    phases = phases / np.where(np.abs(phases) == 0.0, 1.0, np.abs(phases))
    return q * phases.conj()


def block_scaled_random(
    bands: int,
    rng: np.random.Generator,
    intra_scale: float,
    cross_scale: float,
) -> Array:
    half = bands // 2
    a = np.zeros((bands, bands))
    a[:half, :half] = intra_scale * normalize_spectral(random_symmetric(half, rng))
    a[half:, half:] = intra_scale * normalize_spectral(random_symmetric(half, rng))

    cross = rng.normal(size=(half, half))
    cross = cross / np.linalg.svd(cross, compute_uv=False)[0]
    a[:half, half:] = cross_scale * cross
    a[half:, :half] = a[:half, half:].T
    return a


def block_scaled_complex_hopping(
    bands: int,
    rng: np.random.Generator,
    intra_scale: float,
    cross_scale: float,
) -> Array:
    half = bands // 2
    a = np.zeros((bands, bands), dtype=np.complex128)

    a[:half, :half] = intra_scale * normalize_spectral(random_complex(half, rng))
    a[half:, half:] = intra_scale * normalize_spectral(random_complex(half, rng))

    cross = random_complex(half, half, rng)
    cross = cross / np.linalg.svd(cross, compute_uv=False)[0]
    a[:half, half:] = cross_scale * cross
    a[half:, :half] = cross_scale * random_complex(half, half, rng) / np.sqrt(half)
    return a


def make_model(
    bands: int,
    seed: int,
    max_gap: float,
    perturb: float,
    cross_fraction: float,
    modes: Iterable[tuple[int, int]],
) -> FourierModel:
    rng = np.random.default_rng(seed)
    gaps = make_base_gaps(bands, max_gap)
    q = random_orthogonal(bands, rng)
    base = q @ np.diag(gaps) @ q.T

    terms = []
    for gx, gy in modes:
        freq = np.hypot(gx, gy)
        # Higher modes get smaller coefficients, but still enough structure to
        # create nontrivial local variation.
        amp = perturb / freq
        intra = amp
        cross = amp * cross_fraction
        c = block_scaled_random(bands, rng, intra_scale=intra, cross_scale=cross)
        s = block_scaled_random(bands, rng, intra_scale=intra, cross_scale=cross)
        terms.append(FourierTerm(gx=gx, gy=gy, cos=q @ c @ q.T, sin=q @ s @ q.T))

    return FourierModel(base=base, terms=tuple(terms))


def make_rotation_model(
    bands: int,
    seed: int,
    max_gap: float,
    angle_scale: float,
    rotations: int,
    variation_scale: float,
    modes: list[tuple[int, int]],
) -> RotationModel:
    rng = np.random.default_rng(seed)
    gaps = make_base_gaps(bands, max_gap)
    half = bands // 2

    terms = []
    for _ in range(rotations):
        if rng.random() < 0.7:
            i = int(rng.integers(0, half))
            j = int(rng.integers(half, bands))
        elif rng.random() < 0.5:
            i, j = rng.choice(half, size=2, replace=False)
            i = int(i)
            j = int(j)
        else:
            i, j = rng.choice(np.arange(half, bands), size=2, replace=False)
            i = int(i)
            j = int(j)

        gx, gy = modes[int(rng.integers(0, len(modes)))]
        freq = np.hypot(gx, gy)
        terms.append(
            RotationTerm(
                i=i,
                j=j,
                gx=gx,
                gy=gy,
                phase=float(rng.uniform(0.0, 2.0 * np.pi)),
                amplitude=angle_scale / np.sqrt(freq),
            )
        )

    variation_modes = np.array([modes[int(rng.integers(0, len(modes)))] for _ in range(bands)])
    variation = BandVariation(
        gx=variation_modes[:, 0],
        gy=variation_modes[:, 1],
        phase=rng.uniform(0.0, 2.0 * np.pi, size=bands),
        weight=rng.uniform(-1.0, 1.0, size=bands),
    )
    return RotationModel(
        gaps=gaps,
        rotations=tuple(terms),
        variation=variation,
        variation_scale=variation_scale,
    )


def make_tight_binding_model(
    bands: int,
    seed: int,
    max_gap: float,
    hopping_scale: float,
    cross_fraction: float,
    onsite_mix: float,
    ranges: list[tuple[int, int]],
    mu: float = 0.0,
) -> TightBindingModel:
    rng = np.random.default_rng(seed)
    gaps = make_base_gaps(bands, max_gap)
    q = random_unitary(bands, rng)

    onsite_eigenbasis = np.diag(gaps).astype(np.complex128)
    onsite_eigenbasis += onsite_mix * normalize_spectral(random_hermitian(bands, rng))
    onsite = q @ onsite_eigenbasis @ q.conj().T
    onsite = 0.5 * (onsite + onsite.conj().T)

    hoppings = []
    for rx, ry in ranges:
        distance = np.hypot(rx, ry)
        amp = hopping_scale / (distance**1.35)
        hopping_eigenbasis = block_scaled_complex_hopping(
            bands,
            rng,
            intra_scale=amp,
            cross_scale=amp * cross_fraction,
        )
        hopping = q @ hopping_eigenbasis @ q.conj().T
        hoppings.append(HoppingTerm(rx=rx, ry=ry, matrix=hopping))

    return TightBindingModel(onsite=onsite, hoppings=tuple(hoppings), mu=mu)


def triangle_vertices(n: int) -> Iterable[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]:
    h = 1.0 / n
    for ix in range(n):
        for iy in range(n):
            v00 = (ix * h, iy * h)
            v10 = ((ix + 1) * h, iy * h)
            v01 = (ix * h, (iy + 1) * h)
            v11 = ((ix + 1) * h, (iy + 1) * h)
            yield (v00, v10, v01)
            yield (v11, v01, v10)


def scan_mesh(model: FourierModel, n: int) -> MeshStats:
    grid = [[model.a((ix / n, iy / n)) for iy in range(n + 1)] for ix in range(n + 1)]
    weyl_passed = 0
    sb_passed = 0
    both_passed = 0
    min_weyl_slack = np.inf
    min_sb_margin = np.inf

    for ix in range(n):
        for iy in range(n):
            triangle_mats = (
                (grid[ix][iy], grid[ix + 1][iy], grid[ix][iy + 1]),
                (grid[ix + 1][iy + 1], grid[ix][iy + 1], grid[ix + 1][iy]),
            )
            for mats in triangle_mats:
                weyl = weyl_certificate(mats)
                sb = sign_block_certificate(mats)

                if weyl.passed:
                    weyl_passed += 1
                if sb.passed:
                    sb_passed += 1
                if weyl.passed and sb.passed:
                    both_passed += 1

                min_weyl_slack = min(min_weyl_slack, 1.0 - weyl.best_value)
                min_sb_margin = min(min_sb_margin, sb.best_value)

    triangles = 2 * n * n
    return MeshStats(
        n=n,
        triangles=triangles,
        weyl_passed=weyl_passed,
        sb_passed=sb_passed,
        both_passed=both_passed,
        min_weyl_slack=float(min_weyl_slack),
        min_sb_margin=float(min_sb_margin),
    )


def triangle_is_vertex_cut(mats: tuple[Array, ...]) -> bool:
    """Heuristic metal cut detector from vertex spectra.

    This is intentionally a practical marker, not a proof of all possible
    interior crossings for nonlinear H(k).  It flags triangles whose vertex
    occupation count changes or whose sorted vertex eigenvalues straddle mu=0.
    """

    evals = [np.linalg.eigvalsh(a) for a in mats]
    occupations = [int(np.sum(e < 0.0)) for e in evals]
    if len(set(occupations)) > 1:
        return True

    stacked = np.stack(evals, axis=0)
    return bool(np.any((np.min(stacked, axis=0) <= 0.0) & (np.max(stacked, axis=0) >= 0.0)))


def triangle_min_vertex_gap(mats: tuple[Array, ...]) -> float:
    return min(float(np.min(np.abs(np.linalg.eigvalsh(a)))) for a in mats)


def scan_mesh_resolved(model: FourierModel, n: int, cut_h: float, cut_energy: float) -> ResolutionStats:
    grid = [[model.a((ix / n, iy / n)) for iy in range(n + 1)] for ix in range(n + 1)]
    h = 1.0 / n
    cut_allowed = h <= cut_h

    weyl_safe = 0
    weyl_cut = 0
    weyl_unresolved = 0
    sb_safe = 0
    sb_cut = 0
    sb_unresolved = 0
    raw_cut = 0
    min_weyl_slack = np.inf
    min_sb_margin = np.inf

    for ix in range(n):
        for iy in range(n):
            triangle_mats = (
                (grid[ix][iy], grid[ix + 1][iy], grid[ix][iy + 1]),
                (grid[ix + 1][iy + 1], grid[ix][iy + 1], grid[ix + 1][iy]),
            )
            for mats in triangle_mats:
                is_cut = triangle_is_vertex_cut(mats)
                if not is_cut and cut_energy > 0.0:
                    is_cut = triangle_min_vertex_gap(mats) <= cut_energy
                raw_cut += int(is_cut)

                weyl = weyl_certificate(mats)
                sb = sign_block_certificate(mats)
                min_weyl_slack = min(min_weyl_slack, 1.0 - weyl.best_value)
                min_sb_margin = min(min_sb_margin, sb.best_value)

                if weyl.passed:
                    weyl_safe += 1
                elif is_cut and cut_allowed:
                    weyl_cut += 1
                else:
                    weyl_unresolved += 1

                if sb.passed:
                    sb_safe += 1
                elif is_cut and cut_allowed:
                    sb_cut += 1
                else:
                    sb_unresolved += 1

    triangles = 2 * n * n
    return ResolutionStats(
        n=n,
        triangles=triangles,
        cut_h=cut_h,
        weyl_safe=weyl_safe,
        weyl_cut=weyl_cut,
        weyl_unresolved=weyl_unresolved,
        sb_safe=sb_safe,
        sb_cut=sb_cut,
        sb_unresolved=sb_unresolved,
        raw_cut=raw_cut,
        min_weyl_slack=float(min_weyl_slack),
        min_sb_margin=float(min_sb_margin),
    )


def sample_gap(model: FourierModel, sample_n: int) -> tuple[int, float, float]:
    occupancies = []
    min_gap = np.inf
    max_radius = 0.0
    for ix in range(sample_n):
        x = ix / sample_n
        for iy in range(sample_n):
            y = iy / sample_n
            evals = np.linalg.eigvalsh(model.a((x, y)))
            occupancies.append(int(np.sum(evals < 0.0)))
            min_gap = min(min_gap, float(np.min(np.abs(evals))))
            max_radius = max(max_radius, float(np.max(np.abs(evals))))
    return len(set(occupancies)), min_gap, max_radius


def find_first_all(model: FourierModel, ns: list[int]) -> tuple[int | None, int | None, list[MeshStats]]:
    first_weyl: int | None = None
    first_sb: int | None = None
    stats: list[MeshStats] = []
    for n in ns:
        stat = scan_mesh(model, n)
        stats.append(stat)
        if first_weyl is None and stat.weyl_all:
            first_weyl = n
        if first_sb is None and stat.sb_all:
            first_sb = n
        if first_weyl is not None and first_sb is not None:
            break
    return first_weyl, first_sb, stats


def parse_modes(text: str) -> list[tuple[int, int]]:
    modes = []
    for item in text.split(","):
        gx_text, gy_text = item.split(":")
        modes.append((int(gx_text), int(gy_text)))
    return modes


def parse_ranges(text: str) -> list[tuple[int, int]]:
    ranges = []
    for item in text.split(","):
        rx_text, ry_text = item.split(":")
        ranges.append((int(rx_text), int(ry_text)))
    return ranges


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["tight-binding", "rotation", "fourier"], default="tight-binding")
    parser.add_argument("--bands", type=int, default=40)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-gap", type=float, default=8.0)
    parser.add_argument("--perturb", type=float, default=0.65)
    parser.add_argument("--cross-fraction", type=float, default=0.35)
    parser.add_argument("--angle-scale", type=float, default=0.22)
    parser.add_argument("--rotations", type=int, default=120)
    parser.add_argument("--variation-scale", type=float, default=0.25)
    parser.add_argument("--hopping-scale", type=float, default=0.22)
    parser.add_argument("--onsite-mix", type=float, default=0.25)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--cut-h", type=float, default=0.0, help="accept vertex-cut triangles when 1/n <= cut_h; 0 disables")
    parser.add_argument("--cut-energy", type=float, default=0.0, help="also treat triangles with a vertex gap below this as near-Fermi cuts")
    parser.add_argument("--ranges", default="1:0,0:1,1:1,1:-1,2:0,0:2,2:1,1:2,2:-1,1:-2")
    parser.add_argument("--modes", default="1:0,0:1,1:1,2:1,1:2,2:0,0:2")
    parser.add_argument("--sample-gap-n", type=int, default=31)
    parser.add_argument("--ns", type=int, nargs="+", default=[2, 4, 8, 12, 16, 24, 32, 48, 64])
    args = parser.parse_args()

    modes = parse_modes(args.modes)
    if args.model == "tight-binding":
        model = make_tight_binding_model(
            bands=args.bands,
            seed=args.seed,
            max_gap=args.max_gap,
            hopping_scale=args.hopping_scale,
            cross_fraction=args.cross_fraction,
            onsite_mix=args.onsite_mix,
            ranges=parse_ranges(args.ranges),
            mu=args.mu,
        )
    elif args.model == "rotation":
        model = make_rotation_model(
            bands=args.bands,
            seed=args.seed,
            max_gap=args.max_gap,
            angle_scale=args.angle_scale,
            rotations=args.rotations,
            variation_scale=args.variation_scale,
            modes=modes,
        )
    else:
        model = make_model(
            bands=args.bands,
            seed=args.seed,
            max_gap=args.max_gap,
            perturb=args.perturb,
            cross_fraction=args.cross_fraction,
            modes=modes,
        )

    print(f"Random {args.model} BZ model")
    print(f"  BZ: unit square, area = 1")
    print(f"  bands: {args.bands}")
    print(f"  max_gap: {args.max_gap:g}")
    if args.model == "tight-binding":
        print(f"  hopping_scale: {args.hopping_scale:g}")
        print(f"  cross_fraction: {args.cross_fraction:g}")
        print(f"  onsite_mix: {args.onsite_mix:g}")
        print(f"  mu: {args.mu:g}")
        print(f"  hopping ranges: {args.ranges}")
    elif args.model == "rotation":
        print(f"  angle_scale: {args.angle_scale:g}")
        print(f"  rotations: {args.rotations}")
        print(f"  variation_scale: {args.variation_scale:g}")
        print(f"  modes: {args.modes}")
    else:
        print(f"  perturb: {args.perturb:g}")
        print(f"  cross_fraction: {args.cross_fraction:g}")
        print(f"  modes: {args.modes}")

    distinct_occ, min_gap, max_radius = sample_gap(model, args.sample_gap_n)
    print(f"  sampled distinct occupation counts: {distinct_occ}")
    print(f"  sampled min |eig|: {min_gap:.6f}")
    print(f"  sampled max |eig|: {max_radius:.6f}")
    print()

    if args.cut_h > 0.0:
        print(f"Resolved metal scan with cut_h={args.cut_h:g}, cut_energy={args.cut_energy:g}")
        print("  n    triangles  raw cuts  Weyl safe/cut/unres        SB safe/cut/unres          min Weyl slack  min SB margin")
        first_weyl_resolved = None
        first_sb_resolved = None
        started = time.perf_counter()
        stats = []
        for n in args.ns:
            stat = scan_mesh_resolved(model, n, cut_h=args.cut_h, cut_energy=args.cut_energy)
            stats.append(stat)
            print(
                f"  {n:3d}  {stat.triangles:9d}  {stat.raw_cut:8d}  "
                f"{stat.weyl_safe:5d}/{stat.weyl_cut:<5d}/{stat.weyl_unresolved:<5d}  "
                f"{stat.sb_safe:5d}/{stat.sb_cut:<5d}/{stat.sb_unresolved:<5d}  "
                f"{stat.min_weyl_slack:14.6f}  "
                f"{stat.min_sb_margin:13.6f}"
            )
            if first_weyl_resolved is None and stat.weyl_resolved_all:
                first_weyl_resolved = n
            if first_sb_resolved is None and stat.sb_resolved_all:
                first_sb_resolved = n
            if first_weyl_resolved is not None and first_sb_resolved is not None:
                break

        elapsed = time.perf_counter() - started
        if first_weyl_resolved is None:
            print(f"  Weyl+cut did not resolve all triangles up to n={args.ns[-1]}.")
        else:
            print(f"  Weyl+cut first resolves whole BZ at n={first_weyl_resolved}: {2 * first_weyl_resolved * first_weyl_resolved} triangles.")

        if first_sb_resolved is None:
            print(f"  Sign-block+cut did not resolve all triangles up to n={args.ns[-1]}.")
        else:
            print(f"  Sign-block+cut first resolves whole BZ at n={first_sb_resolved}: {2 * first_sb_resolved * first_sb_resolved} triangles.")

        if first_weyl_resolved is not None and first_sb_resolved is not None:
            tri_ratio = (first_weyl_resolved / first_sb_resolved) ** 2
            print(f"  Triangle-count ratio estimate: {tri_ratio:.1f}x fewer triangles for sign-block+cut.")
        print(f"  resolved scan wall time: {elapsed:.3f}s")
        return

    started = time.perf_counter()
    first_weyl, first_sb, stats = find_first_all(model, args.ns)
    elapsed = time.perf_counter() - started

    print("Mesh scan")
    print("  n    triangles  Weyl pass      SB pass        min Weyl slack  min SB margin")
    for stat in stats:
        print(
            f"  {stat.n:3d}  {stat.triangles:9d}  "
            f"{stat.weyl_passed:6d}/{stat.triangles:<6d}  "
            f"{stat.sb_passed:6d}/{stat.triangles:<6d}  "
            f"{stat.min_weyl_slack:14.6f}  "
            f"{stat.min_sb_margin:13.6f}"
        )
    print()

    if first_weyl is None:
        print(f"  Weyl did not certify all triangles up to n={args.ns[-1]}.")
    else:
        print(f"  Weyl first certifies whole BZ at n={first_weyl}: {2 * first_weyl * first_weyl} triangles.")

    if first_sb is None:
        print(f"  Sign-block did not certify all triangles up to n={args.ns[-1]}.")
    else:
        print(f"  Sign-block first certifies whole BZ at n={first_sb}: {2 * first_sb * first_sb} triangles.")

    if first_weyl is not None and first_sb is not None:
        tri_ratio = (first_weyl / first_sb) ** 2
        print(f"  Triangle-count ratio estimate: {tri_ratio:.1f}x fewer triangles for sign-block.")

    print(f"  scan wall time: {elapsed:.3f}s")


if __name__ == "__main__":
    main()
