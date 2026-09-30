"""Bounded, repeatable charge/certification comparison with exact references.

Run with BLAS and OpenMP restricted to one thread. Historical comparisons run this script from its corresponding Git revision.
All runs use identical initial meshes, depths, targets and refinement caps. Each run has a
30-second alarm; a timeout or resource failure is recorded, not discarded.
"""

import argparse
import json
from pathlib import Path
import signal
from statistics import median
from time import perf_counter

import numpy as np

from fermisimplex import CertificateStatus, SpectralMesh, certify_simplex


def multiband(size):
    safe = np.r_[
        -np.linspace(2, 6, (size - 2) // 2),
        np.linspace(4, 8, size - 2 - (size - 2) // 2),
    ]
    onsite = np.diag(np.r_[0.0, 3.0, safe]).astype(complex)
    forward = np.zeros_like(onsite)
    forward[0, 0], forward[1, 1] = 0.5, 0.1
    forward[0, 1] = forward[1, 0] = 0.2j
    # A fixed complex basis change exercises matrix arithmetic without changing
    # the exact spectrum or the occupied measure.
    rng = np.random.default_rng(14)
    rotation, _ = np.linalg.qr(
        rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    )
    tb = {
        key: rotation @ matrix @ rotation.conj().T
        for key, matrix in {
            (0,): onsite,
            (1,): forward,
            (-1,): forward.conj().T,
        }.items()
    }
    mu = 0.23
    threshold = [
        r.real
        for r in np.roots([0.36, 3 - 1.2 * mu, mu * mu - 3 * mu - 0.16])
        if abs(r.imag) < 1e-14 and -1 < r.real < 1
    ][0]
    reference = (safe < mu).sum() + 1 - np.arccos(threshold) / np.pi
    return tb, mu, float(reference), 1


def models():
    yield (
        "cosine",
        (
            {(1,): np.array([[0.5]]), (-1,): np.array([[0.5]])},
            0.23,
            1 - np.arccos(0.23) / np.pi,
            1,
        ),
    )
    yield (
        "pocket",
        (lambda x: np.array([[(x - 0.37) ** 2 - 0.04**2]], complex), 0.0, 0.08, 1),
    )
    yield (
        "disk",
        (
            lambda x, y: np.array(
                [[(x - 0.33) ** 2 + (y - 0.47) ** 2 - 0.12**2]], complex
            ),
            0.0,
            np.pi * 0.12**2,
            2,
        ),
    )
    yield (
        "annulus",
        (
            lambda x, y: np.array(
                [
                    [
                        (((x - 0.5) ** 2 + (y - 0.5) ** 2) - 0.16**2)
                        * (((x - 0.5) ** 2 + (y - 0.5) ** 2) - 0.29**2)
                    ]
                ],
                complex,
            ),
            0.0,
            np.pi * (0.29**2 - 0.16**2),
            2,
        ),
    )
    yield (
        "dirac",
        (
            lambda x, y: np.array(
                [[y - 0.5, x - 0.5 - 0.05j], [x - 0.5 + 0.05j, 0.5 - y]]
            ),
            0.23,
            1 + np.pi * (0.23**2 - 0.05**2),
            2,
        ),
    )
    for size in (12, 36):
        yield f"mixed_{size}", multiband(size)
    tb, mu, reference, dimension = multiband(36)

    def callable_model(x):
        return sum(
            matrix * np.exp(-2j * np.pi * key[0] * x) for key, matrix in tb.items()
        )

    yield "mixed_36_callable", (callable_model, mu, reference, dimension)


def alarm(*_):
    raise TimeoutError("30 second benchmark limit")


def charge_runs(repeats):
    rows = []
    for name, (model, mu, reference, dimension) in models():
        for target in (1e100, 1e-3, 1e-5 if dimension == 1 else 1e-4):
            row = dict(
                model=name,
                method="occupation",
                target=target,
                reference=reference,
                root_level=2,
                depth=2,
            )
            times = []
            try:
                for _ in range(repeats):
                    signal.alarm(30)
                    mesh = SpectralMesh(model, root_level=2)
                    start = perf_counter()
                    result = mesh.integrate_charge(
                        mu=mu,
                        target_error=target,
                        max_refinements=3000,
                        error_depth=2,
                    )
                    times.append(perf_counter() - start)
                    signal.alarm(0)
                error = abs(result.value - reference)
                stats = result.error_stats
                row.update(
                    value=result.value,
                    actual_error=error,
                    estimated_error=result.stopping_error,
                    covered=bool(error <= result.stopping_error + 1e-12),
                    density_cut_error=result.density_cut_error,
                    seconds=median(times),
                    timing_samples=times,
                    vertices=mesh.active_vertices,
                    leaves=mesh.active_simplices,
                    hamiltonian_evaluations=result.stats.evaluations
                    + stats.hamiltonian_evaluations,
                    eigensystems=result.stats.evaluations
                    + stats.full_eigensystems
                    + stats.reduced_eigensystems
                    + stats.norm_eigensystems,
                    refinements=result.stats.refinements,
                    reductions=stats.schur_reductions,
                )
            except (RuntimeError, TimeoutError) as error:
                row["failure"] = str(error)
            finally:
                signal.alarm(0)
            rows.append(row)
            print(json.dumps(row), flush=True)
    return rows


def certification_runs():
    rows = []
    for h in (1.0, 0.1, 0.01):
        for center in (0.17, 0.31, 0.47, 0.63, 0.81):
            for radius in (0.015, 0.05, 0.12):
                for pocket in (True, False):
                    sign = -1 if pocket else 1

                    def model(x):
                        return np.array(
                            [[h * h * ((x - center) ** 2 + sign * radius**2)]], complex
                        )

                    values, vectors = np.linalg.eigh(np.array([model(0), model(1)]))
                    legacy = certify_simplex(
                        values, vectors, linearization_error_bound=0
                    )
                    new = SpectralMesh(model, root_level=0).occupation_enclosures(
                        mu=0, depth=2
                    )
                    deeper = SpectralMesh(model, root_level=0).occupation_enclosures(
                        mu=0, depth=6
                    )
                    rows.append(
                        dict(
                            h=h,
                            center=center,
                            radius=radius,
                            pocket=pocket,
                            legacy_gapped=legacy.status
                            is CertificateStatus.CertifiedGapped,
                            quadratic_gapped=all(e.fixed_occupation for e in new),
                            quadratic_depth6_gapped=all(
                                e.fixed_occupation for e in deeper
                            ),
                        )
                    )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    signal.signal(signal.SIGALRM, alarm)
    results = dict(charge=charge_runs(args.repeats), certification=certification_runs())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
