# Projected-error estimator experiments

> **Status: analysis only.** Nothing in this directory is a supported
> FermiSimplex API, and the prototype changes on this branch are not proposed
> for `main`.

This experiment investigates why an affine matrix Hamiltonian can have curved
ordered eigenvalues even though the old residual

\[
H(k)-H_{\mathrm{linear}}(k)
\]

vanishes. The immediate goal is a useful adaptive charge-error estimate, not a
new certificate.

## Branch layout

- `ablation.cpp` is the controlled comparison to continue from. It keeps the
  target-band union, endpoint envelope, and single-anchor methods independent.
- `cpp/src/integration/projected_error.cpp` contains the earlier bundled
  prototype: vertex union, guard bands, residual inflation, and a cost-based
  full-eigenvalue fallback. It is retained to preserve the experiment and its
  focused regression, not because that design is recommended.
- `FERMISIMPLEX_BUILD_EXPERIMENTS` is disabled by default, so the ablation
  program is never part of an ordinary build.

## What is being compared

For a one-dimensional affine random pencil

\[
H(x)=A+xB,\qquad x\in[0,h],
\]

the program diagonalizes the endpoints and uses the exact midpoint
eigensystem only as reference data. It compares three estimators at the
midpoint.

### 1. Single anchor

Choose the endpoint with the largest gap around the uncertain band block
\([L,U)\). If \(V_i\) contains that endpoint's uncertain-band eigenvectors,
diagonalize the small matrix

\[
V_i^\dagger H(h/2)V_i.
\]

### 2. Endpoint envelope

Perform the same small projection independently in every endpoint subspace and
retain the largest positive and negative interpolation deviations:

\[
\epsilon_+=\max_i\epsilon_{i,+},\qquad
\epsilon_-=\max_i\epsilon_{i,-}.
\]

This uses no extra bands, no interpolated eigenspace, and no full sampled
eigensolve.

### 3. Target-band union

Form only the span of the endpoint uncertain-band subspaces,

\[
\mathcal U=\operatorname{span}(V_0,V_1),
\]

interpolate their projectors inside \(\mathcal U\), select an
\((U-L)\)-dimensional midpoint subspace, and project \(H(h/2)\) into it.

This is the clean union ablation: it deliberately contains no guard bands and
no full-eigensystem fallback.

## Terminology and counters

- `h` is the interval length. Smaller `h` models a refined simplex.
- `certified interval` in early notes meant the **certificate-selected
  occupation-uncertain interval** \([L,U)\). The projected estimator itself is
  not certified.
- `central singleton` forcibly tests only the central ordered band. It is a
  diagnostic, not the production band-selection rule.
- `196/494 underestimates` means that 500 random pencils were generated, 494
  had a nonempty occupation-uncertain interval, and the estimator was smaller
  than the exact midpoint ordered-band defect by more than \(10^{-8}\) in 196
  of those 494 cases. It does not mean that 196 charge integrals failed.

For each exact midpoint band defect

\[
d_j=\lambda_j(h/2)-\frac{\lambda_j(0)+\lambda_j(h)}{2},
\]

the diagnostic compares both directions:

\[
\epsilon_+^\star=\max_j\max(d_j,0),\qquad
\epsilon_-^\star=\max_j\max(-d_j,0).
\]

## Reproduce

Configure an isolated release build:

```sh
cmake -S . -B build/projected-error-experiment -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DFERMISIMPLEX_BUILD_PYTHON=OFF \
  -DFERMISIMPLEX_BUILD_TESTING=OFF \
  -DFERMISIMPLEX_BUILD_BENCHMARKS=OFF \
  -DFERMISIMPLEX_BUILD_EXPERIMENTS=ON
```

Build and run:

```sh
cmake --build build/projected-error-experiment \
  --target fermisimplex_projected_error_ablation

OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
build/projected-error-experiment/cpp/fermisimplex_projected_error_ablation
```

The timing section measures only the estimator linear algebra at one midpoint.
Endpoint eigensystems and the midpoint Hamiltonian are already available.
Hamiltonian construction, certification, adaptive refinement, simplex cuts,
and vertex eigensolves are excluded. The full LAPACK column is reference-only.

## Representative results

One local single-threaded Apple Accelerate run produced:

| Random \(8\times8\) diagnostic | Single anchor | Endpoint envelope | Union |
|---|---:|---:|---:|
| uncertain interval, \(h=1\) | 375/494 | 196/494 | 313/494 |
| uncertain interval, \(h=0.25\) | 111/297 | 30/297 | 137/297 |
| uncertain interval, \(h=0.125\) | 41/129 | 7/129 | 45/129 |
| central singleton, \(h=0.125\) | 98/500 | 1/500 | 182/500 |

Worst missed midpoint error for the central singleton at \(h=0.125\):

| Method | Worst shortfall |
|---|---:|
| Single anchor | \(7.16\times10^{-2}\) |
| Endpoint envelope | \(2.00\times10^{-4}\) |
| Union | \(5.60\times10^{-2}\) |

Estimator-only timings in microseconds:

| Matrix size | Uncertain bands | Anchor | Envelope | Union | Full LAPACK reference |
|---:|---:|---:|---:|---:|---:|
| 8 | 1 | 0.47 | 0.91 | 1.53 | 3.59 |
| 8 | 3 | 1.02 | 2.10 | 5.30 | 3.53 |
| 32 | 1 | 2.09 | 4.27 | 3.95 | 76.90 |
| 32 | 3 | 6.23 | 12.62 | 13.69 | 77.39 |
| 60 | 1 | 4.64 | 10.83 | 7.98 | 396.03 |
| 60 | 3 | 8.55 | 17.25 | 19.50 | 394.86 |

Timings are machine-dependent; the relative behavior is the intended result.

## Current interpretation

The union changes the answer, but it is not consistently better. It supplies a
larger set of candidate directions and then still needs a heuristic choice of
the midpoint subspace. Projector interpolation follows eigenvector character,
while charge integration needs a globally ordered band index. Those can differ
near mixing and crossings.

The endpoint envelope is simpler and substantially more reliable in these
tests. It avoids inventing an interpolated subspace and asks every real vertex
subspace for an estimate. It remains an adaptive estimate rather than a
certificate: the true interior eigenspace can rotate outside all selected
vertex blocks.

## Recommended next experiment

1. Discard guard bands and full sampled eigensolve fallbacks for now.
2. Implement only the endpoint envelope behind an explicitly experimental
   code path.
3. Measure actual charge error, refinement count, and estimator-only time on
   the same deterministic workloads.
4. Compare against the single-anchor estimator before considering any
   production change.
