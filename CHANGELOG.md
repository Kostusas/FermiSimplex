# Changelog

All notable changes to FermiSimplex are documented in this file.

## Unreleased

## 0.2.0 - 2026-08-07

- Replaced the projected charge-error heuristic with a fixed-depth recursive
  active-space estimator using corrected frozen-Schur reductions and actual
  Hamiltonian evaluations at temporary microsimplex vertices.
- Added terminal midpoint non-affinity sampling, shifted simplex volumes, a
  small-space recursion rule, and a tighter large-active-space fallback.
- Added `error_depth` and detailed charge-estimator work statistics to the C++
  and Python APIs.
- Set the default charge `error_depth` to 2.
- Made current-mesh charge a direct linear-simplex calculation with no target,
  recursive error estimation, or temporary sampling.
- Allowed density-matrix `preview_depth=0` to integrate the existing mesh
  without preview evaluations or refinement.
- Applied the corrected frozen Schur model through a per-layer dense safe
  resolvent, two reused thin work buffers, and cached reduced matrices; no
  pointwise safe-block factorization is required.
- Reused certified chemical-potential intervals for terminal curvature radii
  whenever re-anchoring has not changed the effective model.
- Removed repeated Hamiltonian coordinate, shape, finiteness, and Hermiticity
  validation from the evaluation hot path; model values are trusted after
  setup.
- Removed the charge-only curvature, preview-depth, certified-error-bound, and
  legacy projected-error APIs. This is an intentional pre-1.0 API break.

## 0.1.0 - 2026-07-28

- Added adaptive Fermi-surface extraction, charge integration, and density
  matrices through a shared Python and C++ numerical core.
- Added occupation certification, projected charge-error estimation, and
  reusable eigensystem caching on adaptive simplex meshes.
- Added production tests, performance benchmarks, examples, and visual
  documentation.
