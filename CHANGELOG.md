# Changelog

All notable changes to FermiSimplex are documented in this file.

## Unreleased

## 0.2.0 - 2026-08-06

- Replaced the projected charge-error heuristic with a fixed-depth recursive
  active-space estimator using corrected frozen-Schur reductions and actual
  Hamiltonian evaluations at temporary microsimplex vertices.
- Added terminal midpoint non-affinity sampling, shifted simplex volumes, a
  small-space recursion rule, and a tighter large-active-space fallback.
- Added `error_depth` and detailed charge-estimator work statistics to the C++
  and Python APIs.
- Added one-correction frozen safe-block elimination using thin matrix products
  and per-layer reduced-model caches; no pointwise safe-block factorization is
  required.
- Removed the charge-only curvature, preview-depth, certified-error-bound, and
  legacy projected-error APIs. This is an intentional pre-1.0 API break.

## 0.1.0 - 2026-07-28

- Added adaptive Fermi-surface extraction, charge integration, and density
  matrices through a shared Python and C++ numerical core.
- Added occupation certification, projected charge-error estimation, and
  reusable eigensystem caching on adaptive simplex meshes.
- Added production tests, performance benchmarks, examples, and visual
  documentation.
