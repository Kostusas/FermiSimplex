# Performance measurement

FermiSimplex measures computational cost relative to the same complex
Hermitian LAPACK eigensolve used by the library. The benchmark suite is C++
only; Python startup, callbacks, and bindings are deliberately excluded.

## Build and run

Use an optimized build and one BLAS/LAPACK thread:

```sh
cmake -S . -B build/cpp-benchmarks -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DFERMISIMPLEX_BUILD_PYTHON=OFF \
  -DFERMISIMPLEX_BUILD_TESTING=OFF \
  -DFERMISIMPLEX_BUILD_BENCHMARKS=ON
cmake --build build/cpp-benchmarks \
  --target fermisimplex_run_performance_benchmark
```

The custom target writes
`build/cpp-benchmarks/benchmarks/fermisimplex-performance.json`. The
equivalent Pixi command is:

```sh
pixi run benchmark-cpp
```

The executable also supports direct use:

```sh
build/cpp-benchmarks/cpp/fermisimplex_performance_benchmark \
  --preset quick \
  --output result.json
```

Available presets are `quick`, `ci`, and `full`. Use `quick` only as a smoke
test. The `ci` preset has stable cases and sample counts intended for version
tracking. `full` adds larger matrices and more tight-binding terms.

Each preset performs an untimed LAPACK warm-up before collecting samples. This
reduces process-start, dynamic-library, and CPU-frequency effects without
mixing warm-up work into reported timings.

## Measurements

The headline metric is total LAPACK-equivalents per newly evaluated vertex:
`total operation time / (new vertices * LAPACK time)`.

This includes vertex evaluation, diagonalization, caching, every simplex-rule
or classification call, refinement bookkeeping, and result construction. It
excludes `SpectralMesh` construction. The target is therefore stated directly:
`total_lapack_equivalents_per_vertex <= 2`.

The secondary metric divides the same total time by actual simplex visits.
For charge, visits include coarse and preview contributions. For Fermi
surfaces, visits count every classified frontier simplex. Neither metric uses
the final active-simplex count as a proxy for work.

The terminal summary puts the primary metric first, reports the measured
LAPACK time beside it, and marks whether each case meets the two-solve target.
It shows per-simplex timings only as a secondary diagnostic for the largest
simplex matrix size in the selected preset.

The end-to-end workloads are:

- `charge_current_mesh_total`: one complete charge estimate on a fixed 2D
  mesh with preview depth one;
- `charge_adaptive_total`: converged adaptive charge integration of a crossing
  dense model;
- `fermi_surface_total`: adaptive 2D Fermi-surface refinement of a crossing
  dense model down to the requested feature size.

Supporting diagnostic measurements report:

- `lapack_reused_workspace`: `zheevd` with one workspace query and reused work
  arrays. Input matrices are prepared outside the timer, so this measures the
  LAPACK call itself;
- `lapack_current_wrapper`: the current FermiSimplex wrapper, including its
  per-call workspace query and allocations;
- `lapack_reference_best`: the faster of those two measured LAPACK paths for
  the current matrix size and LAPACK provider;
- model evaluation, Hamiltonian validation, spectrum construction, and cache
  insertion per vertex;
- certification, charge integration, and Fermi classification per simplex;
- controlled root-mesh evaluation and classification with deterministic
  vertex and simplex counts.

Diagnostic results include `lapack_equivalents_per_operation`, normalized to
`lapack_reference_best`. End-to-end results additionally include total time,
new vertices, actual simplex visits, refinements, time per vertex and visit,
and LAPACK-equivalents per vertex and visit. Both raw LAPACK paths remain in
the output because small-matrix behavior can depend on the LAPACK provider.

The JSON schema uses stable benchmark names and records the commit, dirty-tree
state, compiler, build type, system, LAPACK linkage, thread settings, counts,
median, range, and median absolute deviation. Compare
`median_ns_per_operation` only between runs on the same runner. The
LAPACK-equivalent ratios are more portable, but still require the same LAPACK
implementation and thread configuration.

## CI use

Run performance jobs on a fixed or dedicated runner. Shared cloud runners can
be useful for collecting artifacts, but their timing variance makes strict
regression thresholds unreliable.

For each revision:

1. build in `Release` mode;
2. run `fermisimplex_run_performance_benchmark`;
3. retain the JSON file as a CI artifact;
4. compare cases by the tuple `(name, ndim, ndof, hopping_terms, root_level)`;
5. graph `total_lapack_equivalents_per_vertex` for end-to-end workloads;
6. use per-simplex and phase timings to diagnose changes;
7. flag a regression only when it exceeds both a relative threshold and the
   observed run-to-run dispersion.

A practical initial policy is to report changes above 5% and fail only after a
repeat confirms a change above 10%. Counts such as vertices and simplices are
deterministic and may use exact comparisons.

Very small absolute timings remain sensitive to clock resolution, CPU state,
and allocator behavior even when operations are batched. Apply regression
thresholds to the medium and large matrix cases first; retain the smallest
cases primarily to track scaling and fixed overhead.
