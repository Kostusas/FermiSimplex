# Development and architecture

FermiSimplex keeps the reusable numerical library and Python interface in
separate source trees:

```text
cpp/include/fermisimplex/   public C++ API
cpp/src/core/                    Hamiltonians, mesh, eigensystem cache
cpp/src/certification/           occupation proofs and bounds
cpp/src/integration/             charge and density-matrix rules
cpp/src/fermi_surface/           refinement and surface extraction
cpp/src/linalg/                  narrow BLAS/LAPACK wrappers
python/fermisimplex/        small public Python layer
python/src/bindings/             nanobind interface
examples/                        runnable examples
benchmarks/                      maintained stress cases
docs/                            theory and reproducible README assets
```

AdaptiveSimplex owns generic geometry, refinement, vertex caching, cut-simplex
moments, and adaptive-loop mechanics. FermiSimplex adds only the spectral
rules and certificates.

## Python development

Pixi configures the C++ tests, installs an editable Python package against the
same AdaptiveSimplex source, and runs both test suites:

```bash
pixi run test
```

Useful individual tasks are

```bash
pixi run test-cpp
pixi run test-python
pixi run smoke-test
pixi run build-package
```

The optional plotting dependencies are declared as `examples` extras in
`pyproject.toml`.

## Standalone C++ library

The C++ numerical core can be configured without Python or nanobind:

```bash
cmake -S cpp -B build/core \
  -DFERMISIMPLEX_BUILD_TESTING=ON
cmake --build build/core
ctest --test-dir build/core --output-on-failure
cmake --install build/core --prefix /desired/prefix
```

Installed consumers use

```cmake
find_package(fermisimplex CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE fermisimplex::core)
```

and include the umbrella header:

```cpp
#include <fermisimplex/fermisimplex.h>
```

The package exports AdaptiveSimplex as a public dependency because
`SpectralMesh` intentionally exposes its geometry and eigensystem cache to C++
consumers.

## CMake options

- `FERMISIMPLEX_BUILD_PYTHON`
- `FERMISIMPLEX_BUILD_TESTING`
- `FERMISIMPLEX_BUILD_BENCHMARKS`
- `FERMISIMPLEX_INSTALL_CPP`

Python and tests default on only for a top-level build. Benchmarks remain
opt-in.

## Documentation assets

Regenerate the committed README visuals with

```bash
pixi run python -m docs.tools.generate_readme_assets
```

The generator exercises only the public Python API. See
[visuals.md](visuals.md) for the models and output policy.
