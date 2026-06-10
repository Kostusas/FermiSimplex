# LinearTetrahedron

LinearTetrahedron is the zero-temperature linear tetrahedron backend used by
MeanFi. It owns the physics-specific C++/Python boundary:

- dense tight-binding Fourier evaluation,
- LAPACK diagonalization at adaptive simplex vertices,
- zero-temperature occupation and density contribution rules,
- Python packaging and result conversion.

The adaptive mesh mechanics come from the refactored C++20 AdaptiveSimplex
library. A pinned source-only copy is vendored as a fallback so normal Python
package builds do not need a separate AdaptiveSimplex install. If an external
AdaptiveSimplex CMake package is available, the build uses that package instead.

## Local Development

The Pixi `test` task installs the package in editable mode and runs pytest.

```bash
pixi run test
```

To test against an external AdaptiveSimplex install instead of the vendored
fallback, point CMake at that prefix:

```bash
CMAKE_PREFIX_PATH=/path/to/adaptivesimplex/prefix pixi run test
```
