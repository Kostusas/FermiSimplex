# LinearTetrahedron

LinearTetrahedron is the zero-temperature linear tetrahedron backend used by
MeanFi. It owns the physics-specific C++/Python boundary:

- dense tight-binding Fourier evaluation,
- LAPACK diagonalization at adaptive simplex vertices,
- zero-temperature occupation and density contribution rules,
- Python packaging and result conversion.

The adaptive mesh mechanics come from the refactored C++20 AdaptiveSimplex
library, which must be available as a CMake package.

## Local Development

```bash
CMAKE_PREFIX_PATH=/path/to/adaptivesimplex/prefix pixi run test
```

The Pixi `test` task installs the package in editable mode and runs pytest.
When AdaptiveSimplex is installed into a standard CMake prefix, the
`CMAKE_PREFIX_PATH` override is not needed.

```bash
pixi run test
```
