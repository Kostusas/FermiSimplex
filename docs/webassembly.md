# Building for Pyodide

Native builds retain their existing BLAS provider, ABI, compiler settings and
OpenMP defaults. The optional Pyodide build selects the target runtime's LP64
f2c OpenBLAS provider; it does not change algorithms or the Python API.

In a configured `pyodide-build` environment matching the target runtime:

```sh
FERMISIMPLEX_PYODIDE=1 pyodide build . \
  -Ccmake.define.FERMISIMPLEX_PYODIDE_OPENBLAS=/path/to/target/libopenblas.so
```

Use the OpenBLAS library from that Pyodide distribution, never a host library.
The build skips the native `scipy-openblas32` dependency, uses Pyodide's exact
Fortran function signatures, and defaults density OpenMP to off. It requires an
Emscripten toolchain. Other Emscripten BLAS providers are not assumed to share
Pyodide's ABI.

Load `scipy` with `pyodide.loadPackage` before installing/importing this wheel;
it loads the runtime's shared OpenBLAS library. A Pyodide distribution recipe
should declare this dependency. Native Python installations do not need this
loading step or a new SciPy dependency.

Distribute a separately tagged browser wheel matching the Python and Pyodide ABI,
alongside the normal native wheels. A static website can consume that artifact
without cloning, patching or compiling FermiSimplex. This build option does not
itself publish a wheel or add FermiSimplex to the Pyodide package index.

## Automated release

CI builds and tests this wheel with Python 3.13, Pyodide 0.28.3,
pyodide-build 0.30.7 and Emscripten 4.0.9. `ci/build_pyodide.py` downloads the
matching OpenBLAS archive and verifies its SHA-256 before linking. The wheel
smoke tests run in the actual Wasm runtime under Node.js.

On a release tag, the Release workflow waits for both native and browser builds.
After native PyPI publication succeeds, it creates the corresponding GitHub
release (a prerelease for beta tags) and attaches the browser wheel automatically.
The browser artifact is kept separate from the native distributions sent to PyPI.
Manual Release runs publish native packages to TestPyPI and retain the browser
wheel as an Actions artifact, without creating a GitHub release.

For `v0.2.0b2`, the release asset is:

```text
https://github.com/Kostusas/FermiSimplex/releases/download/v0.2.0b2/fermisimplex-0.2.0b2-cp313-cp313-pyodide_2025_0_wasm32.whl
```

A website build can download this file and serve it from its own static assets,
then install it with `micropip` after loading SciPy. This avoids relying on
cross-origin access to GitHub release downloads at runtime. No source checkout,
patching, or compiler is needed in the website repository.

To reproduce the CI build, install the versions above, run
`pyodide xbuildenv install 0.28.3`, activate Emscripten 4.0.9, and run:

```sh
python ci/build_pyodide.py
npm install --prefix build/pyodide-runtime --no-package-lock --no-save pyodide@0.28.3
node ci/test_pyodide.cjs dist/pyodide/*.whl build/pyodide-runtime/node_modules/pyodide
```
