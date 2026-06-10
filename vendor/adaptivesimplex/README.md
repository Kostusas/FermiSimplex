# Vendored AdaptiveSimplex

This is a source-only copy of the header-only AdaptiveSimplex CMake package,
vendored so `lineartetrahedron` can build from a plain Python package install
without requiring users to install a separate CMake package first.

Provenance:

- repository: https://gitlab.kwant-project.org/qt/adaptivesimplex.git
- branch: full-refactor
- revision: c6eaa7ab251aa1b7c6be084e0c09d848282c1fe8

If a system `adaptivesimplex` CMake package is discoverable,
`lineartetrahedron` uses that package instead of this vendored fallback.
