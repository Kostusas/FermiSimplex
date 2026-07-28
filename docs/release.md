# Release process

FermiSimplex uses SemVer-style versions:

- Patch releases fix bugs and documentation.
- Minor releases add compatible APIs or examples.
- Major releases may remove or change public APIs.

Release tags use `vX.Y.Z`. The tag must match the version in `pyproject.toml`,
`pixi.toml`, the CMake projects, and `CITATION.cff`.

## Automation

GitLab is the upstream repository. Its push mirror copies branches and tags to
GitHub, and both providers independently run the same test, source-package,
wheel-smoke, and benchmark checks. GitLab covers Linux AMD64; GitHub runs the
same checks across its available Linux, macOS, and Windows runners.

The GitHub release workflow has two entry points:

- a manual run builds all distributions and publishes them to TestPyPI;
- pushing `vX.Y.Z` builds the same distributions and, after approval of the
  `pypi` environment, publishes them to PyPI.

No long-lived PyPI token is stored in the repository. The publishing jobs use
PyPI Trusted Publishing and request an identity token only after every build
has succeeded.

## One-time repository setup

1. In the GitHub mirror, create environments named `testpypi` and `pypi`.
2. Require a maintainer approval for the `pypi` environment.
3. On PyPI and TestPyPI, register GitHub trusted publishers with:
   - owner `Kostusas`;
   - repository `FermiSimplex`;
   - workflow `release.yml`;
   - the corresponding `pypi` or `testpypi` environment.
4. Protect release tags on GitLab.

## Checklist

1. Update the version in `pyproject.toml`, `pixi.toml`, the CMake projects, and
   `CITATION.cff`.
2. Move the release notes in `CHANGELOG.md` under a dated version heading.
3. Add `date-released` to `CITATION.cff`.
4. Run:

   ```bash
   pixi run --frozen ci-test
   pixi run --frozen ci-package
   pixi run --frozen ci-benchmark
   ```

5. Run the GitHub release workflow manually and verify the TestPyPI wheels in
   clean environments.
6. Create and push an annotated `vX.Y.Z` tag from the upstream GitLab
   repository.
7. Approve the GitHub `pypi` environment after all distribution jobs pass.
8. Verify the PyPI files and install FermiSimplex in a clean environment.
