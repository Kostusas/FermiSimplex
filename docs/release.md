# Release process

FermiSimplex uses SemVer-style versions:

- Patch releases fix bugs and documentation.
- Before 1.0, minor releases may change public APIs while the numerical
  foundations are still stabilizing.
- After 1.0, minor releases add compatible APIs or examples.
- Major releases may remove or change public APIs.

Release tags match the Python package version, including any pre-release suffix:
for example, `v0.2.0b1` for a beta or `v0.2.0rc1` for a release candidate.
`pyproject.toml`, `pixi.toml` and `CITATION.cff` contain that full version.
The CMake projects use its numeric base, `0.2.0`, because CMake does not accept
Python pre-release suffixes. `ci/check_metadata.py` checks both forms.

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

Branches named `codex/beta-*` run the full Python 3.11–3.13 wheel matrix in CI
without uploading packages. Use these checks when preparing a beta; neither a
release tag nor the manual publishing workflow is needed for build verification.

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

1. Update the full version in `pyproject.toml`, `pixi.toml` and `CITATION.cff`.
   Update the CMake projects only when the numeric base version changes.
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

## 0.2.0b2 browser-enabled beta

The version metadata and release notes are prepared for `v0.2.0b2`. Merge the
release preparation branch into main, wait for CI, and tag that merge commit:

```sh
git tag -a v0.2.0b2 -m "FermiSimplex 0.2.0b2"
git push origin v0.2.0b2
```

The mirrored GitHub Release workflow builds the native distributions and a
Pyodide 0.28.3 wheel. Approve the existing `pypi` environment when requested.
After the native upload, the workflow attaches the browser wheel to the GitHub
release automatically. No manual build or wheel upload is needed.

Verify PyPI and the browser release asset before tagging MeanFi `v2.0.0b2`.
See [WebAssembly](webassembly.md) for the asset URL and runtime requirements.
