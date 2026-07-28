from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = [
    ROOT / ".gitlab-ci.yml",
    *sorted((ROOT / ".github" / "workflows").glob("*.yml")),
]
PIXI_VERSION = "v0.70.2"
SHARED_TASKS = ("ci-test", "ci-package", "ci-benchmark")

for path in WORKFLOWS:
    parsed = yaml.safe_load(path.read_text())
    if not isinstance(parsed, dict):
        raise SystemExit(f"{path.relative_to(ROOT)} is not a YAML mapping")
    print(f"parsed {path.relative_to(ROOT)}")

gitlab = (ROOT / ".gitlab-ci.yml").read_text()
github = "\n".join(
    path.read_text() for path in sorted((ROOT / ".github" / "workflows").glob("*.yml"))
)

for task in SHARED_TASKS:
    command = f"pixi run --frozen {task}"
    for provider, workflows in (("GitLab", gitlab), ("GitHub", github)):
        if command not in workflows:
            raise SystemExit(f"{provider} CI does not run shared task: {command}")

if f"/download/{PIXI_VERSION}/" not in gitlab:
    raise SystemExit(f"GitLab CI does not install Pixi {PIXI_VERSION}")

github_pixi_versions = [
    line.split(":", 1)[1].strip()
    for line in github.splitlines()
    if line.strip().startswith("pixi-version:")
]
if not github_pixi_versions or any(
    version != PIXI_VERSION for version in github_pixi_versions
):
    raise SystemExit(
        f"every GitHub setup-pixi step must use pixi-version: {PIXI_VERSION}"
    )

if "cp313-*" not in github or "cibuildwheel" not in github:
    raise SystemExit("GitHub CI does not run the CPython 3.13 wheel smoke build")

gitlab_wheel_markers = (
    "/opt/python/cp313-cp313/bin/python",
    "pip wheel",
    "auditwheel repair",
    "pytest ci/test_wheel.py",
)
if any(marker not in gitlab for marker in gitlab_wheel_markers):
    raise SystemExit("GitLab CI does not run the CPython 3.13 wheel smoke build")

print("GitLab and GitHub CI share equivalent test, package, benchmark, and wheel checks")
