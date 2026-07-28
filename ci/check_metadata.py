from __future__ import annotations

import os
from pathlib import Path
import re
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_AUTHOR = "Kostas Vilkelis"
EXPECTED_LICENSE = "BSD-3-Clause"
VERSION_PATTERN = r"[0-9]+\.[0-9]+\.[0-9]+(?:[A-Za-z0-9.-]+)?"


def fail(message: str) -> None:
    print(f"metadata check failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def one_match(pattern: str, text: str, source: str) -> str:
    matches = re.findall(pattern, text, flags=re.MULTILINE | re.DOTALL)
    if len(matches) != 1:
        fail(f"expected one {pattern!r} match in {source}, found {len(matches)}")
    return matches[0]


with (ROOT / "pyproject.toml").open("rb") as file:
    pyproject = tomllib.load(file)
with (ROOT / "pixi.toml").open("rb") as file:
    pixi = tomllib.load(file)

project = pyproject["project"]
version = project["version"]
versions = {
    "pyproject.toml": version,
    "pixi.toml": pixi["package"]["version"],
}
for relative in ("CMakeLists.txt", "cpp/CMakeLists.txt", "python/CMakeLists.txt"):
    text = (ROOT / relative).read_text()
    versions[relative] = one_match(
        rf"\bproject\s*\([^)]*?\bVERSION\s+({VERSION_PATTERN})",
        text,
        relative,
    )

citation = (ROOT / "CITATION.cff").read_text()
versions["CITATION.cff"] = one_match(
    rf'^version:\s*["\']?({VERSION_PATTERN})["\']?\s*$',
    citation,
    "CITATION.cff",
)

if set(versions.values()) != {version}:
    fail("versions differ: " + ", ".join(f"{path}={value}" for path, value in versions.items()))
if project["name"].lower() != "fermisimplex":
    fail("the distribution name must normalize to fermisimplex")
if project.get("authors") != [{"name": EXPECTED_AUTHOR}]:
    fail(f"pyproject author must be {EXPECTED_AUTHOR!r}")
if project.get("license") != EXPECTED_LICENSE:
    fail(f"pyproject license must be {EXPECTED_LICENSE}")
if f'given-names: "Kostas"' not in citation or f'family-names: "Vilkelis"' not in citation:
    fail("CITATION.cff author does not match AdaptiveSimplex")
if f'license: "{EXPECTED_LICENSE}"' not in citation:
    fail("CITATION.cff license does not match AdaptiveSimplex")
if f"Copyright (c) 2026, {EXPECTED_AUTHOR}" not in (ROOT / "LICENSE").read_text():
    fail("LICENSE copyright does not match AdaptiveSimplex")

release_tag = os.environ.get("FERMISIMPLEX_RELEASE_TAG", "")
if release_tag:
    if release_tag != f"v{version}":
        fail(f"release tag {release_tag!r} must equal v{version}")
    released = one_match(
        r'^date-released:\s*["\']?([0-9]{4}-[0-9]{2}-[0-9]{2})["\']?\s*$',
        citation,
        "CITATION.cff",
    )
    changelog = (ROOT / "CHANGELOG.md").read_text()
    if f"## {version} - {released}" not in changelog:
        fail("CHANGELOG.md needs a dated heading matching CITATION.cff")

print(f"FermiSimplex metadata is consistent at version {version}")
