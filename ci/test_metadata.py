from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]


@pytest.fixture
def release_tree(tmp_path):
    for relative in (
        "ci/check_metadata.py",
        "pyproject.toml",
        "pixi.toml",
        "CMakeLists.txt",
        "cpp/CMakeLists.txt",
        "python/CMakeLists.txt",
        "CITATION.cff",
        "CHANGELOG.md",
        "LICENSE",
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    return tmp_path


def check(tree, tag=""):
    return subprocess.run(
        [sys.executable, str(tree / "ci/check_metadata.py")],
        env={**os.environ, "FERMISIMPLEX_RELEASE_TAG": tag},
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("version", ["0.2.0", "0.2.0b1", "0.2.0rc1"])
def test_release_version_accepts_numeric_cmake_base(release_tree, version):
    for relative in ("pyproject.toml", "pixi.toml", "CITATION.cff", "CHANGELOG.md"):
        path = release_tree / relative
        path.write_text(path.read_text().replace(VERSION, version))

    result = check(release_tree, f"v{version}")
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "relative", ["CMakeLists.txt", "cpp/CMakeLists.txt", "python/CMakeLists.txt"]
)
def test_release_rejects_wrong_cmake_base(release_tree, relative):
    path = release_tree / relative
    path.write_text(path.read_text().replace("VERSION 0.2.0", "VERSION 0.3.0"))

    result = check(release_tree)
    assert result.returncode != 0
    assert "CMake versions must equal 0.2.0" in result.stderr


@pytest.mark.parametrize("relative", ["pixi.toml", "CITATION.cff"])
def test_release_rejects_mismatched_python_version(release_tree, relative):
    path = release_tree / relative
    path.write_text(path.read_text().replace(VERSION, "9.9.9"))

    result = check(release_tree)
    assert result.returncode != 0
    assert "versions differ" in result.stderr


def test_release_rejects_wrong_tag(release_tree):
    result = check(release_tree, "v0.2.0")
    assert result.returncode != 0
    assert f"must equal v{VERSION}" in result.stderr


def test_release_rejects_invalid_python_version(release_tree):
    path = release_tree / "pyproject.toml"
    path.write_text(
        re.sub(r'version = "[^"]+"', 'version = "invalid"', path.read_text(), count=1)
    )

    result = check(release_tree)
    assert result.returncode != 0
    assert "invalid Python package version" in result.stderr
