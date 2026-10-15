"""Check cross-build dependency selection without changing native defaults."""
from pathlib import Path

import pytest
from scikit_build_core.build import get_requires_for_build_wheel
from scikit_build_core.settings.skbuild_read_settings import SettingsReader


@pytest.mark.parametrize("enabled", [None, "0", "1"])
def test_pyodide_is_an_opt_in_build(monkeypatch, enabled):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    monkeypatch.delenv("FERMISIMPLEX_PYODIDE", raising=False)
    if enabled is not None:
        monkeypatch.setenv("FERMISIMPLEX_PYODIDE", enabled)
    requirements = get_requires_for_build_wheel()
    assert ("scipy-openblas32==0.3.34.0.0" in requirements) == (enabled != "1")
    settings = SettingsReader.from_file("pyproject.toml", state="wheel").settings
    assert settings.cmake.define["FERMISIMPLEX_BUILD_TESTING"] == "OFF"
    assert (settings.cmake.define.get("FERMISIMPLEX_PYODIDE") == "ON") == (enabled == "1")
