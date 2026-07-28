from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = [
    ROOT / ".gitlab-ci.yml",
    *sorted((ROOT / ".github" / "workflows").glob("*.yml")),
]

for path in WORKFLOWS:
    parsed = yaml.safe_load(path.read_text())
    if not isinstance(parsed, dict):
        raise SystemExit(f"{path.relative_to(ROOT)} is not a YAML mapping")
    print(f"parsed {path.relative_to(ROOT)}")
