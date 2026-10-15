"""Build against the OpenBLAS provider shipped with Pyodide 0.28.3."""
from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
import subprocess
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
URL = "https://cdn.jsdelivr.net/pyodide/v0.28.3/full/libopenblas-0.3.26.zip"
SHA256 = "15c2c82b0fdfdde83a81190f35f5f4a0d15bf236a7e93944c433ae5ac6bf3ee5"


def main():
    target = ROOT / "build" / "pyodide-libs" / "libopenblas.so"
    target.parent.mkdir(parents=True, exist_ok=True)
    archive = urllib.request.urlopen(URL, timeout=120).read()
    if hashlib.sha256(archive).hexdigest() != SHA256:
        raise SystemExit("Pyodide OpenBLAS archive checksum mismatch")
    with zipfile.ZipFile(io.BytesIO(archive)) as wheel:
        libraries = [name for name in wheel.namelist() if Path(name).name == target.name]
        if len(libraries) != 1:
            raise SystemExit("Expected one Pyodide OpenBLAS library")
        target.write_bytes(wheel.read(libraries[0]))
    output = ROOT / "dist" / "pyodide"
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["pyodide", "build", str(ROOT),
         "--outdir", str(output),
         f"-Ccmake.define.FERMISIMPLEX_PYODIDE_OPENBLAS={target}"],
        env={**os.environ, "FERMISIMPLEX_PYODIDE": "1"},
        check=True,
    )


if __name__ == "__main__":
    main()
