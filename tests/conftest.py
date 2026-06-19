from __future__ import annotations

import pytest

from lineartetrahedron.backend import NATIVE_AVAILABLE


requires_native = pytest.mark.skipif(
    not NATIVE_AVAILABLE,
    reason="compiled native extension is unavailable",
)
