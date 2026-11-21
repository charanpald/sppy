"""NumPy-style sparse arrays backed by Eigen."""

from __future__ import annotations

from sppy import _native
from sppy.sparse_array import SparseArray
from sppy.util import asarray, diag, eye, ones, rand, zeros
from sppy.version import __version__

__all__ = [
    "SparseArray",
    "asarray",
    "get_num_threads",
    "set_num_threads",
    "openmp_enabled",
    "diag",
    "eye",
    "rand",
    "ones",
    "zeros",
    "__version__",
    "test",
]

openmp_enabled: bool = _native.openmp_enabled


def get_num_threads() -> int:
    """Return Eigen's configured numerical thread count."""
    return _native.get_num_threads()


def set_num_threads(n: int) -> None:
    """Set Eigen's positive thread count; parallel kernels require OpenMP support."""
    _native.set_num_threads(n)


def test(*pytest_args: str) -> int:
    """Run pytest from a source checkout and return its exit status.

    Tests are development resources in the top-level tests/ directory; wheels
    do not bundle them. Pass pytest arguments to select or configure tests.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    tests = root / "tests"
    if not (root / "pyproject.toml").is_file() or not (tests / "conftest.py").is_file():
        raise RuntimeError(
            "Tests are available in the source checkout. Run 'uv run pytest' from the project root."
        )

    import pytest

    return pytest.main([str(tests), *pytest_args])
