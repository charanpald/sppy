"""NumPy-style sparse arrays backed by Eigen."""

from sppy.sparse_array import SparseArray
from sppy._native import get_num_threads, set_num_threads, openmp_enabled
from sppy.util import asarray, diag, eye, rand, ones, zeros
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


def test(*pytest_args):
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
