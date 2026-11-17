"""NumPy-style sparse arrays backed by Eigen."""

from sppy.csarray import SparseArray, csarray
from sppy.util import asarray, diag, eye, rand, ones, zeros
from sppy.version import __version__

__all__ = [
    "SparseArray",
    "asarray",
    "csarray",
    "diag",
    "eye",
    "rand",
    "ones",
    "zeros",
    "__version__",
    "test",
]


def test():
    """Run the installed regression suite and return pytest's exit status."""
    from pathlib import Path

    import pytest

    return pytest.main([str(Path(__file__).parent)])
