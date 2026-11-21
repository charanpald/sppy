"""Matrix Market interoperability uses isolated streams and temporary paths."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import sppy
from sppy.io import mmread, mmwrite


@pytest.mark.parametrize("dtype", [np.float64, np.int64, np.complex128])
@pytest.mark.parametrize("use_stream", [False, True])
def test_roundtrip(
    dtype: np.dtype[Any], use_stream: bool, order: str, tmp_path: Path
) -> None:
    # Arrange
    """Verify roundtrip."""
    dense = np.array([[0, 2], [-3, 0]], dtype=dtype)
    array = sppy.asarray(dense)
    destination = io.BytesIO() if use_stream else tmp_path / "matrix.mtx"

    # Act
    mmwrite(destination, array)
    if use_stream:
        destination.seek(0)
    result = mmread(destination, order=order)

    # Assert
    assert result.dtype == dense.dtype
    assert result.order == order
    np.testing.assert_array_equal(result.toarray(), dense)


def test_roundtrip_preserves_float_precision() -> None:
    # Arrange
    """Verify roundtrip preserves float precision."""
    dense = np.array([[0.0, np.pi], [1e-15, 0.0]])
    array = sppy.asarray(dense)
    stream = io.BytesIO()

    # Act
    mmwrite(stream, array)
    stream.seek(0)
    result = mmread(stream)

    # Assert
    np.testing.assert_array_equal(result.toarray(), dense)


def test_read_symmetric_file() -> None:
    # Arrange
    """Verify read symmetric file."""
    source = io.BytesIO(
        b"%%MatrixMarket matrix coordinate real symmetric\n% comment\n2 2 2\n1 1 2\n2 1 3\n"
    )
    expected = np.array([[2, 3], [3, 0]])

    # Act
    result = mmread(source)

    # Assert
    np.testing.assert_array_equal(result.toarray(), expected)


def test_read_complex_values() -> None:
    # Arrange
    """Verify read complex values."""
    source = io.BytesIO(
        b"%%MatrixMarket matrix coordinate complex general\n1 1 1\n1 1 2 3\n"
    )

    # Act
    result = mmread(source)

    # Assert
    np.testing.assert_array_equal(result.toarray(), [[2 + 3j]])


def test_write_rejects_vector(tmp_path: Path) -> None:
    # Arrange
    """Verify write rejects vector."""
    array = sppy.ones(3)

    # Act / Assert
    with pytest.raises(ValueError, match="two-dimensional"):
        mmwrite(tmp_path / "vector.mtx", array)
