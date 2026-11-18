"""Matrix Market interoperability uses isolated streams and temporary paths."""

import io

import numpy as np
import pytest

import sppy
from sppy.io import mmread, mmwrite


@pytest.mark.parametrize("dtype", [np.float64, np.int64, np.complex128])
@pytest.mark.parametrize("use_stream", [False, True])
def test_roundtrip(dtype, use_stream, order, tmp_path):
    # Arrange
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


def test_roundtrip_preserves_float_precision():
    # Arrange
    dense = np.array([[0.0, np.pi], [1e-15, 0.0]])
    array = sppy.asarray(dense)
    stream = io.BytesIO()

    # Act
    mmwrite(stream, array)
    stream.seek(0)
    result = mmread(stream)

    # Assert
    np.testing.assert_array_equal(result.toarray(), dense)


def test_read_symmetric_file():
    # Arrange
    source = io.BytesIO(
        b"%%MatrixMarket matrix coordinate real symmetric\n% comment\n2 2 2\n1 1 2\n2 1 3\n"
    )
    expected = np.array([[2, 3], [3, 0]])

    # Act
    result = mmread(source)

    # Assert
    np.testing.assert_array_equal(result.toarray(), expected)


def test_read_complex_values():
    # Arrange
    source = io.BytesIO(
        b"%%MatrixMarket matrix coordinate complex general\n1 1 1\n1 1 2 3\n"
    )

    # Act
    result = mmread(source)

    # Assert
    np.testing.assert_array_equal(result.toarray(), [[2 + 3j]])


def test_write_rejects_vector(tmp_path):
    # Arrange
    array = sppy.ones(3)

    # Act / Assert
    with pytest.raises(ValueError, match="two-dimensional"):
        mmwrite(tmp_path / "vector.mtx", array)
