"""Contracts at the actual pybind11/Eigen boundary, without a second compiler."""

import numpy as np
import pytest
from sppy import _native


@pytest.mark.native
@pytest.mark.parametrize(
    "name",
    [
        "int8",
        "int16",
        "int32",
        "int64",
        "float32",
        "float64",
        "complex64",
        "complex128",
    ],
)
@pytest.mark.parametrize("order", ["C", "F"])
def test_native_assembly_and_compressed_storage(name, order):
    matrix = getattr(_native, f"{name}_{order}")(2, 3)
    matrix.assemble(
        np.array([0, 0, 1], np.int32),
        np.array([1, 1, 2], np.int32),
        np.array([2, 3, 4], dtype=name),
    )
    assert matrix.nnz == 2
    assert matrix.coeff(0, 1) == 5
    values, indices, indptr = matrix.compressed_arrays()
    assert values.dtype == np.dtype(name)
    assert indptr[0] == 0 and indptr[-1] == 2
    assert (np.diff(indptr) >= 0).all()
    assert (indices >= 0).all()


@pytest.mark.native
@pytest.mark.parametrize(
    "rows,columns,values",
    [
        ([3], [0], [1.0]),
        ([0], [-1], [1.0]),
        ([0, 1], [0], [1.0]),
        ([0], [0], [1.0, 2.0]),
    ],
)
def test_native_boundary_rejects_invalid_coordinates(rows, columns, values):
    matrix = _native.float64_F(2, 2)
    with pytest.raises(ValueError):
        matrix.assemble(
            np.asarray(rows, np.int32),
            np.asarray(columns, np.int32),
            np.asarray(values),
        )
