"""Deterministic modular arithmetic boundaries exercised by sanitizer builds."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import DTypeLike

import sppy


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("uncompressed", [False, True])
def test_mismatched_integer_patterns_wrap_without_undefined_arithmetic(
    dtype: DTypeLike, order: str, uncompressed: bool
) -> None:
    """Integer merges must wrap overflow, including negation of unmatched minima."""
    limits = np.iinfo(dtype)
    left = np.array([[limits.max, 0, limits.min, limits.max]], dtype=dtype)
    right = np.array([[1, limits.min, 0, 2]], dtype=dtype)
    a, b = sppy.asarray(left, order=order), sppy.asarray(right, order=order)
    if uncompressed:
        a.uncompress()
        b.uncompress()
    for operation in (np.add, np.subtract, np.multiply):
        actual = {
            np.add: lambda: a + b,
            np.subtract: lambda: a - b,
            np.multiply: lambda: a * b,
        }[operation]()
        np.testing.assert_array_equal(actual.toarray(), operation(left, right))
    np.testing.assert_array_equal((a + a).toarray(), left + left)


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("unsorted", [False, True])
def test_duplicate_integer_extremes_wrap_in_triplet_and_compressed_import(
    dtype: DTypeLike, order: str, unsorted: bool
) -> None:
    """Triplet and noncanonical compressed assembly must coalesce overflow safely."""
    limits = np.iinfo(dtype)
    values = np.array([limits.max, 1, limits.min, -1], dtype=dtype)
    rows = np.array([0, 0, 1, 1], dtype=np.int32)
    columns = rows.copy()
    if unsorted:
        permutation = [2, 0, 3, 1]
        values, rows, columns = (
            values[permutation],
            rows[permutation],
            columns[permutation],
        )
    expected = np.zeros((2, 2), dtype=dtype)
    np.add.at(expected, (rows, columns), values)
    array = sppy.SparseArray.from_triplets((2, 2), rows, columns, values, order=order)
    np.testing.assert_array_equal(array.toarray(), expected)
    compressed = sppy.SparseArray.from_compressed(
        (2, 2),
        np.array([limits.max, 1, limits.min, -1], dtype=dtype),
        np.array([0, 0, 1, 1], dtype=np.int32),
        np.array([0, 2, 4], dtype=np.int32),
        order=order,
    )
    np.testing.assert_array_equal(compressed.toarray(), expected)


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("uncompressed", [False, True])
@pytest.mark.parametrize("width", [1, 2, 3, 5, 8])
def test_integer_dense_product_widths_wrap_safely(
    dtype: DTypeLike, order: str, uncompressed: bool, width: int
) -> None:
    """Generic, vector, and block products must avoid signed overflow at every width."""
    limits = np.iinfo(dtype)
    dense = np.array([[limits.min, limits.min], [limits.max, 1]], dtype=dtype)
    rhs = np.ones((2, width), dtype=dtype)
    rhs[1, -1] = 2
    array = sppy.asarray(dense, order=order)
    if uncompressed:
        array.uncompress()
    np.testing.assert_array_equal(array @ rhs, dense @ rhs)
    np.testing.assert_array_equal(array @ rhs[:, 0], dense @ rhs[:, 0])
