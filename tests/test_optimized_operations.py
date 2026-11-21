"""Numerical and storage contracts for native benchmark optimizations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import DTypeLike

import sppy
from sppy import _native
from sppy.linalg import SparseLU


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("axis", [None, 0, 1])
def test_native_reductions_preserve_promotion_and_output(
    axis: int | None, compressed: bool, dtype: np.dtype[Any], order: str
) -> None:
    """Verify native reductions preserve promotion and output."""
    dense = np.array([[0, 2, -3], [4, 0, 5]], dtype=dtype)
    if dtype.kind == "c":
        dense += 1j * dense
    array = sppy.asarray(dense, order=order)
    array[0, 0] = 0
    if not compressed:
        array.reserve(np.full(array.outer_size, 5, dtype=np.int32))
    expected = dense.sum(axis=axis, keepdims=True)
    output = np.empty_like(expected)
    actual = array.sum(axis=axis, keepdims=True, out=output)
    assert actual is output
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    # Non-default dtype keeps NumPy's full casting/reduction path.
    np.testing.assert_allclose(
        array.sum(axis=axis, dtype=np.complex128),
        dense.sum(axis=axis, dtype=np.complex128),
    )


def test_sum_int64_overflow_and_small_integer_promotion(order: str) -> None:
    """Verify sum int64 overflow and small integer promotion."""
    dense = np.array([[np.iinfo(np.int64).max, 1], [2, -3]], dtype=np.int64)
    array = sppy.asarray(dense, order=order)
    for axis in [None, 0, 1]:
        np.testing.assert_array_equal(array.sum(axis=axis), dense.sum(axis=axis))
    small = np.full((10, 10), 100, dtype=np.int8)
    actual = sppy.asarray(small, order=order).sum()
    assert actual.dtype == small.sum().dtype
    assert actual == 10000


@pytest.mark.parametrize("same_object", [False, True])
def test_equal_pattern_arithmetic_retains_zeros_and_ownership(
    same_object: bool, dtype: np.dtype[Any], order: str
) -> None:
    """Verify equal pattern arithmetic retains zeros and ownership."""
    dense = np.array([[1, 0], [2, -3]], dtype=dtype)
    left = sppy.asarray(dense, order=order)
    left[0, 1] = 0
    right = left if same_object else left.copy()
    for actual, expected in [
        (left + right, dense + dense),
        (left - right, dense - dense),
        (left.multiply(right), dense * dense),
    ]:
        np.testing.assert_array_equal(actual.toarray(), expected)
        assert actual.nnz == 4
        actual.fill(9)
        np.testing.assert_array_equal(left.toarray(), dense)
        np.testing.assert_array_equal(right.toarray(), dense)


@pytest.mark.parametrize(
    "key",
    [
        (np.array([-1, 0]), np.array([1, -1])),
        (np.array([[0], [1]]), np.array([0, 2])),
        (np.array([], dtype=np.int64), np.array([], dtype=np.int64)),
        (np.array(1), np.array(2)),
    ],
)
def test_signed_paired_gather_broadcasts_and_normalizes(
    key: Any, dtype: np.dtype[Any], order: str
) -> None:
    """Verify signed paired gather broadcasts and normalizes."""
    dense = np.arange(6).reshape(2, 3).astype(dtype)
    np.testing.assert_array_equal(sppy.asarray(dense, order=order)[key], dense[key])


@pytest.mark.parametrize("bad", [-(2**63), 2**63 - 1, -3, 2])
def test_signed_paired_gather_rejects_wide_invalid_indices(
    bad: Any, order: str
) -> None:
    """Verify signed paired gather rejects wide invalid indices."""
    array = sppy.eye(2, order=order)
    with pytest.raises(ValueError, match="range"):
        array[np.array([bad], np.int64), np.array([0])]


@pytest.mark.parametrize("pointers", [[0, 0, 2, 2, 3, 3], [0, 1, 1, 1, 3, 3]])
@pytest.mark.parametrize("canonical", [False, True])
def test_compressed_ordering_ignores_repeated_empty_boundaries(
    pointers: Any, canonical: bool, order: str
) -> None:
    """Verify compressed ordering ignores repeated empty boundaries."""
    indices = np.array([2, 0, 1] if canonical else [2, 0, 0], np.int32)
    if canonical:
        for begin, end in zip(pointers[:-1], pointers[1:]):
            indices[begin:end].sort()
    values = np.array([1.0, 2.0, 3.0])
    shape = (5, 3) if order == "C" else (3, 5)
    expected = np.zeros(shape)
    for outer, (begin, end) in enumerate(zip(pointers[:-1], pointers[1:])):
        for k in range(begin, end):
            position = (outer, indices[k]) if order == "C" else (indices[k], outer)
            expected[position] += values[k]
    actual = sppy.SparseArray.from_compressed(
        shape, values, indices, np.array(pointers, np.int32), order=order
    )
    np.testing.assert_array_equal(actual.toarray(), expected)
    _, inner, ptr = actual.compressed_arrays()
    for begin, end in zip(ptr[:-1], ptr[1:]):
        assert np.all(np.diff(inner[begin:end]) > 0)


@pytest.mark.parametrize("columns", [1, 4, 8, 17])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_lu_batched_solve_uses_refactorized_storage(
    columns: int, dtype: np.dtype[Any], order: str
) -> None:
    """Verify lu batched solve uses refactorized storage."""
    dense = np.array([[0, 2, 1], [3, 1, 0], [1, 0, 4]], dtype=dtype)
    if np.dtype(dtype).kind == "c":
        dense += 0.5j * dense
    array = sppy.asarray(dense, order=order)
    array.reserve(np.full(array.outer_size, 4, np.int32))
    rhs = np.arange(3 * columns, dtype=dtype).reshape(3, columns) + 1
    factor = SparseLU(array)
    for scale in [1, 2, 3]:
        if scale > 1:
            factor.factorize(array * scale)
        actual = factor.solve(rhs)
        np.testing.assert_allclose(dense * scale @ actual, rhs, rtol=2e-5, atol=2e-5)
        actual[:] = 0
        np.testing.assert_allclose(
            dense * scale @ factor.solve(rhs), rhs, rtol=2e-5, atol=2e-5
        )
    array.fill(0)
    np.testing.assert_allclose(dense * 3 @ factor.solve(rhs), rhs, rtol=2e-5, atol=2e-5)


@pytest.mark.native
def test_native_solver_storage_rejects_incompatible_dtype() -> None:
    """Verify native solver storage rejects incompatible dtype."""
    factor = _native.Factor_float64("lu", "colamd", 0.0, 1.0, -1.0)
    with pytest.raises(TypeError, match="scalar type"):
        factor.compute_matrix(sppy.eye(2, dtype=np.complex128)._array)


def test_compressed_validation_and_triplets_match_duplicate_scatter(order: str) -> None:
    """Verify compressed validation and triplets match duplicate scatter."""
    rng = np.random.default_rng(42)
    for _ in range(40):
        rows, columns = map(int, rng.integers(0, 8, size=2))
        count = 25 if rows and columns else 0
        ri = (
            rng.integers(rows, size=count, dtype=np.int32)
            if rows
            else np.array([], np.int32)
        )
        ci = (
            rng.integers(columns, size=count, dtype=np.int32)
            if columns
            else np.array([], np.int32)
        )
        values = rng.integers(-3, 4, size=count).astype(np.float64)
        expected = np.zeros((rows, columns))
        np.add.at(expected, (ri, ci), values)
        assembled = sppy.SparseArray.from_triplets(
            (rows, columns), ri, ci, values, order=order
        )
        np.testing.assert_array_equal(assembled.toarray(), expected)
        outer = ri if order == "C" else ci
        inner = ci if order == "C" else ri
        permutation = np.argsort(outer, kind="stable")
        size = rows if order == "C" else columns
        indptr = np.r_[0, np.bincount(outer, minlength=size).cumsum()].astype(np.int32)
        imported = sppy.SparseArray.from_compressed(
            (rows, columns),
            values[permutation],
            inner[permutation],
            indptr,
            order=order,
        )
        np.testing.assert_array_equal(imported.toarray(), expected)


@pytest.mark.parametrize("columns", [1, 8])
@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize(
    "value_dtype", [np.float32, np.float64, np.complex64, np.complex128]
)
def test_product_kernels_handle_rectangles_and_sparse_gaps(
    columns: int, compressed: bool, value_dtype: DTypeLike, order: str
) -> None:
    """Verify product kernels handle rectangles and sparse gaps."""
    rng = np.random.default_rng(3)
    dense = rng.normal(size=(7, 11)).astype(value_dtype)
    dense[rng.random(dense.shape) < 0.6] = 0
    rhs = rng.normal(size=(11, columns)).astype(value_dtype)
    if np.dtype(value_dtype).kind == "c":
        dense += 0.3j * dense
        rhs += 0.7j * rhs
    array = sppy.asarray(dense, order=order)
    if not compressed:
        array.reserve(np.full(array.outer_size, 12, np.int32))
    np.testing.assert_allclose(array @ rhs, dense @ rhs, rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("columns", [1, 8])
def test_complex_nonfinite_product_uses_ieee_fallback(columns: int, order: str) -> None:
    """Verify complex nonfinite product uses ieee fallback."""
    from scipy.sparse import csr_array

    dense = np.array([[complex(np.inf, 0), 2 + 3j], [1 + 2j, 4 - 2j]])
    rhs = np.full((2, columns), 1 + 2j)
    array = sppy.asarray(dense, order=order)
    with np.errstate(invalid="ignore"):
        expected = csr_array(dense) @ rhs
        actual = array @ rhs
    np.testing.assert_allclose(actual, expected, equal_nan=True)


@pytest.mark.parametrize("size", [0, 1, 7, 8, 9, 71])
@pytest.mark.parametrize("value_dtype", [np.float32, np.float64])
def test_native_norm_handles_accumulator_tail_and_nonfinite(
    size: int, value_dtype: DTypeLike, order: str
) -> None:
    """Verify native norm handles accumulator tail and nonfinite."""
    from sppy.linalg import norm

    dense = np.arange(1, size + 1, dtype=value_dtype).reshape(1, size)
    array = sppy.asarray(dense, order=order)
    np.testing.assert_allclose(norm(array), np.linalg.norm(dense), rtol=1e-6)
    if size:
        array[0, size - 1] = np.inf
        assert np.isinf(norm(array))
        array[0, size - 1] = np.nan
        assert np.isnan(norm(array))


def test_complex_zero_pruning_preserves_nonfinite_and_source(order: str) -> None:
    """Verify complex zero pruning preserves nonfinite and source."""
    array = sppy.asarray(
        [[0j, complex(np.nan, 0)], [complex(np.inf, 0), 2j]], order=order
    )
    array[0, 0] = -0j
    actual = array.copy()
    actual.eliminate_zeros()
    assert actual.nnz == 3
    assert array.nnz == 4
    assert np.isnan(actual[0, 1].real)
    assert np.isinf(actual[1, 0].real)
    assert actual[1, 1] == 2j


@pytest.mark.parametrize("value_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("compressed", [False, True])
def test_large_rectangular_product_scatter_keeps_canonical_structural_zeros(
    value_dtype: DTypeLike, compressed: bool, order: str
) -> None:
    """Verify large rectangular product scatter keeps canonical structural zeros."""
    from scipy.sparse import csr_array

    rng = np.random.default_rng(17)
    rows, inner, columns = 2048, 300, 2051
    ri = rng.integers(rows, size=rows * 3, dtype=np.int32)
    ci = rng.integers(inner, size=rows * 3, dtype=np.int32)
    rj = rng.integers(inner, size=columns * 3, dtype=np.int32)
    cj = rng.integers(columns, size=columns * 3, dtype=np.int32)
    values_left = rng.integers(-2, 3, size=ri.size).astype(value_dtype)
    values_right = rng.integers(-2, 3, size=rj.size).astype(value_dtype)
    left = sppy.SparseArray.from_triplets(
        (rows, inner), ri, ci, values_left, order=order
    )
    right = sppy.SparseArray.from_triplets(
        (inner, columns), rj, cj, values_right, order=order
    )
    reference_left = csr_array((values_left, (ri, ci)), shape=left.shape)
    reference_right = csr_array((values_right, (rj, cj)), shape=right.shape)
    if not compressed:
        left.reserve(np.full(left.outer_size, 32, np.int32))
        right.reserve(np.full(right.outer_size, 32, np.int32))
    result = left @ right
    difference = result.to_scipy_csr() - reference_left @ reference_right
    assert not np.any(difference.data)
    reference_left.data[:] = 1
    reference_right.data[:] = 1
    assert result.nnz == (reference_left @ reference_right).nnz
    assert result.order == order
    assert result.is_compressed
    _, indices, indptr = result.compressed_arrays()
    for begin, end in zip(indptr[:-1], indptr[1:]):
        assert np.all(np.diff(indices[begin:end]) > 0)


@pytest.mark.parametrize(
    "value_dtype", [np.float32, np.float64, np.complex64, np.complex128]
)
@pytest.mark.parametrize("columns", [4, 8, 17])
def test_supernodal_block_solve_handles_sparse_pivoted_systems(
    value_dtype: DTypeLike, columns: int, order: str
) -> None:
    """Verify supernodal block solve handles sparse pivoted systems."""
    rng = np.random.default_rng(61)
    size = 73
    dense = rng.normal(size=(size, size)).astype(value_dtype)
    dense[rng.random(dense.shape) < 0.92] = 0
    dense += 5 * np.eye(size, dtype=value_dtype)
    if np.dtype(value_dtype).kind == "c":
        dense += 0.4j * dense
    dense = dense[rng.permutation(size)]
    rhs = rng.normal(size=(size, columns)).astype(value_dtype)[:, ::-1]
    factor = SparseLU(sppy.asarray(dense, order=order))
    result = factor.solve(rhs)
    np.testing.assert_allclose(dense @ result, rhs, rtol=5e-5, atol=5e-5)


@pytest.mark.parametrize("corruption", [None, "duplicates", "unsorted"])
@pytest.mark.parametrize("value_dtype", [np.float64, np.complex128])
def test_large_run_ordered_triplets_validate_every_outer_vector(
    corruption: str, value_dtype: DTypeLike, order: str
) -> None:
    """Verify large run ordered triplets validate every outer vector."""
    from scipy.sparse import csr_array

    size = 5000
    rows, columns = [], []
    for offset in range(-2, 3):
        r = np.arange(max(0, -offset), min(size, size - offset), dtype=np.int32)
        rows.append(r)
        columns.append(r + offset)
    rows, columns = np.concatenate(rows), np.concatenate(columns)
    values = (np.arange(rows.size) % 5 - 2).astype(value_dtype)
    if np.dtype(value_dtype).kind == "c":
        values += 1j * values
    if corruption == "duplicates":
        rows = np.r_[rows, np.int32(0), np.int32(0)]
        columns = np.r_[columns, np.int32(0), np.int32(0)]
        values = np.r_[values, value_dtype(2), value_dtype(-2)]
    elif corruption == "unsorted":
        positions = np.flatnonzero(rows == 0)
        permutation = np.arange(rows.size)
        permutation[positions] = positions[::-1]
        rows, columns, values = (
            rows[permutation],
            columns[permutation],
            values[permutation],
        )
    expected = csr_array((values, (rows, columns)), shape=(size, size))
    actual = sppy.SparseArray.from_triplets(
        (size, size), rows, columns, values, order=order
    )
    difference = actual.to_scipy_csr() - expected
    assert not np.any(difference.data)
    assert actual.nnz == expected.nnz
    _, indices, indptr = actual.compressed_arrays()
    for begin, end in zip(indptr[:-1], indptr[1:]):
        assert np.all(np.diff(indices[begin:end]) > 0)
