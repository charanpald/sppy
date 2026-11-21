"""Generated sparse numerical contracts checked against NumPy and SciPy."""

from __future__ import annotations

from typing import Any

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.typing import NDArray
from scipy import sparse

import sppy
from sppy.linalg import SparseLU, cg

DTYPES = (
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
)


@st.composite
def matrices(
    draw: st.DrawFn,
    shape: tuple[int, int] | None = None,
    integers: bool = False,
    dtype: np.dtype[Any] | None = None,
) -> NDArray[Any]:
    """Generate small matrices with zeros, signed endpoints, and varied dtypes."""
    dtype = (
        np.dtype(draw(st.sampled_from(DTYPES[:4] if integers else DTYPES)))
        if dtype is None
        else dtype
    )
    if shape is None:
        shape = draw(st.tuples(st.integers(0, 5), st.integers(0, 5)))
    if dtype.kind == "i":
        limits = np.iinfo(dtype)
        elements = st.one_of(
            st.integers(-10, 10), st.sampled_from([int(limits.min), int(limits.max)])
        )
    elif dtype.kind == "c":
        elements = st.tuples(st.integers(-5, 5), st.integers(-5, 5)).map(
            lambda pair: complex(*pair)
        )
    else:
        elements = st.integers(-10, 10).map(float)
    return draw(arrays(dtype, shape, elements=elements))


@settings(max_examples=100, deadline=None, derandomize=True)
@given(data=st.data(), order=st.sampled_from(["C", "F"]), uncompressed=st.booleans())
def test_generated_coefficient_arithmetic(
    data: st.DataObject, order: str, uncompressed: bool
) -> None:
    """Compare equal-shape coefficient operations with dense modular arithmetic."""
    left = data.draw(matrices())
    right = data.draw(matrices(shape=left.shape, dtype=left.dtype))
    a, b = sppy.asarray(left, order=order), sppy.asarray(right, order=order)
    if a.size:
        a.put(0, [0], [0])
        left[0, 0] = 0
    if uncompressed:
        a.uncompress()
        b.uncompress()
    for actual, expected in (
        (a + b, left + right),
        (a - b, left - right),
        (a * b, left * right),
    ):
        np.testing.assert_array_equal(actual.toarray(), expected)
        assert actual.dtype == left.dtype
        assert actual.order == order


@settings(max_examples=100, deadline=None, derandomize=True)
@given(data=st.data(), order=st.sampled_from(["C", "F"]))
def test_generated_products(data: st.DataObject, order: str) -> None:
    """Compare rectangular sparse/dense products across dimensions and dtypes."""
    left = data.draw(matrices())
    width = data.draw(st.integers(0, 5))
    right = data.draw(matrices(shape=(left.shape[1], width), dtype=left.dtype))
    a, b = sppy.asarray(left, order=order), sppy.asarray(right, order=order)
    expected = left @ right
    np.testing.assert_array_equal((a @ b).toarray(), expected)
    np.testing.assert_array_equal(a @ right, expected)


@settings(max_examples=80, deadline=None, derandomize=True)
@given(matrix=matrices(), order=st.sampled_from(["C", "F"]))
def test_generated_reductions(matrix: NDArray[Any], order: str) -> None:
    """Compare sums and counts while preserving NumPy's integer accumulator dtype."""
    array = sppy.asarray(matrix, order=order)
    for axis in (None, 0, 1, -1):
        np.testing.assert_allclose(
            array.sum(axis=axis), matrix.sum(axis=axis), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_array_equal(
            array.count_nonzero(axis=axis), np.count_nonzero(matrix, axis=axis)
        )


@settings(max_examples=100, deadline=None, derandomize=True)
@given(data=st.data(), order=st.sampled_from(["C", "F"]), reverse=st.booleans())
def test_generated_duplicate_assembly(
    data: st.DataObject, order: str, reverse: bool
) -> None:
    """Check duplicate summation and compressed import against NumPy and SciPy."""
    shape = data.draw(st.tuples(st.integers(1, 5), st.integers(1, 5)))
    values = data.draw(
        matrices(shape=(1, data.draw(st.integers(0, 30))), integers=True)
    ).ravel()
    rows = np.array(
        data.draw(
            st.lists(
                st.integers(0, shape[0] - 1), min_size=len(values), max_size=len(values)
            )
        ),
        dtype=np.int32,
    )
    columns = np.array(
        data.draw(
            st.lists(
                st.integers(0, shape[1] - 1), min_size=len(values), max_size=len(values)
            )
        ),
        dtype=np.int32,
    )
    permutation = np.lexsort((columns, rows))
    if reverse:
        permutation = permutation[::-1]
    rows, columns, values = rows[permutation], columns[permutation], values[permutation]
    expected = np.zeros(shape, dtype=values.dtype)
    np.add.at(expected, (rows, columns), values)
    array = sppy.SparseArray.from_triplets(shape, rows, columns, values, order=order)
    np.testing.assert_array_equal(array.toarray(), expected)
    reference = sparse.coo_array((values, (rows, columns)), shape=shape)
    np.testing.assert_array_equal(array.toarray(), reference.toarray())
    data_values, indices, indptr = array.compressed_arrays()
    restored = sppy.SparseArray.from_compressed(
        shape, data_values, indices, indptr, order=order
    )
    np.testing.assert_array_equal(restored.toarray(), expected)


@settings(max_examples=80, deadline=None, derandomize=True)
@given(
    matrix=matrices(),
    order=st.sampled_from(["C", "F"]),
    start=st.one_of(st.none(), st.integers(-8, 8)),
    stop=st.one_of(st.none(), st.integers(-8, 8)),
    step=st.integers(-6, 6).filter(bool),
)
def test_generated_slices(
    matrix: NDArray[Any], order: str, start: int | None, stop: int | None, step: int
) -> None:
    """Compare stepped sparse copies with dense slicing, including empty axes."""
    key = (slice(start, stop, step), slice(None, None, -1))
    np.testing.assert_array_equal(
        sppy.asarray(matrix, order=order)[key].toarray(), matrix[key]
    )


@settings(max_examples=100, deadline=None, derandomize=True)
@given(
    matrix=matrices(integers=True),
    order=st.sampled_from(["C", "F"]),
    delta=st.integers(1, 10),
)
def test_generated_exact_integer_comparison(
    matrix: NDArray[Any], order: str, delta: int
) -> None:
    """Zero tolerance must distinguish integers without rounding or overflow."""
    other = matrix.copy()
    if other.size:
        old = int(other.flat[0])
        limit = np.iinfo(other.dtype)
        other.flat[0] = old + delta if old <= limit.max - delta else old - delta
    array = sppy.asarray(matrix, order=order)
    assert array.is_approx(
        sppy.asarray(other, order=order), rtol=0, atol=0
    ) == np.array_equal(matrix, other)


@settings(max_examples=50, deadline=None, derandomize=True)
@given(
    data=st.data(),
    order=st.sampled_from(["C", "F"]),
    dtype=st.sampled_from([np.float32, np.float64, np.complex64, np.complex128]),
)
def test_generated_positive_definite_solves(
    data: st.DataObject, order: str, dtype: Any
) -> None:
    """Validate direct and iterative solutions of generated well-conditioned systems."""
    size = data.draw(st.integers(1, 6))
    base = data.draw(
        arrays(np.float64, (size, size), elements=st.integers(-3, 3).map(float))
    ).astype(dtype)
    if np.dtype(dtype).kind == "c":
        base += 0.5j * base.T
    dense = base.conj().T @ base + np.eye(size, dtype=dtype) * size
    expected = np.arange(1, size + 1, dtype=dtype)
    rhs = dense @ expected
    array = sppy.asarray(dense, order=order)
    direct = SparseLU(array).solve(rhs)
    iterative = cg(array, rhs, rtol=1e-6, maxiter=100)
    assert iterative.info == 0
    np.testing.assert_allclose(direct, expected, rtol=5e-5, atol=5e-5)
    np.testing.assert_allclose(iterative.solution, expected, rtol=5e-5, atol=5e-5)


@settings(max_examples=80, deadline=None, derandomize=True)
@given(
    shape=st.tuples(st.integers(0, 5), st.integers(0, 5)),
    order=st.sampled_from(["C", "F"]),
    scalar=st.sampled_from([np.inf, -np.inf, np.nan, complex(np.inf, 1)]),
)
def test_generated_nonfinite_scaling(
    shape: tuple[int, int], order: str, scalar: float | complex
) -> None:
    """Transform implicit zeros consistently with NumPy for nonfinite scalars."""
    dense = np.zeros(shape)
    if dense.size:
        dense.flat[0] = 2
    with np.errstate(invalid="ignore"):
        expected = dense * scalar
        actual = sppy.asarray(dense, order=order) * scalar
    np.testing.assert_allclose(actual.toarray(), expected, equal_nan=True)
