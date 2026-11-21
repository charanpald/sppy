"""Factories for owned Eigen sparse arrays."""

from __future__ import annotations

import operator
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

from sppy.sparse_array import SparseArray


def asarray(
    a: ArrayLike | SparseArray,
    dtype: DTypeLike | None = None,
    order: str | None = None,
    *,
    copy: bool | None = None,
) -> SparseArray:
    """Convert data to sparse storage, reusing compatible sparse input if possible."""
    if order not in {None, "K", "A", "C", "F"}:
        raise ValueError("Invalid order")
    if isinstance(a, SparseArray):
        dtype = a.dtype if dtype is None else np.dtype(dtype)
        if dtype == a.dtype and order in {None, "K", "A", a.order} and copy is not True:
            return a
        if copy is False:
            raise ValueError("Conversion requires a copy")
        return a.astype(dtype, order=order or "K")
    if copy is False:
        raise ValueError("Conversion to Eigen storage requires a copy")
    if not hasattr(a, "shape"):
        a = np.asarray(a, dtype=dtype)
    return SparseArray(a, dtype=dtype, order=order if order in {"C", "F"} else "C")


def zeros(
    shape: int | tuple[int, ...], dtype: DTypeLike | None = float, *, order: str = "C"
) -> SparseArray:
    """Create an empty sparse vector or matrix."""
    return SparseArray(shape, dtype=dtype, order=order)


def ones(
    shape: int | tuple[int, ...], dtype: DTypeLike | None = float, *, order: str = "C"
) -> SparseArray:
    """Create an array storing one at every position."""
    result = zeros(shape, dtype, order=order)
    result.ones()
    return result


def eye(
    n: int,
    m: int | None = None,
    k: int = 0,
    dtype: DTypeLike | None = float,
    order: str = "C",
) -> SparseArray:
    """Create a sparse identity, optionally rectangular or offset by k."""
    k = operator.index(k)
    m = n if m is None else m
    result = zeros((n, m), dtype=dtype, order=order)
    length = max(0, min(n + min(k, 0), m - max(k, 0)))
    rows = np.arange(length, dtype=np.int32) + max(-k, 0)
    cols = np.arange(length, dtype=np.int32) + max(k, 0)
    result.put(np.ones(length, dtype=dtype), rows, cols, init=True)
    return result


def diag(
    v: ArrayLike | SparseArray, k: int = 0, *, order: str = "C"
) -> SparseArray | NDArray[Any]:
    """Construct an offset diagonal or extract one from a matrix."""
    k = operator.index(k)
    if isinstance(v, SparseArray):
        if v.ndim == 2:
            return v.diagonal(k)
        v = v.toarray()
    v = np.asarray(v)
    if v.ndim == 2:
        return np.diag(v, k=k)
    if v.ndim != 1:
        raise ValueError("diag requires a vector or matrix")
    result = zeros((len(v) + abs(k),) * 2, dtype=v.dtype, order=order)
    rows = np.arange(len(v), dtype=np.int32) + max(-k, 0)
    cols = np.arange(len(v), dtype=np.int32) + max(k, 0)
    result.put(v, rows, cols, init=True)
    return result


def rand(
    shape: int | tuple[int, ...],
    density: float,
    dtype: DTypeLike | None = float,
    *,
    order: str = "C",
    rng: np.random.Generator | int | None = None,
) -> SparseArray:
    """Sample values at random coordinates, with replacement.

    Pass a NumPy Generator or seed for reproducible input. Actual density can be lower
    than requested because duplicate coordinates overwrite earlier samples.
    """
    if not np.isfinite(density) or not 0 <= density <= 1:
        raise ValueError("density must be between zero and one")
    result = zeros(shape, dtype=dtype, order=order)
    count = int(result.size * density)
    if not count:
        return result
    rng = np.random.default_rng(rng)
    indices = rng.integers(0, result.size, count)
    values = rng.random(count)
    if result.dtype.kind == "c":
        values = values + 1j * rng.random(count)
    coordinates = np.unravel_index(indices, result.shape)
    result._put_assignment(values.astype(dtype), coordinates)
    return result
