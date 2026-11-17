"""NumPy-style constructors for Eigen sparse arrays."""

import numpy as np

from sppy.csarray import SparseArray


def asarray(a, dtype=None, order=None, *, copy=None):
    """Convert to a sparse array, reusing compatible sparse input when possible.

    Dense input is copied into Eigen storage. Supported dtypes are signed
    integers (8/16/32/64 bits) and float32/float64.
    """
    if order not in {None, "K", "A", "C", "F"}:
        raise ValueError("Invalid order")
    if isinstance(a, SparseArray):
        dtype = a.dtype if dtype is None else np.dtype(dtype)
        same_order = order in {None, "K", "A", a.order}
        if same_order and dtype == a.dtype and copy is not True:
            return a
        if copy is False:
            raise ValueError("Conversion requires a copy")
        return a.astype(dtype, order=order or "K")
    if copy is False:
        raise ValueError("Conversion to Eigen storage requires a copy")
    if not hasattr(a, "shape"):
        a = np.asarray(a, dtype=dtype)
    return SparseArray(a, dtype=dtype, order=order if order in {"C", "F"} else "C")


def zeros(shape, dtype=float, storage_type=None, *, order="C", **kwargs):
    """Create an empty sparse array; order selects row or column storage."""
    legacy = kwargs.pop("storageType", None)
    if kwargs:
        raise TypeError(f"Unexpected keyword: {next(iter(kwargs))}")
    storage = storage_type if storage_type is not None else legacy
    return SparseArray(
        shape,
        dtype=dtype,
        storagetype=storage or "row",
        order=None if storage else order,
    )


def ones(shape, dtype=float, storage_type=None, *, order="C", **kwargs):
    """Create an array of ones (every element is stored explicitly)."""
    result = zeros(shape, dtype, storage_type, order=order, **kwargs)
    result._array.ones()
    return result


def eye(n, m=None, k=0, dtype=float, order="C"):
    """Create a sparse identity, optionally rectangular or offset by k.

    The old ``eye(n, dtype)`` positional spelling is also accepted.
    """
    if m is not None and not isinstance(m, (int, np.integer)):
        dtype, m = m, None
    m = n if m is None else m
    result = zeros((n, m), dtype=dtype, order=order)
    length = max(0, min(n + min(k, 0), m - max(k, 0)))
    rows = np.arange(length, dtype=np.int32) + max(-k, 0)
    cols = np.arange(length, dtype=np.int32) + max(k, 0)
    result.put(np.ones(length, dtype=dtype), rows, cols, init=True)
    return result


def diag(v, k=0, *, order="C", storagetype=None):
    """Construct an offset diagonal, or extract one from a matrix."""
    # Legacy second positional storage argument.
    if isinstance(k, str):
        storagetype, k = k, 0
    if isinstance(v, SparseArray):
        if v.ndim == 2:
            return v.diagonal(k)
        v = v.toarray()
    v = np.asarray(v)
    if v.ndim == 2:
        return np.diag(v, k=k)
    if v.ndim != 1:
        raise ValueError("diag requires a vector or matrix")
    result = zeros(
        (len(v) + abs(k),) * 2, dtype=v.dtype, order=order, storage_type=storagetype
    )
    rows = np.arange(len(v), dtype=np.int32) + max(-k, 0)
    cols = np.arange(len(v), dtype=np.int32) + max(k, 0)
    result.put(v, rows, cols, init=True)
    return result


def rand(shape, density, dtype=float, storagetype="col", *, order=None, rng=None):
    """Sample sparse values; pass a NumPy Generator for reproducible new code.

    Coordinates are sampled with replacement, so actual density may be lower
    than the requested density. The legacy global NumPy RNG remains supported.
    """
    if not 0 <= density <= 1:
        raise ValueError("density must be between zero and one")
    result = SparseArray(shape, dtype, storagetype, order=order)
    count = int(result.size * density)
    if not result.size or not count:
        return result
    if rng is None:
        indices = np.random.randint(0, result.size, count)
        values = np.random.random(count)
    else:
        indices = rng.integers(0, result.size, count)
        values = rng.random(count)
    coordinates = np.unravel_index(indices, result.shape)
    result._put_coordinates(values.astype(dtype), coordinates)
    return result
