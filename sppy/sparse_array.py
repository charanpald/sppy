"""One- and two-dimensional arrays with owned Eigen sparse storage."""

from __future__ import annotations

import math
import operator
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy
from numpy.typing import ArrayLike, DTypeLike, NDArray

from sppy import _native
from sppy._typing import Scalar

if TYPE_CHECKING:
    from scipy.sparse import csc_matrix, csr_matrix


_DTYPES = {
    numpy.dtype(name): name
    for name in (
        "int8",
        "int16",
        "int32",
        "int64",
        "float32",
        "float64",
        "complex64",
        "complex128",
    )
}


def _shape(shape: int | numpy.integer[Any] | tuple[int, ...]) -> tuple[int, ...]:
    """Validate and normalize a sparse vector or matrix shape."""
    if isinstance(shape, (int, numpy.integer)):
        shape = (operator.index(shape),)
    if not isinstance(shape, tuple) or len(shape) not in {1, 2}:
        raise ValueError("Only one- and two-dimensional shapes are supported")
    if any(
        not isinstance(n, (int, numpy.integer))
        or isinstance(n, (bool, numpy.bool_))
        or n < 0
        or n > numpy.iinfo(numpy.int32).max
        for n in shape
    ):
        raise ValueError("Dimensions must be nonnegative 32-bit integers")
    return tuple(map(int, shape))


class SparseArray:
    """An owned Eigen sparse vector or matrix.

    Parameters:
        data: A shape (integer or tuple), dense data, sparse array, or SciPy sparse object.
        dtype: Signed integer, float32/64, or complex64/128 dtype. Inferred from data.
        order: "C" for row-major or "F" for column-major storage. Defaults to "C".

    Tuple input specifies a shape. Use :func:`sppy.asarray` to treat tuples as data.
    Sparse indexing returns independent copies, rather than views.
    """

    _array: _native._Matrix[Any]

    __array_priority__ = 1000

    def __init__(
        self,
        data: ArrayLike | SparseArray | int | tuple[int, ...],
        dtype: DTypeLike | None = None,
        *,
        order: str = "C",
    ) -> None:
        """Create owned sparse storage from a shape, dense data, or sparse input."""
        if order not in {"C", "F"}:
            raise ValueError("order must be 'C' or 'F'")
        source: Any = data
        if not isinstance(data, (tuple, int, numpy.integer)) and not hasattr(
            data, "shape"
        ):
            source = numpy.asarray(data)
        shape = _shape(
            source if isinstance(source, (tuple, int, numpy.integer)) else source.shape
        )
        dtype = numpy.dtype(
            dtype if dtype is not None else getattr(source, "dtype", numpy.float64)
        )
        if dtype not in _DTYPES:
            raise ValueError(f"Unsupported dtype: {dtype}")
        self._dtype, self._order, self._shape = dtype, order, shape
        if isinstance(source, SparseArray):
            self._array = source.astype(dtype, order=order, copy=True)._array
            return
        if len(shape) == 2 and getattr(source, "format", None) in {"csr", "csc"}:
            source_order = "C" if source.format == "csr" else "F"
            converted = self.from_compressed(
                shape,
                source.data,
                source.indices,
                source.indptr,
                dtype=dtype,
                order=source_order,
            )
            self._array = converted.astype(dtype, order=order, copy=False)._array
            return
        backend = getattr(
            _native, f"{_DTYPES[dtype]}_{order if len(shape) == 2 else 'F'}"
        )
        self._array = backend(shape[0], shape[1] if len(shape) == 2 else 1)
        if hasattr(source, "tocoo"):
            coo = source.tocoo(copy=True)
            if coo.ndim == 1:
                self._put_coordinates(coo.data.astype(dtype), coo.coords)
            else:
                self._put_coordinates(coo.data.astype(dtype), (coo.row, coo.col))
        elif hasattr(source, "shape"):
            source = numpy.asarray(source, dtype=dtype)
            indices = source.nonzero()
            self._put_coordinates(source[indices], indices)

    @classmethod
    def _from_backend(
        cls,
        backend: Any,
        dtype: DTypeLike | None,
        order: str,
        shape: tuple[int, ...] | None = None,
    ) -> SparseArray:
        """Wrap owned native storage with its logical shape and dtype."""
        result = object.__new__(cls)
        result._array, result._dtype, result._order = backend, numpy.dtype(dtype), order
        result._shape = tuple(backend.shape) if shape is None else tuple(shape)
        return result

    def _wrap_backend(
        self, backend: Any, shape: tuple[int, ...] | None = None
    ) -> SparseArray:
        """Wrap native storage using this array's dtype and order."""
        return self._from_backend(backend, self.dtype, self.order, shape)

    @property
    def dtype(self) -> numpy.dtype[Any]:
        """Stored NumPy scalar dtype."""
        return self._dtype

    @property
    def shape(self) -> tuple[int, ...]:
        """Logical vector or matrix dimensions."""
        return self._shape

    @property
    def ndim(self) -> int:
        """Number of logical array dimensions."""
        return len(self.shape)

    @property
    def size(self) -> int:
        """Number of logical entries, including implicit zeros."""
        return self.shape[0] if self.ndim == 1 else self.shape[0] * self.shape[1]

    @property
    def order(self) -> str:
        """'C' for row storage, 'F' for column storage."""
        return self._order

    @property
    def nnz(self) -> int:
        """Number of stored entries, including explicit zeros."""
        return self._array.nnz

    @property
    def is_compressed(self) -> bool:
        """Whether storage uses compressed sparse buffers."""
        return self._array.is_compressed

    @property
    def inner_size(self) -> int:
        """Length of each inner storage vector."""
        return self._array.inner_size

    @property
    def outer_size(self) -> int:
        """Number of outer storage vectors."""
        return self._array.outer_size

    def stored_indices(self) -> tuple[NDArray[numpy.int32], ...]:
        """Coordinates in storage order, aligned with :meth:`values`, including zeros."""
        coordinates = self._array.coordinates()
        return coordinates if self.ndim == 2 else coordinates[:1]

    def nonzero(self) -> tuple[NDArray[numpy.int32], ...]:
        """Coordinates of nonzero values, excluding explicitly stored zeros."""
        mask = self.values() != 0
        return tuple(index[mask] for index in self.stored_indices())

    def values(self) -> NDArray[Any]:
        """Return a copy of stored values aligned with :meth:`stored_indices`."""
        return self._array.values()

    def _put_coordinates(
        self, values: ArrayLike, coordinates: tuple[ArrayLike, ...]
    ) -> None:
        """Replace storage with coordinates, summing duplicate entries."""
        indices = tuple(numpy.asarray(i) for i in coordinates)
        if len(indices) != self.ndim or any(i.ndim != 1 for i in indices):
            raise ValueError("Expected one coordinate vector per axis")
        values = numpy.asarray(values, dtype=self.dtype)
        if values.ndim != 1 or any(len(i) != len(values) for i in indices):
            raise ValueError("Coordinate and value lengths differ")
        for axis, index in enumerate(indices):
            if index.size and (
                index.dtype.kind not in "iu"
                or (index < 0).any()
                or (index >= self.shape[axis]).any()
            ):
                raise ValueError("Coordinate out of bounds")
        indices = tuple(numpy.ascontiguousarray(i, dtype=numpy.int32) for i in indices)
        rows = indices[0]
        columns = (
            indices[1]
            if self.ndim == 2
            else numpy.zeros(len(values), dtype=numpy.int32)
        )
        self._array.assemble(rows, columns, numpy.ascontiguousarray(values))

    def _put_assignment(
        self, values: ArrayLike, coordinates: tuple[ArrayLike, ...]
    ) -> None:
        # Assignment has last-write-wins semantics; assembly sums duplicates.
        """Assign coordinates with last-write-wins semantics."""
        rows = numpy.ascontiguousarray(coordinates[0], dtype=numpy.int32)
        columns = (
            numpy.ascontiguousarray(coordinates[1], dtype=numpy.int32)
            if self.ndim == 2
            else numpy.zeros(len(rows), dtype=numpy.int32)
        )
        self._array.assign(
            rows, columns, numpy.ascontiguousarray(values, dtype=self.dtype)
        )

    def put(
        self,
        data: ArrayLike | SparseArray | int | tuple[int, ...],
        row_indices: ArrayLike,
        column_indices: ArrayLike | None = None,
        *,
        init: bool = False,
    ) -> None:
        """Assign paired coordinates; init=True replaces storage and sums duplicates."""
        indices: tuple[NDArray[Any], ...] = (numpy.asarray(row_indices),)
        if self.ndim == 2:
            if column_indices is None:
                raise ValueError("Matrix assignment requires column indices")
            indices += (numpy.asarray(column_indices),)
        elif column_indices is not None:
            raise ValueError("Vectors have one coordinate axis")
        if any(i.ndim != 1 for i in indices) or any(
            len(i) != len(indices[0]) for i in indices
        ):
            raise ValueError("Coordinate lengths differ")
        for axis, index in enumerate(indices):
            if index.size and (
                index.dtype.kind not in "iu"
                or (index < 0).any()
                or (index >= self.shape[axis]).any()
            ):
                raise ValueError("Coordinate out of bounds")
        data = numpy.asarray(data)
        if data.ndim != 0 and (data.ndim != 1 or len(data) != len(indices[0])):
            raise ValueError("Expected a scalar or one value per coordinate")
        values = numpy.broadcast_to(data, (len(indices[0]),)).astype(self.dtype)
        if init:
            self._put_coordinates(values, indices)
        else:
            self._put_assignment(values, indices)

    @classmethod
    def from_triplets(
        cls,
        shape: int | tuple[int, ...],
        rows: ArrayLike,
        columns: ArrayLike,
        values: ArrayLike,
        *,
        dtype: DTypeLike | None = None,
        order: str = "C",
    ) -> SparseArray:
        """Assemble a matrix from unsorted coordinates; duplicate entries are summed."""
        values = numpy.asarray(values, dtype=dtype)
        result = cls(_shape(shape), dtype=values.dtype, order=order)
        if result.ndim != 2:
            raise ValueError("Triplets require a matrix shape")
        result._put_coordinates(values, (rows, columns))
        return result

    @classmethod
    def from_compressed(
        cls,
        shape: int | tuple[int, ...],
        values: ArrayLike,
        indices: ArrayLike,
        indptr: ArrayLike,
        *,
        order: str = "C",
        dtype: DTypeLike | None = None,
    ) -> SparseArray:
        """Copy CSR (C) or CSC (F) arrays, validating pointer bounds and indices."""
        shape = _shape(shape)
        if len(shape) != 2 or order not in {"C", "F"}:
            raise ValueError("Compressed storage requires a matrix and C/F order")
        values, indices, indptr = (
            numpy.asarray(values, dtype=dtype),
            numpy.asarray(indices),
            numpy.asarray(indptr),
        )
        outer, inner = shape if order == "C" else shape[::-1]
        if (
            values.ndim != 1
            or indices.ndim != 1
            or indptr.ndim != 1
            or len(indices) != len(values)
            or len(indptr) != outer + 1
        ):
            raise ValueError("Invalid compressed array lengths")
        dtype = values.dtype
        if dtype not in _DTYPES:
            raise ValueError(f"Unsupported dtype: {dtype}")
        # Native validation avoids repeated NumPy scans for the usual SciPy
        # int32 buffers. Validate wider indices before narrowing them.
        if indices.dtype != numpy.dtype(numpy.int32) or indptr.dtype != numpy.dtype(
            numpy.int32
        ):
            if (
                indptr.dtype.kind not in "iu"
                or indptr[0] != 0
                or indptr[-1] != len(values)
                or (indptr[1:] < indptr[:-1]).any()
                or (indptr > len(values)).any()
            ):
                raise ValueError("Invalid compressed pointers")
            if indices.size and (
                indices.dtype.kind not in "iu"
                or (indices < 0).any()
                or (indices >= inner).any()
            ):
                raise ValueError("Invalid compressed inner indices")
            if len(values) > numpy.iinfo(numpy.int32).max:
                raise ValueError("Too many stored entries")
            indices = numpy.asarray(indices, dtype=numpy.int32)
            indptr = numpy.asarray(indptr, dtype=numpy.int32)
        backend = getattr(_native, f"{_DTYPES[dtype]}_{order}")
        return cls._from_backend(
            backend.from_compressed(*shape, values, indices, indptr),
            dtype,
            order,
            shape,
        )

    def compressed_arrays(
        self,
    ) -> tuple[NDArray[Any], NDArray[numpy.int32], NDArray[numpy.int32]]:
        """Return independent (values, indices, indptr) copies in CSR/CSC order."""
        if self.ndim != 2:
            raise ValueError("Compressed arrays require a matrix")
        return self._array.compressed_arrays()

    def toarray(self) -> NDArray[Any]:
        """Allocate a dense array, preserving dtype."""
        result = numpy.zeros(self.shape, dtype=self.dtype)
        result[self.stored_indices()] = self.values()
        return result

    def to_scipy_csr(self) -> csr_matrix:
        """Copy into a SciPy CSR matrix (SciPy is optional)."""
        from scipy.sparse import csr_matrix

        if self.ndim != 2:
            raise ValueError("SciPy matrices require two dimensions")
        data, indices, indptr = self.astype(
            self.dtype, order="C", copy=False
        ).compressed_arrays()
        return csr_matrix((data, indices, indptr), shape=self.shape)

    def to_scipy_csc(self) -> csc_matrix:
        """Copy into a SciPy CSC matrix (SciPy is optional)."""
        from scipy.sparse import csc_matrix

        if self.ndim != 2:
            raise ValueError("SciPy matrices require two dimensions")
        data, indices, indptr = self.astype(
            self.dtype, order="F", copy=False
        ).compressed_arrays()
        return csc_matrix((data, indices, indptr), shape=self.shape)

    def copy(self, order: str = "K") -> SparseArray:
        """Independent copy, optionally changing storage order."""
        if order not in {"K", "C", "F"}:
            raise ValueError("Invalid order")
        return self.astype(self.dtype, order=order, copy=True)

    def astype(
        self,
        dtype: DTypeLike | None,
        order: str = "K",
        casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "unsafe",
        subok: bool = True,
        copy: bool | None = True,
    ) -> SparseArray:
        """Cast stored values without allocating a dense array."""
        dtype = numpy.dtype(dtype)
        if not numpy.can_cast(self.dtype, dtype, casting=casting):
            raise TypeError(
                f"Cannot cast {self.dtype} to {dtype} with {casting!r} casting"
            )
        if order not in {"K", "A", "C", "F"}:
            raise ValueError("Invalid order")
        order = self.order if order in {"K", "A"} else order
        if not copy and dtype == self.dtype and order == self.order:
            return self
        if dtype == self.dtype:
            if order == self.order or self.ndim == 1:
                backend = self._array.copy()
            else:
                backend = (
                    self._array.to_row_major()
                    if order == "C"
                    else self._array.to_column_major()
                )
            return self._from_backend(backend, dtype, order, self.shape)
        result = SparseArray(self.shape, dtype=dtype, order=order)
        result._put_coordinates(
            self.values().astype(dtype, casting=casting), self.stored_indices()
        )
        return result

    def _binary_sparse(
        self, other: SparseArray | ArrayLike, operation: str
    ) -> SparseArray:
        """Apply a native coefficient operation to equal-shaped operands."""
        if numpy.isscalar(other):
            if other == 0:
                return self.copy()
            raise NotImplementedError(
                "Adding nonzero scalars would densify the sparse array"
            )
        if not isinstance(other, SparseArray):
            other = SparseArray(numpy.asarray(other))
        if self.shape != other.shape:
            raise ValueError("Sparse arithmetic requires equal shapes")
        if self.dtype == other.dtype:
            left = self
            right = (
                other
                if other.order == self.order or self.ndim == 1
                else other.astype(self.dtype, order=self.order, copy=False)
            )
        else:
            dtype = numpy.result_type(self.dtype, other.dtype)
            left = self.astype(dtype, copy=False)
            right = other.astype(dtype, order=self.order, copy=False)
        return left._wrap_backend(
            getattr(left._array, operation)(right._array), self.shape
        )

    def __add__(self, other: SparseArray | ArrayLike) -> SparseArray:
        """Add equal-shaped arrays, or copy when adding zero."""
        return self._binary_sparse(other, "add")

    def __sub__(self, other: SparseArray | ArrayLike) -> SparseArray:
        """Subtract equal-shaped arrays, or copy when subtracting zero."""
        return self._binary_sparse(other, "subtract")

    def __rsub__(self, other: SparseArray | ArrayLike) -> SparseArray:
        """Subtract this array from the left operand."""
        return -self + other

    __radd__ = __add__

    def __neg__(self) -> SparseArray:
        """Return the elementwise additive inverse."""
        return self * -1

    def __abs__(self) -> SparseArray:
        """Return elementwise magnitudes."""
        return self._map_values(numpy.abs)

    def multiply(self, other: SparseArray | ArrayLike) -> SparseArray:
        """Elementwise sparse multiplication, with matching shapes."""
        if numpy.isscalar(other):
            return self * other
        return self._binary_sparse(other, "multiply")

    def dot(
        self, other: SparseArray | ArrayLike
    ) -> SparseArray | NDArray[Any] | Scalar:
        """Matrix/vector product, equivalent to @ (without conjugation)."""
        if isinstance(other, SparseArray):
            if self.shape[-1] != other.shape[0]:
                raise ValueError("Product shape mismatch")
            dtype = (
                self.dtype
                if self.dtype == other.dtype
                else numpy.result_type(self.dtype, other.dtype)
            )
            if self.ndim == 1:
                if other.ndim == 1:
                    indices = self.stored_indices()[0]
                    return numpy.dot(
                        self.values().astype(dtype),
                        numpy.asarray(other[indices], dtype=dtype),
                    )
                return other.T.dot(self)
            storage = "F" if other.ndim == 1 else self.order
            left = (
                self
                if dtype == self.dtype and storage == self.order
                else self.astype(dtype, order=storage, copy=False)
            )
            right = (
                other
                if dtype == other.dtype and (storage == other.order or other.ndim == 1)
                else other.astype(dtype, order=storage, copy=False)
            )
            backend = left._array.matmul(right._array)
            shape = (self.shape[0],) if other.ndim == 1 else tuple(backend.shape)
            return self._from_backend(backend, dtype, self.order, shape)
        dense = numpy.asarray(other)
        if dense.ndim not in {1, 2} or self.shape[-1] != dense.shape[0]:
            raise ValueError("Product shape mismatch")
        dtype = (
            self.dtype
            if self.dtype == dense.dtype
            else numpy.result_type(self.dtype, dense.dtype)
        )
        if self.ndim == 1:
            return self.values().astype(dtype) @ dense[self.stored_indices()[0]].astype(
                dtype
            )
        left = self if dtype == self.dtype else self.astype(dtype, copy=False)
        operand = dense[:, None] if dense.ndim == 1 else dense
        result = left._array.dense_matmul(numpy.ascontiguousarray(operand, dtype=dtype))
        return result[:, 0] if dense.ndim == 1 else result

    __matmul__ = dot

    def __rmatmul__(self, other: SparseArray | ArrayLike) -> NDArray[Any] | Scalar:
        """Multiply a dense left operand by this array."""
        dense = numpy.asarray(other)
        if dense.ndim == 1:
            return self.T.dot(dense)
        return self.T.dot(dense.T).T

    def parallel_dot(
        self, other: SparseArray | ArrayLike
    ) -> SparseArray | NDArray[Any] | Scalar:
        """Dense product using Eigen's configured thread count and optional OpenMP."""
        return self.dot(other)

    def transpose(self, *axes: int | Sequence[int]) -> SparseArray:
        """Return a sparse copy with axes reversed."""
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        if any(
            not isinstance(a, (int, numpy.integer)) or not -self.ndim <= a < self.ndim
            for a in axes
        ):
            raise ValueError("Invalid transpose axes")
        normalized_axes = tuple(operator.index(cast(int, a)) % self.ndim for a in axes)
        axes = normalized_axes
        if axes and (len(axes) != self.ndim or len(set(axes)) != self.ndim):
            raise ValueError("Invalid transpose axes")
        if axes == tuple(range(self.ndim)) or self.ndim == 1:
            return self.copy()
        if axes and axes != tuple(reversed(range(self.ndim))):
            raise ValueError("Invalid transpose axes")
        return self._wrap_backend(self._array.transpose())

    T = property(transpose)

    def conjugate(self) -> SparseArray:
        """Conjugate stored values without changing sparsity."""
        return self._map_values(numpy.conjugate)

    def adjoint(self) -> SparseArray:
        """Conjugate transpose (H)."""
        return (
            self.conjugate()
            if self.ndim == 1
            else self._wrap_backend(self._array.adjoint())
        )

    H = property(adjoint)

    @property
    def real(self) -> SparseArray:
        """Real components in independent sparse storage."""
        return self._map_values(numpy.real)

    @property
    def imag(self) -> SparseArray:
        """Imaginary components in independent sparse storage."""
        return self._map_values(numpy.imag)

    def reserve(self, n: int | ArrayLike) -> None:
        """Reserve a total capacity, or additional capacity per outer vector."""
        if numpy.ndim(n) == 0:
            self._array.reserve(operator.index(cast(int, n)))
        else:
            n = numpy.asarray(n)
            if (
                n.dtype.kind not in "iu"
                or (n < 0).any()
                or (n > numpy.iinfo(numpy.int32).max).any()
            ):
                raise ValueError("Invalid capacities")
            self._array.reserve_per_outer(numpy.ascontiguousarray(n, dtype=numpy.int32))

    def compress(self) -> None:
        """Convert to compressed storage."""
        self._array.compress()

    def uncompress(self) -> None:
        """Enable uncompressed insertion storage."""
        self._array.uncompress()

    def squeeze(self) -> None:
        """Compress storage and release unused entry capacity."""
        self._array.squeeze()

    def set_zero(self) -> None:
        """Remove all entries while preserving shape."""
        self._array.set_zero()

    def fill(self, value: Scalar) -> None:
        """Fill every position; nonzero values store the entire logical array."""
        if value == 0:
            self.set_zero()
        else:
            indices = numpy.unravel_index(numpy.arange(self.size), self.shape)
            self._put_coordinates(
                numpy.full(self.size, value, dtype=self.dtype), indices
            )

    def ones(self) -> None:
        """Fill every position with one."""
        self.fill(1)

    def resize(self, shape: int | tuple[int, ...], *, preserve: bool = False) -> None:
        """Resize in place; preserve=True retains entries within new bounds."""
        shape = _shape(shape)
        if len(shape) != self.ndim:
            raise ValueError("Resize cannot change dimensionality")
        self._array.resize(shape[0], shape[1] if self.ndim == 2 else 1, preserve)
        self._shape = shape

    def prune(self, tolerance: float = 1e-12) -> None:
        """Remove entries whose absolute value is at most tolerance, in place."""
        self._array.prune(tolerance)

    def eliminate_zeros(self) -> None:
        """Remove only exact stored zeros, in place."""
        self.prune(0)

    def triangular(
        self, *, lower: bool = True, unit_diagonal: bool = False
    ) -> SparseArray:
        """Materialize a sparse lower/upper triangular copy."""
        if self.ndim != 2:
            raise ValueError("Triangular operations require a matrix")
        return self._wrap_backend(self._array.triangular(lower, unit_diagonal))

    def selfadjoint(self, *, lower: bool = True) -> SparseArray:
        """Reconstruct a Hermitian matrix from its lower or upper triangle."""
        if self.ndim != 2:
            raise ValueError("Self-adjoint operations require a matrix")
        return self._wrap_backend(self._array.selfadjoint(lower))

    def permuted(
        self, rows: ArrayLike | None = None, columns: ArrayLike | None = None
    ) -> SparseArray:
        """Gather rows/columns in permutation order; vectors accept only rows."""

        def check(values: ArrayLike | None, size: int) -> NDArray[Any]:
            """Validate a gather permutation, using the identity when omitted."""
            if values is None:
                return numpy.arange(size)
            values = numpy.asarray(values)
            if (
                values.ndim != 1
                or values.dtype.kind not in "iu"
                or not numpy.array_equal(numpy.sort(values), numpy.arange(size))
            ):
                raise ValueError("Expected a permutation of the axis")
            return values

        rows = check(rows, self.shape[0])
        if self.ndim == 1:
            if columns is not None:
                raise ValueError("Vectors have one axis")
            return cast(SparseArray, self[rows])
        return self.subarray(rows, check(columns, self.shape[1]))

    def row_indices(self, row: int) -> NDArray[numpy.int32]:
        """Stored column indices in a row."""
        if self.ndim != 2 or not 0 <= row < self.shape[0]:
            raise ValueError("Row out of bounds")
        rows, columns = self.stored_indices()
        return numpy.sort(columns[rows == row])

    def column_indices(self, column: int) -> NDArray[numpy.int32]:
        """Stored row indices in a column."""
        if self.ndim != 2 or not 0 <= column < self.shape[1]:
            raise ValueError("Column out of bounds")
        rows, columns = self.stored_indices()
        return numpy.sort(rows[columns == column])

    def nonzero_rows_list(self) -> list[NDArray[numpy.int32]]:
        """Return stored column indices for each matrix row."""
        return [self.row_indices(i) for i in range(self.shape[0])]

    def nonzero_rows_ptr(self) -> tuple[NDArray[numpy.int32], NDArray[numpy.int32]]:
        """Return independent CSR row pointers and stored column indices."""
        _, indices, indptr = self.astype(
            self.dtype, order="C", copy=False
        ).compressed_arrays()
        return indptr, indices

    def submatrix(
        self, start_row: int, start_column: int, rows: int, columns: int
    ) -> SparseArray:
        """Copy a rectangular block."""
        if (
            self.ndim != 2
            or min(start_row, start_column, rows, columns) < 0
            or start_row + rows > self.shape[0]
            or start_column + columns > self.shape[1]
        ):
            raise ValueError("Invalid block bounds")
        return cast(
            SparseArray,
            self[start_row : start_row + rows, start_column : start_column + columns],
        )

    def __truediv__(self, scalar: Scalar) -> SparseArray:
        """Divide by a scalar; division by zero also transforms implicit zeros."""
        if not numpy.isscalar(scalar):
            raise ValueError("Sparse division requires a scalar")
        return self._map_values(lambda values: numpy.true_divide(values, scalar))

    def _inplace(self, result: SparseArray) -> SparseArray:
        """Replace storage after enforcing shape and same-kind casting."""
        if result.shape != self.shape:
            raise ValueError("In-place operations cannot change shape")
        result = result.astype(
            self.dtype, order=self.order, casting="same_kind", copy=False
        )
        self._array = result._array
        return self

    def __iadd__(self, other: SparseArray | ArrayLike) -> SparseArray:
        """Add in place with same-kind casting."""
        return self._inplace(self + other)

    def __isub__(self, other: SparseArray | ArrayLike) -> SparseArray:
        """Subtract in place with same-kind casting."""
        return self._inplace(self - other)

    def __imul__(self, other: SparseArray | ArrayLike) -> SparseArray:
        """Multiply elementwise in place with same-kind casting."""
        return self._inplace(self * other)

    def __itruediv__(self, other: Scalar) -> SparseArray:
        """Divide in place with same-kind casting."""
        return self._inplace(self / other)

    def vdot(self, other: SparseArray | ArrayLike) -> Scalar:
        """Conjugating inner product of equally sized sparse vectors."""
        if not isinstance(other, SparseArray):
            other = SparseArray(numpy.asarray(other))
        if self.ndim != 1 or other.ndim != 1 or self.shape != other.shape:
            raise ValueError("vdot requires equal-length vectors")
        return cast(Scalar, self.conjugate().dot(other))

    def outer(self, other: SparseArray | ArrayLike) -> SparseArray:
        """Sparse vector outer product without conjugation."""
        if not isinstance(other, SparseArray):
            other = SparseArray(numpy.asarray(other))
        if self.ndim != 1 or other.ndim != 1:
            raise ValueError("outer requires vectors")
        dtype = numpy.result_type(self.dtype, other.dtype)
        left, right = self.astype(dtype, copy=False), other.astype(dtype, copy=False)
        backend = left._array.matmul(right._array.transpose())
        return self._from_backend(backend, dtype, "F").copy(order=self.order)

    def lp_norm(self, p: int = 2) -> float:
        """Entrywise p-norm for p >= 1, including the infinity norm."""
        if p < 1 or numpy.isnan(p):
            raise ValueError("p must be at least one")
        values = numpy.abs(
            self.values().astype(numpy.result_type(self.dtype, numpy.float64))
        )
        if p == numpy.inf:
            return values.max(initial=0)
        return numpy.sum(values**p) ** (1 / p)

    def normalize(self) -> SparseArray:
        """Normalize a floating-point or complex vector in place."""
        if self.dtype.kind not in "fc":
            raise TypeError(
                "In-place normalization requires floating-point or complex storage"
            )
        return self._inplace(self.normalized())

    def set_identity(self) -> None:
        """Replace a matrix with an identity of the same (possibly rectangular) shape."""
        if self.ndim != 2:
            raise ValueError("Identity requires a matrix")
        indices = numpy.arange(min(self.shape), dtype=numpy.int32)
        self._put_coordinates(
            numpy.ones(len(indices), dtype=self.dtype), (indices, indices)
        )

    def set_diagonal(self, values: ArrayLike, offset: int = 0) -> None:
        """Assign a diagonal without allocating the full matrix."""
        offset = operator.index(offset)
        if self.ndim != 2:
            raise ValueError("Diagonal assignment requires a matrix")
        length = max(
            0, min(self.shape[0] + min(offset, 0), self.shape[1] - max(offset, 0))
        )
        rows = numpy.arange(length) + max(-offset, 0)
        columns = numpy.arange(length) + max(offset, 0)
        self[rows, columns] = values

    def swap(self, other: SparseArray) -> None:
        """Exchange owned storage in constant time; dtype/order/dimensionality must match."""
        if (
            not isinstance(other, SparseArray)
            or self.dtype != other.dtype
            or self.order != other.order
            or self.ndim != other.ndim
        ):
            raise ValueError("Swap requires matching dtype, order, and dimensionality")
        self._array, other._array = other._array, self._array
        self._shape, other._shape = other._shape, self._shape

    def insert_from_triplets(
        self, rows: ArrayLike, columns: ArrayLike, values: ArrayLike
    ) -> None:
        """Add triplets to existing entries, summing duplicate contributions."""
        addition = SparseArray.from_triplets(
            self.shape, rows, columns, values, dtype=self.dtype, order=self.order
        )
        self._array = self._array.add(addition._array)

    def scale_rows(self, values: ArrayLike) -> SparseArray:
        """Return diag(values) @ self without forming the diagonal matrix."""
        if self.ndim != 2:
            raise ValueError("Row scaling requires a matrix")
        values = numpy.asarray(values)
        if values.shape != (self.shape[0],):
            raise ValueError("Expected one scale per row")
        indices = self.stored_indices()
        data = self.values() * values[indices[0]]
        result = SparseArray(self.shape, dtype=data.dtype, order=self.order)
        result._put_coordinates(data, indices)
        return result

    def scale_columns(self, values: ArrayLike) -> SparseArray:
        """Return self @ diag(values) without forming the diagonal matrix."""
        return self.T.scale_rows(values).T

    def unary(self, function: Callable[[NDArray[Any]], ArrayLike]) -> SparseArray:
        """Map a NumPy-compatible function, including implicit zeros.

        If function(0) is nonzero, every logical position must be stored.
        """
        return self._map_values(function)

    def abs2(self) -> SparseArray:
        """Squared magnitudes, with real output for complex input."""
        return self._map_values(lambda x: numpy.abs(x) ** 2)

    def sqrt(self) -> SparseArray:
        """Apply square root to every logical entry."""
        return self._map_values(numpy.sqrt)

    def exp(self) -> SparseArray:
        """Apply exponential to every logical entry."""
        return self._map_values(numpy.exp)

    def log(self) -> SparseArray:
        """Apply natural logarithm to every logical entry."""
        return self._map_values(numpy.log)

    def _cwise_extreme(
        self,
        other: SparseArray | ArrayLike,
        function: numpy.ufunc,
    ) -> SparseArray:
        """Combine stored coordinates using an elementwise extremum."""
        if not isinstance(other, SparseArray):
            other = SparseArray(numpy.asarray(other))
        if other.shape != self.shape:
            raise ValueError("Elementwise operands must have matching shapes")
        left = dict(zip(zip(*self.stored_indices()), self.values()))
        right = dict(zip(zip(*other.stored_indices()), other.values()))
        coordinates = sorted(left.keys() | right.keys())
        dtype = numpy.result_type(self.dtype, other.dtype)
        values = numpy.asarray(
            [function(left.get(i, 0), right.get(i, 0)) for i in coordinates],
            dtype=dtype,
        )
        indices = tuple(
            numpy.asarray([i[axis] for i in coordinates], dtype=numpy.int32)
            for axis in range(self.ndim)
        )
        result = SparseArray(self.shape, dtype=dtype, order=self.order)
        result._put_coordinates(values, indices)
        result.eliminate_zeros()
        return result

    def minimum(self, other: SparseArray | ArrayLike) -> SparseArray:
        """Elementwise minimum, including implicit zeros."""
        return self._cwise_extreme(other, numpy.minimum)

    def maximum(self, other: SparseArray | ArrayLike) -> SparseArray:
        """Elementwise maximum, including implicit zeros."""
        return self._cwise_extreme(other, numpy.maximum)

    def squared_norm(self) -> float:
        """Squared Euclidean/Frobenius norm, accumulated in floating point."""
        values = self.values().astype(numpy.result_type(self.dtype, numpy.float64))
        return numpy.vdot(values, values).real

    def normalized(self) -> SparseArray:
        """Return a vector divided by its Euclidean norm; zero vectors remain zero."""
        if self.ndim != 1:
            raise ValueError("Normalization requires a vector")
        length = numpy.sqrt(self.squared_norm())
        return self.copy() if length == 0 else self * (1 / length)

    def is_approx(
        self, other: SparseArray | ArrayLike, *, rtol: float = 1e-5, atol: float = 1e-8
    ) -> bool:
        """Compare equal-shaped arrays using a Frobenius error bound, without densifying."""
        if not isinstance(other, SparseArray) or other.shape != self.shape:
            return False
        if rtol < 0 or atol < 0 or not numpy.isfinite([rtol, atol]).all():
            raise ValueError("Tolerances must be nonnegative and finite")
        if self.dtype.kind == other.dtype.kind == "i":
            # Subtract as Python integers before any conversion to floating point.
            # This preserves adjacent int64 values as well as opposite endpoints.
            left = dict(zip(zip(*self.stored_indices()), map(int, self.values())))
            right = dict(zip(zip(*other.stored_indices()), map(int, other.values())))
            squared_distance = sum(
                (left.get(index, 0) - right.get(index, 0)) ** 2
                for index in left.keys() | right.keys()
            )
            distance = math.sqrt(squared_distance)
        else:
            dtype = numpy.result_type(self.dtype, other.dtype, numpy.float64)
            difference = self.astype(dtype, copy=False) - other.astype(
                dtype, copy=False
            )
            distance = numpy.sqrt(difference.squared_norm())
        return bool(distance <= atol + rtol * numpy.sqrt(other.squared_norm()))

    def is_zero(self, tolerance: float = 0) -> bool:
        """Whether all stored values are within an absolute tolerance of zero."""
        if not numpy.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Invalid tolerance")
        values = self.values()
        if self.dtype.kind == "i":
            # abs(min_signed_integer) overflows in fixed-width integer dtypes.
            return bool(((values >= -tolerance) & (values <= tolerance)).all())
        return bool((numpy.abs(values) <= tolerance).all())

    def __getstate__(self) -> tuple[Any, ...]:
        """Export owned coordinates and values for serialization."""
        return self.shape, self.dtype, self.order, self.stored_indices(), self.values()

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        """Restore owned storage from serialized coordinates and values."""
        shape, dtype, order, indices, values = state
        SparseArray.__init__(self, shape, dtype=dtype, order=order)
        self._put_coordinates(values, indices)

    def __repr__(self) -> str:
        """Return a concise representation of array metadata."""
        return f"SparseArray(shape={self.shape}, dtype={self.dtype}, nnz={self.nnz}, order={self.order!r})"

    def __str__(self) -> str:
        """Describe dtype, shape, storage size, and order."""
        return f"SparseArray dtype:{self.dtype} shape:{self.shape} non-zeros:{self.nnz} order:{self.order}"

    def _indices(self, index: Any, axis: int) -> tuple[NDArray[Any], bool]:
        """Normalize and validate indices for a single axis."""
        if isinstance(index, slice):
            return numpy.arange(
                *index.indices(self.shape[axis]), dtype=numpy.int64
            ), False
        scalar = isinstance(index, (int, numpy.integer))
        values = numpy.asarray(index)
        if values.dtype.kind == "b":
            if values.ndim != 1 or len(values) != self.shape[axis]:
                raise ValueError("Boolean index has the wrong shape")
            return numpy.flatnonzero(values), False
        if values.size == 0 and isinstance(index, (list, tuple)):
            values = values.astype(numpy.int64)
        if values.dtype.kind not in "iu":
            raise ValueError("Indices must be integers")
        if values.size and (
            (values >= self.shape[axis]).any() or (values < -self.shape[axis]).any()
        ):
            raise ValueError("Index out of range")
        values = numpy.where(values < 0, values + self.shape[axis], values).astype(
            numpy.int64
        )
        return values, scalar

    def _index_tuple(self, key: Any) -> tuple[Any, ...]:
        """Expand ellipsis and omitted axes in an indexing key."""
        key = key if isinstance(key, tuple) else (key,)
        if any(item is Ellipsis for item in key):
            if sum(item is Ellipsis for item in key) > 1:
                raise IndexError("Only one ellipsis is allowed")
            position = next(i for i, item in enumerate(key) if item is Ellipsis)
            key = (
                key[:position]
                + (slice(None),) * (self.ndim - len(key) + 1)
                + key[position + 1 :]
            )
        if len(key) > self.ndim:
            raise ValueError("Too many indices")
        return key + (slice(None),) * (self.ndim - len(key))

    def __getitem__(self, key: Any) -> SparseArray | NDArray[Any] | Scalar:
        """Index with integers, slices (including steps), or integer/boolean arrays."""
        key = self._index_tuple(key)
        if all(
            isinstance(index, slice)
            or (
                isinstance(index, (int, numpy.integer))
                and not isinstance(index, (bool, numpy.bool_))
            )
            for index in key
        ):
            axes, shape, dropped = [], [], []
            for axis, index in enumerate(key):
                size = self.shape[axis]
                if isinstance(index, slice):
                    start, stop, step = index.indices(size)
                    count = len(range(start, stop, step))
                    # A step larger than the axis can select at most one value.
                    # Keep descriptors representable in signed native arithmetic.
                    step = max(-size - 1, min(size + 1, step))
                    axes.append((start, step, count))
                    shape.append(count)
                    dropped.append(False)
                else:
                    index = operator.index(index)
                    if index < 0:
                        index += size
                    if not 0 <= index < size:
                        raise ValueError("Index out of range")
                    axes.append((index, 1, 1))
                    dropped.append(True)
            if not shape:
                return self._array.coeff(
                    axes[0][0], axes[1][0] if self.ndim == 2 else 0
                )
            if self.ndim == 1:
                axes.append((0, 1, 1))
                mode = 1
            else:
                mode = 2 if dropped[0] else 1 if dropped[1] else 0
            backend = self._array.slice_basic(*axes[0], *axes[1], mode)
            return self._from_backend(backend, self.dtype, self.order, tuple(shape))
        if self.ndim == 2 and all(
            isinstance(index, numpy.ndarray) and index.dtype.kind == "i"
            for index in key
        ):
            rows, cols = numpy.broadcast_arrays(*key)
            result = self._array.gather_signed(rows.ravel(), cols.ravel())
            return result.reshape(rows.shape)
        parsed = [self._indices(index, axis) for axis, index in enumerate(key)]
        if all(scalar for _, scalar in parsed):
            native = tuple(int(values) for values, _ in parsed)
            return self._array.coeff(native[0], native[1] if self.ndim == 2 else 0)
        if self.ndim == 2 and all(not isinstance(index, slice) for index in key):
            rows, cols = numpy.broadcast_arrays(*(values for values, _ in parsed))
            result = self._array.gather(
                numpy.ascontiguousarray(rows.ravel(), dtype=numpy.int32),
                numpy.ascontiguousarray(cols.ravel(), dtype=numpy.int32),
            )
            return result.reshape(rows.shape)
        selections = [numpy.atleast_1d(values) for values, _ in parsed]
        if any(values.ndim != 1 for values in selections):
            raise IndexError("Mixed slicing supports one-dimensional index arrays")
        selection_shape = tuple(
            len(values) for values, (_, scalar) in zip(selections, parsed) if not scalar
        )
        selection = SparseArray(selection_shape, dtype=self.dtype, order=self.order)
        maps = []
        for values in selections:
            mapping: dict[int, list[int]] = {}
            for destination_index, source in enumerate(values):
                mapping.setdefault(int(source), []).append(destination_index)
            maps.append(mapping)
        from itertools import product

        output_indices: list[list[int]] = [[] for _ in selection_shape]
        output_values = []
        for source, value in zip(zip(*self.stored_indices()), self.values()):
            destinations = [
                mapping.get(int(index), ()) for mapping, index in zip(maps, source)
            ]
            for destination in product(*destinations):
                output_values.append(value)
                target = [
                    index
                    for index, (_, scalar) in zip(destination, parsed)
                    if not scalar
                ]
                for indices, index in zip(output_indices, target):
                    indices.append(index)
        selection._put_coordinates(
            numpy.asarray(output_values, dtype=self.dtype), tuple(output_indices)
        )
        return selection

    def __mul__(self, other: SparseArray | ArrayLike) -> SparseArray:
        """Multiply elementwise, promoting scalar types as NumPy does."""
        if isinstance(other, SparseArray):
            return self.multiply(other)
        if not numpy.isscalar(other):
            return self.multiply(SparseArray(numpy.asarray(other)))
        if not numpy.isfinite(other):
            return self._map_values(lambda values: values * other)
        values = self.values() * other
        result = SparseArray(self.shape, dtype=values.dtype, order=self.order)
        result._put_coordinates(values, self.stored_indices())
        return result

    def __setitem__(self, key: Any, value: ArrayLike) -> None:
        """Assign scalars or broadcastable arrays to basic and advanced indices."""
        key = self._index_tuple(key)
        parsed = [self._indices(index, axis) for axis, index in enumerate(key)]
        if all(scalar for _, scalar in parsed):
            native = tuple(int(values) for values, _ in parsed)
            self._array.set(
                native[0], native[1] if self.ndim == 2 else 0, self.dtype.type(value)
            )
            return
        if any(isinstance(index, slice) for index in key):
            selections = [numpy.atleast_1d(values) for values, _ in parsed]
            coordinates = numpy.meshgrid(*selections, indexing="ij")
            shape = tuple(
                len(values)
                for values, (_, scalar) in zip(selections, parsed)
                if not scalar
            )
        else:
            coordinates = numpy.broadcast_arrays(*(values for values, _ in parsed))
            shape = coordinates[0].shape
        values = numpy.broadcast_to(value, shape).astype(self.dtype).ravel()
        self._put_assignment(values, tuple(index.ravel() for index in coordinates))

    def ceil(self) -> SparseArray:
        """Apply numpy.ceil elementwise, including implicit zeros."""
        return self._map_values(numpy.ceil)

    def clip(
        self,
        min: Scalar | ArrayLike | None = None,
        max: Scalar | ArrayLike | None = None,
        out: NDArray[Any] | None = None,
        **kwargs: Any,
    ) -> SparseArray:
        """Clamp values to bounds as NumPy does, including implicit zeros."""
        if min is None and max is None:
            raise ValueError("At least one clipping bound is required")
        result = self._map_values(lambda values: numpy.clip(values, min, max, **kwargs))
        if out is not None:
            if not isinstance(out, SparseArray) or out.shape != self.shape:
                raise ValueError("out must be a sparse array of the same shape")
            converted = result.astype(out.dtype, order=out.order)
            out._array = converted._array
            return out
        return result

    def cos(self) -> SparseArray:
        """Apply numpy.cos elementwise, including implicit zeros."""
        return self._map_values(numpy.cos)

    def floor(self) -> SparseArray:
        """Apply numpy.floor elementwise, including implicit zeros."""
        return self._map_values(numpy.floor)

    def max(
        self,
        axis: int | tuple[int, ...] | None = None,
        out: NDArray[Any] | None = None,
        keepdims: bool = False,
    ) -> Scalar | NDArray[Any]:
        """Maximum including implicit zeros; empty arrays return NaN."""
        return self._extreme(numpy.maximum, axis, out, keepdims)

    def mean(
        self,
        axis: int | tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        out: NDArray[Any] | None = None,
        keepdims: bool = False,
    ) -> Scalar | NDArray[Any]:
        """Compute means without materializing implicit zeros."""
        axis = self._normalize_axis(axis)
        if dtype is None:
            dtype = (
                self.dtype if self.dtype.kind in "fc" else numpy.dtype(numpy.float64)
            )
        count = self.size if axis is None else self.shape[axis]
        with numpy.errstate(invalid="ignore", divide="ignore"):
            result = numpy.asarray(self.sum(axis=axis, dtype=dtype)) / count
        return self._finish_reduction(result, axis, out, keepdims)

    def min(
        self,
        axis: int | tuple[int, ...] | None = None,
        out: NDArray[Any] | None = None,
        keepdims: bool = False,
    ) -> Scalar | NDArray[Any]:
        """Minimum including implicit zeros; empty arrays return NaN."""
        return self._extreme(numpy.minimum, axis, out, keepdims)

    def power(self, exponent: float) -> SparseArray:
        """Elementwise power; equivalent to ``self ** exponent``."""
        return self**exponent

    def sign(self) -> SparseArray:
        """Apply numpy.sign elementwise, including implicit zeros."""
        return self._map_values(numpy.sign)

    def sin(self) -> SparseArray:
        """Apply numpy.sin elementwise, including implicit zeros."""
        return self._map_values(numpy.sin)

    def std(
        self,
        axis: int | tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        out: NDArray[Any] | None = None,
        ddof: float = 0,
        keepdims: bool = False,
    ) -> Scalar | NDArray[Any]:
        """Standard deviation with NumPy-compatible reduction arguments."""
        result = numpy.sqrt(
            self.var(axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims)
        )
        if out is not None:
            numpy.copyto(out, result, casting="same_kind")
            return out
        return result

    def sum(
        self,
        axis: int | tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        out: NDArray[Any] | None = None,
        keepdims: bool = False,
    ) -> Scalar | NDArray[Any]:
        """Sum values, including NumPy's axis/dtype/out/keepdims conventions."""
        axis = self._normalize_axis(axis)
        expected_dtype = (
            numpy.dtype(numpy.int64) if self.dtype.kind == "i" else self.dtype
        )
        if dtype is None or numpy.dtype(dtype) == expected_dtype:
            result = self._array.sum(-1 if axis is None or self.ndim == 1 else axis)
            if axis is None or self.ndim == 1:
                result = result[0]
            return self._finish_reduction(result, axis, out, keepdims)
        values = self.values()
        if axis is None or self.ndim == 1:
            result = values.sum(dtype=dtype)
        else:
            target = 1 - axis
            result = numpy.zeros(
                self.shape[target],
                dtype=numpy.empty(0, self.dtype).sum(dtype=dtype).dtype,
            )
            numpy.add.at(result, self.stored_indices()[target], values)
        return self._finish_reduction(result, axis, out, keepdims)

    def trace(
        self,
        offset: int = 0,
        axis1: int = 0,
        axis2: int = 1,
        dtype: DTypeLike | None = None,
        out: NDArray[Any] | None = None,
    ) -> Scalar | NDArray[Any]:
        """Sum a diagonal, with NumPy's trace arguments."""
        return self.diagonal(offset, axis1, axis2).sum(dtype=dtype, out=out)

    def var(
        self,
        axis: int | tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        out: NDArray[Any] | None = None,
        ddof: float = 0,
        keepdims: bool = False,
    ) -> Scalar | NDArray[Any]:
        """Variance including implicit zeros, using a stable centered sum."""
        axis = self._normalize_axis(axis)
        if dtype is None:
            dtype = (
                self.dtype if self.dtype.kind in "fc" else numpy.dtype(numpy.float64)
            )
        mean_dtype = dtype
        result_dtype = (
            numpy.empty((), dtype).real.dtype if self.dtype.kind == "c" else dtype
        )
        values = self.values().astype(mean_dtype)
        mean = numpy.asarray(self.mean(axis=axis, dtype=mean_dtype))
        count = self.size if axis is None else self.shape[axis]
        if axis is None or self.ndim == 1:
            result = numpy.square(numpy.abs(values - mean)).sum(dtype=result_dtype)
            result += (count - len(values)) * numpy.square(numpy.abs(mean))
        else:
            indices = self.stored_indices()[1 - axis]
            result = numpy.zeros(self.shape[1 - axis], dtype=result_dtype)
            numpy.add.at(
                result, indices, numpy.square(numpy.abs(values - mean[indices]))
            )
            result += (
                count - numpy.bincount(indices, minlength=len(result))
            ) * numpy.square(numpy.abs(mean))
        with numpy.errstate(invalid="ignore", divide="ignore"):
            result = numpy.asarray(result) / max(count - ddof, 0)
        return self._finish_reduction(result, axis, out, keepdims)

    def _normalize_axis(self, axis: int | tuple[int, ...] | None) -> int | None:
        """Validate reduction axes and normalize negative indices."""
        if axis is None:
            return None
        if isinstance(axis, tuple):
            if any(item is None or isinstance(item, tuple) for item in axis):
                raise ValueError("Axes must be integers")
            normalized = tuple(self._normalize_axis(item) for item in axis)
            if len(set(normalized)) != len(normalized):
                raise ValueError("Duplicate axes are not allowed")
            if len(normalized) == self.ndim:
                return None
            if len(normalized) != 1:
                raise ValueError("Expected one axis or all axes")
            return normalized[0]
        if (
            not isinstance(axis, (int, numpy.integer))
            or not -self.ndim <= axis < self.ndim
        ):
            raise ValueError("Axis out of range")
        return axis % self.ndim

    def _finish_reduction(
        self,
        result: Any,
        axis: int | tuple[int, ...] | None,
        out: NDArray[Any] | None,
        keepdims: bool,
    ) -> Scalar | NDArray[Any]:
        """Apply keepdims and out conventions to a reduction result."""
        result = numpy.asarray(result)
        if keepdims:
            shape = (
                [1] * self.ndim
                if axis is None
                else [1 if i == axis else n for i, n in enumerate(self.shape)]
            )
            result = result.reshape(shape)
        if out is not None:
            numpy.copyto(out, result, casting="same_kind")
            return out
        return result[()] if result.ndim == 0 else result

    def _extreme(
        self,
        operation: numpy.ufunc,
        axis: int | tuple[int, ...] | None,
        out: NDArray[Any] | None,
        keepdims: bool,
    ) -> Scalar | NDArray[Any]:
        """Reduce extrema while accounting for implicit zeros."""
        axis = self._normalize_axis(axis)
        values = self.values()
        count = self.size if axis is None else self.shape[axis]
        if count == 0:
            result = (
                numpy.nan
                if axis is None or self.ndim == 1
                else numpy.full(self.shape[1 - axis], numpy.nan)
            )
        elif axis is None or self.ndim == 1:
            result = operation.reduce(values) if len(values) else self.dtype.type(0)
            if len(values) < count:
                result = operation(result, 0)
        else:
            identity: Scalar = numpy.inf if operation is numpy.minimum else -numpy.inf
            if self.dtype.kind == "i":
                info = numpy.iinfo(self.dtype)
                identity = info.max if operation is numpy.minimum else info.min
            elif self.dtype.kind == "c":
                # Complex extrema compare real, then imaginary components.
                identity = complex(identity, identity)
            result = numpy.full(self.shape[1 - axis], identity, dtype=self.dtype)
            indices = self.stored_indices()[1 - axis]
            operation.at(result, indices, values)
            mask = numpy.bincount(indices, minlength=len(result)) < count
            result[mask] = operation(result[mask], 0)
        return self._finish_reduction(result, axis, out, keepdims)

    def __pow__(self, exponent: float) -> SparseArray:
        """Raise stored values to a nonnegative scalar power."""
        if exponent == 0:
            from sppy.util import ones

            return ones(self.shape, dtype=self.dtype, order=self.order)
        if exponent < 0:
            raise ValueError("Negative sparse powers would change implicit zeros")
        values = self.values() ** exponent
        result = SparseArray(self.shape, dtype=values.dtype, order=self.order)
        result._put_coordinates(values, self.stored_indices())
        return result

    def _map_values(self, operation: Callable[..., Any]) -> SparseArray:
        """Transform stored values and materialize transformed implicit zeros."""
        values = numpy.asarray(operation(self.values()))
        zero = numpy.asarray(operation(numpy.zeros(1, dtype=self.dtype)))[0]
        result = SparseArray(self.shape, dtype=values.dtype, order=self.order)
        if zero != 0:
            result.fill(zero)
            result._put_assignment(values, self.stored_indices())
        else:
            mask = values != 0
            result._put_coordinates(
                values[mask], tuple(index[mask] for index in self.stored_indices())
            )
        return result

    def __array__(
        self, dtype: DTypeLike | None = None, copy: bool | None = None
    ) -> NDArray[Any]:
        """Explicit conversion to a dense NumPy array always allocates."""
        if copy is False:
            raise ValueError("Sparse-to-dense conversion requires a copy")
        result = self.toarray()
        return result if dtype is None else result.astype(dtype, copy=False)

    def __len__(self) -> int:
        """Return the length of the first logical axis."""
        return self.shape[0]

    def diagonal(self, offset: int = 0, axis1: int = 0, axis2: int = 1) -> NDArray[Any]:
        """Extract a diagonal as a dense vector, without densifying the matrix."""
        offset = operator.index(offset)
        axis1, axis2 = operator.index(axis1), operator.index(axis2)
        if (
            self.ndim != 2
            or not -2 <= axis1 < 2
            or not -2 <= axis2 < 2
            or axis1 % 2 == axis2 % 2
        ):
            raise ValueError("diagonal requires two distinct matrix axes")
        if axis1 % 2 == 1:
            return self.T.diagonal(offset)
        length = max(
            0, min(self.shape[0] + min(offset, 0), self.shape[1] - max(offset, 0))
        )
        rows = numpy.arange(length) + max(-offset, 0)
        cols = numpy.arange(length) + max(offset, 0)
        return cast(NDArray[Any], self[rows, cols])

    def subarray(
        self, row_indices: ArrayLike, column_indices: ArrayLike | None = None
    ) -> SparseArray:
        """Take the Cartesian product of index vectors, like NumPy's ix_."""
        if self.ndim == 1:
            if column_indices is not None:
                raise ValueError("Vectors have one axis")
            return cast(SparseArray, self[row_indices])
        rows = cast(SparseArray, self[numpy.asarray(row_indices), :])
        return cast(SparseArray, rows[:, numpy.asarray(column_indices)])

    def count_nonzero(
        self, axis: int | tuple[int, ...] | None = None
    ) -> int | NDArray[numpy.int64]:
        """Count nonzero values, excluding explicitly stored zeros."""
        axis = self._normalize_axis(axis)
        mask = self.values() != 0
        if axis is None or self.ndim == 1:
            return int(numpy.count_nonzero(mask))
        return numpy.bincount(
            self.stored_indices()[1 - axis][mask], minlength=self.shape[1 - axis]
        )

    __rmul__ = __mul__
