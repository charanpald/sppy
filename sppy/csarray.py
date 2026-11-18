"""Public NumPy-style interface to the typed Eigen sparse kernels."""

from sppy.csarray_sub import (
    csarray_int_colMajor,
    csarray_double_colMajor,
    csarray_float_colMajor,
    csarray_int64_t_colMajor,
    csarray_short_colMajor,
    csarray_signed_char_colMajor,
)
from sppy.csarray_sub import (
    csarray_int_rowMajor,
    csarray_double_rowMajor,
    csarray_float_rowMajor,
    csarray_int64_t_rowMajor,
    csarray_short_rowMajor,
    csarray_signed_char_rowMajor,
)
from sppy.csarray1d_sub import (
    csarray1d_int,
    csarray1d_double,
    csarray1d_float,
    csarray1d_int64_t,
    csarray1d_short,
    csarray1d_signed_char,
)
import array
import numpy


class SparseArray:
    def __init__(self, S, dtype=None, storagetype="col", *, order=None):
        """
        Create a sparse vector or matrix from a shape or existing data.

        :param S: A shape (integer or tuple), dense array, SciPy sparse object, or SparseArray.

        :param dtype: A supported NumPy dtype; inferred from data or float64 for a shape.

        :param storagetype: One of "row" or "col"; defaults to column storage.

        :param order: "C" for row or "F" for column storage; overrides storagetype.
        """
        if order is not None:
            if order not in {"C", "F"}:
                raise ValueError("order must be 'C' or 'F'")
            storagetype = "row" if order == "C" else "col"
        if not isinstance(S, (tuple, int)) and not hasattr(S, "shape"):
            S = numpy.asarray(S)
        if dtype is None:
            dtype = getattr(S, "dtype", numpy.float64)
        self._set_object(S, dtype, storagetype)

    def _set_object(self, S, dtype=numpy.float64, storagetype="col"):
        dtype = numpy.dtype(dtype)
        if storagetype not in {"row", "col"}:
            raise ValueError("Unknown storage type: " + str(storagetype))
        if type(S) == tuple:
            shape = S
        elif type(S) == int:
            shape = (S,)
        else:
            try:
                shape = S.shape
            except AttributeError:
                raise ValueError("Invalid parameter: " + str(S))

        if any(
            not isinstance(n, (int, numpy.integer))
            or n < 0
            or n > numpy.iinfo(numpy.int32).max
            for n in shape
        ):
            raise ValueError("Dimensions must be nonnegative 32-bit integers")
        if len(shape) not in {1, 2}:
            raise ValueError("Only one- and two-dimensional arrays are supported")
        scalar_names = {
            numpy.dtype(numpy.int8): "signed_char",
            numpy.dtype(numpy.int16): "short",
            numpy.dtype(numpy.int32): "int",
            numpy.dtype(numpy.int64): "int64_t",
            numpy.dtype(numpy.float32): "float",
            numpy.dtype(numpy.float64): "double",
        }
        if dtype not in scalar_names:
            raise ValueError(f"Unsupported dtype: {dtype}")
        scalar = scalar_names[dtype]
        if len(shape) == 1:
            backend = globals()[f"csarray1d_{scalar}"]
        else:
            storage = "rowMajor" if storagetype == "row" else "colMajor"
            backend = globals()[f"csarray_{scalar}_{storage}"]
        self._array = backend(shape)

        if isinstance(S, SparseArray):
            self._dtype = dtype
            self.storagetype = storagetype
            self._put_coordinates(S.values().astype(dtype), S.nonzero())
            return
        if hasattr(S, "tocoo"):
            coordinates = S.tocoo(copy=True)
            coordinates.sum_duplicates()
            self._array.put(
                numpy.ascontiguousarray(coordinates.data, dtype=dtype),
                numpy.ascontiguousarray(coordinates.row, dtype=numpy.int32),
                numpy.ascontiguousarray(coordinates.col, dtype=numpy.int32),
                True,
            )
            self._dtype = dtype
            self.storagetype = storagetype
            return
        try:
            nonzeros = S.nonzero()
            if len(shape) == 2:
                row_indices, colInds = nonzeros
                row_indices = numpy.array(row_indices, numpy.int32)
                colInds = numpy.array(colInds, numpy.int32)

                self._array.put(
                    numpy.array(S[S.nonzero()], dtype=dtype).flatten(),
                    row_indices,
                    colInds,
                    True,
                )
            elif len(shape) == 1:
                self._array[nonzeros[0]] = S[nonzeros[0]]
        except AttributeError:
            pass

        self._dtype = dtype
        self.storagetype = storagetype

    def __abs__(self):
        """
        Compute the absolute value of the elements of this matrix.
        """
        resultArray = self._array.__abs__()
        result = csarray(resultArray.shape, self.dtype, self.storagetype)
        result._array = resultArray
        return result

    def __add__(self, other):
        return self._binary_sparse(other, "__add__")

    def _set_backend(self, array, dtype):
        """
        Convert a base class to csarray.
        """
        del self._array
        self._array = array
        self._dtype = dtype

    def __getattr__(self, name):
        array = object.__getattribute__(self, "_array")
        return getattr(array, name)

    def _get_dtype(self):
        return numpy.dtype(self._dtype)

    def _indices(self, index, axis):
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

    def _index_tuple(self, key):
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

    def __getitem__(self, key):
        """Index with integers, slices (including steps), or integer/boolean arrays."""
        key = self._index_tuple(key)
        parsed = [self._indices(index, axis) for axis, index in enumerate(key)]
        if all(scalar for _, scalar in parsed):
            native = tuple(int(values) for values, _ in parsed)
            return self._array[native if self.ndim == 2 else native[0]]
        if self.ndim == 2 and all(not isinstance(index, slice) for index in key):
            rows, cols = numpy.broadcast_arrays(*(values for values, _ in parsed))
            result = self._array[
                numpy.ascontiguousarray(rows.ravel()),
                numpy.ascontiguousarray(cols.ravel()),
            ]
            return result.reshape(rows.shape)
        selections = [numpy.atleast_1d(values) for values, _ in parsed]
        if any(values.ndim != 1 for values in selections):
            raise IndexError("Mixed slicing supports one-dimensional index arrays")
        shape = tuple(
            len(values) for values, (_, scalar) in zip(selections, parsed) if not scalar
        )
        result = SparseArray(shape, dtype=self.dtype, storagetype=self.storagetype)
        maps = []
        for values in selections:
            mapping = {}
            for destination, source in enumerate(values):
                mapping.setdefault(int(source), []).append(destination)
            maps.append(mapping)
        from itertools import product

        output_indices = [[] for _ in shape]
        output_values = []
        for source, value in zip(zip(*self.nonzero()), self.values()):
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
        result._put_coordinates(
            numpy.asarray(output_values, dtype=self.dtype), output_indices
        )
        return result

    def __getstate__(self):
        """
        Used for pickling.
        """
        objDict = {}
        objDict["shape"] = self.shape
        objDict["storagetype"] = self.storagetype
        objDict["dtype"] = self.dtype
        if self.ndim == 2:
            objDict["rowInds"], objDict["colInds"] = self.nonzero()
        else:
            objDict["rowInds"] = self.nonzero()[0]
        objDict["values"] = self.values()
        return objDict

    def __mul__(self, other):
        """Multiply elementwise, promoting scalar types as NumPy does."""
        if isinstance(other, SparseArray):
            return self.multiply(other)
        if not numpy.isscalar(other):
            return self.multiply(SparseArray(numpy.asarray(other)))
        values = self.values() * other
        result = SparseArray(
            self.shape, dtype=values.dtype, storagetype=self.storagetype
        )
        result._put_coordinates(values, self.nonzero())
        return result

    __rmul__ = __mul__

    def __neg__(self):
        """
        Negate all the elements of this matrix.
        """
        resultArray = self._array.__neg__()
        result = csarray(resultArray.shape, self.dtype, self.storagetype)
        result._array = resultArray
        return result

    def __setitem__(self, key, value):
        """Assign scalars or broadcastable arrays to basic and advanced indices."""
        key = self._index_tuple(key)
        parsed = [self._indices(index, axis) for axis, index in enumerate(key)]
        if all(scalar for _, scalar in parsed):
            native = tuple(int(values) for values, _ in parsed)
            self._array[native if self.ndim == 2 else native[0]] = value
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

    def _put_assignment(self, values, coordinates):
        if self.ndim == 1:
            self._array.put(
                values, numpy.ascontiguousarray(coordinates[0], dtype=numpy.int64)
            )
        else:
            self._array.put(
                values,
                *(
                    numpy.ascontiguousarray(index, dtype=numpy.int32)
                    for index in coordinates
                ),
            )

    def __setstate__(self, objDict):
        self._set_object(objDict["shape"], objDict["dtype"], objDict["storagetype"])

        if len(objDict["shape"]) == 2:
            row_indices = numpy.array(objDict["rowInds"], numpy.int32)
            colInds = numpy.array(objDict["colInds"], numpy.int32)

            self._array.put(objDict["values"], row_indices, colInds, True)
            del objDict["colInds"]
        elif len(objDict["shape"]) == 1:
            self._array[objDict["rowInds"]] = objDict["values"]

        del objDict["rowInds"]
        del objDict["values"]

    def __str__(self):
        """
        Return a string representation of the non-zero elements of the array.
        """
        if self.ndim == 2:
            outputStr = (
                "csarray dtype:"
                + str(numpy.dtype(self.dtype))
                + " shape:"
                + str(self.shape)
                + " non-zeros:"
                + str(self.getnnz())
                + " storage:"
                + str(self.storagetype)
                + "\n"
            )
            (row_indices, colInds) = self.nonzero()
            vals = self.values()

            for i in range(self.getnnz()):
                outputStr += (
                    "("
                    + str(row_indices[i])
                    + ", "
                    + str(colInds[i])
                    + ")"
                    + " "
                    + str(vals[i])
                )
                if i != self.getnnz() - 1:
                    outputStr += "\n"
        else:
            outputStr = (
                "csarray dtype:"
                + str(numpy.dtype(self.dtype))
                + " shape:"
                + str(self.shape)
                + " non-zeros:"
                + str(self.getnnz())
                + "\n"
            )
            inds = self.nonzero()[0]
            vals = self[inds]

            for i in range(self.getnnz()):
                outputStr += "(" + str(inds[i]) + ")" + " " + str(vals[i])
                if i != self.getnnz() - 1:
                    outputStr += "\n"

        return outputStr

    def __sub__(self, other):
        return self._binary_sparse(other, "__sub__")

    def ceil(self):
        """Apply numpy.ceil elementwise, including implicit zeros."""
        return self._map_values(numpy.ceil)

    def clip(self, min=None, max=None, out=None, **kwargs):
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

    def compress(self):
        """
        Turn this matrix into compressed sparse format by freeing extra memory
        space in the buffer.
        """
        self._array.compress()

    def copy(self, order="K"):
        """Return an independent copy, optionally changing sparse storage order."""
        if order not in {"K", "C", "F"}:
            raise ValueError("order must be 'K', 'C', or 'F'")
        storage = (
            self.storagetype if order == "K" else ("row" if order == "C" else "col")
        )
        return SparseArray(self, dtype=self.dtype, storagetype=storage)

    def cos(self):
        """Apply numpy.cos elementwise, including implicit zeros."""
        return self._map_values(numpy.cos)

    def diag(self):
        """
        Return a numpy array containing the diagonal entries of this matrix. If
        the matrix is non-square then the diagonal array is the same size as the
        smallest dimension.
        """
        return self._array.diag()

    def dot(self, other):
        """Matrix/vector product; equivalent to ``self @ other``."""
        if isinstance(other, SparseArray):
            dtype = numpy.result_type(self.dtype, other.dtype)
            left = self.astype(dtype, copy=False)
            right = other.astype(dtype, copy=False)
            if right.storagetype != left.storagetype:
                right = SparseArray(right, dtype=dtype, storagetype=left.storagetype)
            if self.ndim == 1:
                if other.ndim != 1:
                    return other.T.dot(self.toarray())
                return left._array.dotCsarray1d(right._array)
            if other.ndim != 2:
                return left.dot(right.toarray())
            return left._wrap_backend(left._array.dotCsarray2d(right._array))
        dense = numpy.ascontiguousarray(other, dtype=numpy.float64)
        if self.ndim == 1:
            return self.toarray().dot(dense)
        if dense.ndim not in {1, 2} or dense.shape[0] != self.shape[1]:
            raise ValueError("Incompatible matrix product dimensions")
        result_dtype = numpy.result_type(self.dtype, numpy.asarray(other).dtype)
        if result_dtype.kind in "iu":
            dense = numpy.asarray(other, dtype=result_dtype)
            rows, cols = self.nonzero()
            values = self.values().astype(result_dtype)
            result = numpy.zeros((self.shape[0],) + dense.shape[1:], dtype=result_dtype)
            if dense.ndim == 1:
                numpy.add.at(result, rows, values * dense[cols])
            else:
                for column in range(dense.shape[1]):
                    numpy.add.at(result[:, column], rows, values * dense[cols, column])
            return result
        result = (
            self._array.dotNumpy1d(dense)
            if dense.ndim == 1
            else self._array.dotNumpy2d(dense)
        )
        return result.astype(result_dtype, copy=False)
        raise ValueError("dot requires a one- or two-dimensional operand")

    def __matmul__(self, other):
        return self.dot(other)

    def __rmatmul__(self, other):
        dense = numpy.asarray(other)
        if dense.ndim == 1:
            return self.T.dot(dense)
        return self.T.dot(dense.T).T

    def floor(self):
        """Apply numpy.floor elementwise, including implicit zeros."""
        return self._map_values(numpy.floor)

    def getnnz(self):
        """
        Return the number of stored entries, including explicit zeros.
        """
        return self._array.getnnz()

    def multiply(self, other):
        """Return an elementwise sparse product of equally shaped arrays."""
        if numpy.isscalar(other):
            return self * other
        if not isinstance(other, SparseArray):
            other = SparseArray(numpy.asarray(other))
        if self.shape != other.shape:
            raise ValueError("Elementwise operands must have the same shape")
        dtype = numpy.result_type(self.dtype, other.dtype)
        left = self.astype(dtype, copy=False)
        right = SparseArray(other, dtype=dtype, storagetype=left.storagetype)
        return left._wrap_backend(left._array.hadamard(right._array))

    hadamard = multiply

    def max(self, axis=None, out=None, keepdims=False):
        """Maximum including implicit zeros; empty legacy arrays return NaN."""
        return self._extreme(numpy.maximum, axis, out, keepdims)

    def mean(self, axis=None, dtype=None, out=None, keepdims=False):
        """Compute means without materializing implicit zeros."""
        axis = self._normalize_axis(axis)
        if dtype is None:
            dtype = self.dtype if self.dtype.kind == "f" else numpy.dtype(numpy.float64)
        count = self.size if axis is None else self.shape[axis]
        with numpy.errstate(invalid="ignore", divide="ignore"):
            result = numpy.asarray(self.sum(axis=axis, dtype=dtype)) / count
        return self._finish_reduction(result, axis, out, keepdims)

    def min(self, axis=None, out=None, keepdims=False):
        """Minimum including implicit zeros; empty legacy arrays return NaN."""
        return self._extreme(numpy.minimum, axis, out, keepdims)

    def nonzero(self):
        """
        Return a tuple of arrays corresponding to nonzero elements.
        """
        return self._array.nonzero()

    def nonzero_rows_list(self):
        """
        Return a list such that the ith element is an array of nonzero elements
        in the ith row of this matrix.
        """
        row_indices = numpy.zeros(self.nnz, dtype=numpy.int32)
        colInds = numpy.zeros(self.nnz, dtype=numpy.int32)
        omegaList = []
        row_indices, colInds = self._array.nonzero()

        for i in range(self.shape[0]):
            omegaList.append(array.array("I"))

        for i in range(self.nnz):
            omegaList[row_indices[i]].append(colInds[i])

        for i in range(self.shape[0]):
            omegaList[i] = numpy.array(omegaList[i], numpy.uint)

        return omegaList

    def nonzero_rows_ptr(self):
        """
        Returns two arrays indPtr, colInds, such that colInds[indPtr[i]:indPtr[i+1]]
        is the set of nonzero elements in the ith row of this matrix.
        """
        row_indices = numpy.zeros(self.nnz, dtype=numpy.int32)
        colInds = numpy.zeros(self.nnz, dtype=numpy.int32)

        result = csarray(self, storagetype="row")
        row_indices, colInds = result._array.nonzero()
        indPtr = numpy.cumsum(
            numpy.bincount(row_indices, minlength=self.shape[0]), dtype=numpy.int32
        )
        indPtr = numpy.array(numpy.r_[numpy.array([0]), indPtr], numpy.int32)

        return indPtr, colInds

    def pdot(self, A):
        """
        Compute the dot product between this and either a csarray or numpy array
        using multithreading.

        :param A: The input numpy array or csarray.
        """
        if isinstance(A, numpy.ndarray):
            if A.ndim == 2:
                result = self._array.pdot2d(A)
            else:
                result = self._array.pdot1d(A)
        else:
            raise ValueError("Cannot pdot with A of type " + str(type(A)))

        return result

    def power(self, exponent):
        """Elementwise power; equivalent to ``self ** exponent``."""
        return self**exponent

    def prune(self, eps=10**-10, precision=10**-20):
        """
        Suppresses all nonzeros which are much smaller in magnitude than eps under
        the tolerence precision.
        """
        self._array.prune(eps, precision)

    def put(self, data, row_indices, colInds, init=False):
        """
        Put some values into this matrix into the corresponding row_indices and colInds. We have
        A[row_indices[i], colInds[i]] = data[i]/data. Notice that this is faster if init=True
        but this setting is to be used only if the matrix has just been created.

        :param data: Values to assign at the paired row and column indices.

        :param row_indices: A 1d numpy array of row indices.

        :param colInds: A 1d numpy array of column indices.
        """
        self._array.put(data, row_indices, colInds, init)

    def reserve(self, n):
        """
        Reserve n nonzero entries and turns the matrix into uncompressed mode.

        :param n: The number of elements of space to reserve.
        :type n: `int`
        """
        self._array.reserve(n)

    def row_indices(self, i):
        """
        Returns the non zero indices for the ith row.

        :param i: The index of the row of the array.
        :type i: `int`
        """
        return self._array.rowInds(i)

    def sign(self):
        """Apply numpy.sign elementwise, including implicit zeros."""
        return self._map_values(numpy.sign)

    def sin(self):
        """Apply numpy.sin elementwise, including implicit zeros."""
        return self._map_values(numpy.sin)

    def std(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
        """Standard deviation with NumPy-compatible reduction arguments."""
        result = numpy.sqrt(
            self.var(axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims)
        )
        if out is not None:
            numpy.copyto(out, result, casting="same_kind")
            return out
        return result

    def submatrix(self, startRow, startCol, blockRows, blockCols):
        """
        Return a submatrix of the matrix given by A[startRow:startRows+blockRows, startCol:startCol+blockCols]
        in an efficient manner.

        :param startRow: The starting row index
        :type startRow: `int`

        :param startCol: The starting column index
        :type startCol: `int`

        :param blockRows: The number of rows to take.
        :type blockRows: `int`

        :param blockCols: The number of columns to take.
        :type blockCols: `int`
        """
        if (
            startRow < 0
            or startRow > self.shape[0]
            or startCol < 0
            or startCol > self.shape[1]
        ):
            raise ValueError("Invalid start row or column index")
        if (
            startRow + blockRows < 0
            or startRow + blockRows > self.shape[0]
            or startCol + blockCols < 0
            or startCol + blockCols > self.shape[1]
        ):
            raise ValueError("Invalid end row or column index")

        result = csarray((blockRows, blockCols), self.dtype)
        result._array = self._array.submatrix(startRow, startCol, blockRows, blockCols)
        return result

    def sum(self, axis=None, dtype=None, out=None, keepdims=False):
        """Sum values, including NumPy's axis/dtype/out/keepdims conventions."""
        axis = self._normalize_axis(axis)
        values = self.values()
        if axis is None or self.ndim == 1:
            result = values.sum(dtype=dtype)
        else:
            target = 1 - axis
            result = numpy.zeros(
                self.shape[target],
                dtype=numpy.empty(0, self.dtype).sum(dtype=dtype).dtype,
            )
            numpy.add.at(result, self.nonzero()[target], values)
        return self._finish_reduction(result, axis, out, keepdims)

    def toarray(self):
        """
        Convert this sparse array into a numpy array.
        """
        return self._array.toarray()

    def to_scipy_csc(self):
        """
        Convert this matrix to a scipy sparse matrix. Returns a copy of the data in
        csc_matrix form.
        """
        try:
            import scipy.sparse
        except ImportError:
            raise

        row_indices, colInds = self.nonzero()
        values = self.values()
        A = scipy.sparse.csc_matrix((values, (row_indices, colInds)), shape=self.shape)
        return A

    def to_scipy_csr(self):
        """
        Convert this matrix to a scipy sparse matrix. Returns a copy of the data in
        csr_matrix form.
        """
        try:
            import scipy.sparse
        except ImportError:
            raise

        row_indices, colInds = self.nonzero()
        values = self.values()
        A = scipy.sparse.csr_matrix((values, (row_indices, colInds)), shape=self.shape)
        return A

    def trace(self, offset=0, axis1=0, axis2=1, dtype=None, out=None):
        """Sum a diagonal, with NumPy's trace arguments."""
        return self.diagonal(offset, axis1, axis2).sum(dtype=dtype, out=out)

    def transpose(self, *axes):
        """Return a copy with axes reversed, following NumPy's transpose convention."""
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        if axes:
            axes = tuple(axis + self.ndim if axis < 0 else axis for axis in axes)
            if axes == tuple(range(self.ndim)):
                return self.copy()
            if axes != tuple(reversed(range(self.ndim))):
                raise ValueError("Invalid transpose axes")
        if self.ndim == 1:
            return self.copy()
        return self._wrap_backend(self._array.transpose())

    def values(self):
        """
        Return the values of this object according to the elements returned
        using nonzero.
        """
        return self._array.values()

    def var(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
        """Variance including implicit zeros, using a stable centered sum."""
        axis = self._normalize_axis(axis)
        if dtype is None:
            dtype = self.dtype if self.dtype.kind == "f" else numpy.dtype(numpy.float64)
        values = self.values().astype(dtype)
        mean = self.mean(axis=axis, dtype=dtype)
        count = self.size if axis is None else self.shape[axis]
        if axis is None or self.ndim == 1:
            result = numpy.square(values - mean).sum(dtype=dtype)
            result += (count - len(values)) * numpy.square(mean)
        else:
            indices = self.nonzero()[1 - axis]
            result = numpy.zeros(self.shape[1 - axis], dtype=dtype)
            numpy.add.at(result, indices, numpy.square(values - mean[indices]))
            result += (
                count - numpy.bincount(indices, minlength=len(result))
            ) * numpy.square(mean)
        with numpy.errstate(invalid="ignore", divide="ignore"):
            result = numpy.asarray(result) / max(count - ddof, 0)
        return self._finish_reduction(result, axis, out, keepdims)

    def _normalize_axis(self, axis):
        if axis is None:
            return None
        if isinstance(axis, tuple):
            if len(axis) == self.ndim and set(axis) == set(range(self.ndim)):
                return None
            if len(axis) != 1:
                raise ValueError("Expected one axis or all axes")
            axis = axis[0]
        if (
            not isinstance(axis, (int, numpy.integer))
            or not -self.ndim <= axis < self.ndim
        ):
            raise ValueError("Axis out of range")
        return axis % self.ndim

    def _finish_reduction(self, result, axis, out, keepdims):
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

    def _extreme(self, operation, axis, out, keepdims):
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
            identity = numpy.inf if operation is numpy.minimum else -numpy.inf
            if self.dtype.kind == "i":
                info = numpy.iinfo(self.dtype)
                identity = info.max if operation is numpy.minimum else info.min
            result = numpy.full(self.shape[1 - axis], identity, dtype=self.dtype)
            indices = self.nonzero()[1 - axis]
            operation.at(result, indices, values)
            mask = numpy.bincount(indices, minlength=len(result)) < count
            result[mask] = operation(result[mask], 0)
        return self._finish_reduction(result, axis, out, keepdims)

    def _binary_sparse(self, other, operation):
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
        dtype = numpy.result_type(self.dtype, other.dtype)
        left = self.astype(dtype, copy=False)
        right = SparseArray(other, dtype=dtype, storagetype=left.storagetype)
        return left._wrap_backend(getattr(left._array, operation)(right._array))

    __radd__ = __add__

    def __rsub__(self, other):
        return -self + other

    def __pow__(self, exponent):
        if exponent == 0:
            from sppy.util import ones

            return ones(self.shape, dtype=self.dtype, order=self.order)
        if exponent < 0:
            raise ValueError("Negative sparse powers would change implicit zeros")
        values = self.values() ** exponent
        result = SparseArray(
            self.shape, dtype=values.dtype, storagetype=self.storagetype
        )
        result._put_coordinates(values, self.nonzero())
        return result

    def _map_values(self, operation):
        values = numpy.asarray(operation(self.values()))
        zero = numpy.asarray(operation(numpy.zeros(1, dtype=self.dtype)))[0]
        result = SparseArray(
            self.shape, dtype=values.dtype, storagetype=self.storagetype
        )
        if zero != 0:
            from sppy.util import ones

            result = ones(self.shape, dtype=values.dtype, order=self.order) * zero
            result._put_assignment(values, self.nonzero())
        else:
            mask = values != 0
            result._put_coordinates(
                values[mask], tuple(index[mask] for index in self.nonzero())
            )
        return result

    def _wrap_backend(self, backend):
        result = object.__new__(SparseArray)
        result._array = backend
        result._dtype = numpy.dtype(backend.dtype())
        result.storagetype = self.storagetype
        return result

    def _put_coordinates(self, values, coordinates):
        if self.ndim == 1:
            self._array.put(
                values, numpy.ascontiguousarray(coordinates[0], dtype=numpy.int64)
            )
        else:
            self._array.put(
                values,
                *(
                    numpy.ascontiguousarray(index, dtype=numpy.int32)
                    for index in coordinates
                ),
                True,
            )

    def astype(self, dtype, order="K", casting="unsafe", subok=True, copy=True):
        """Cast stored values without allocating a dense array."""
        dtype = numpy.dtype(dtype)
        if not numpy.can_cast(self.dtype, dtype, casting=casting):
            raise TypeError(
                f"Cannot cast {self.dtype} to {dtype} with {casting!r} casting"
            )
        if order not in {"K", "C", "F", "A"}:
            raise ValueError("Invalid order")
        storage = (
            self.storagetype
            if order in {"K", "A"}
            else ("row" if order == "C" else "col")
        )
        if not copy and dtype == self.dtype and storage == self.storagetype:
            return self
        result = SparseArray(self.shape, dtype=dtype, storagetype=storage)
        result._put_coordinates(
            self.values().astype(dtype, casting=casting), self.nonzero()
        )
        return result

    def __array__(self, dtype=None, copy=None):
        """Explicit conversion to a dense NumPy array always allocates."""
        if copy is False:
            raise ValueError("Sparse-to-dense conversion requires a copy")
        result = self.toarray()
        return result if dtype is None else result.astype(dtype, copy=False)

    def __len__(self):
        return self.shape[0]

    @property
    def order(self):
        """Sparse storage traversal order: 'C' for rows, 'F' for columns."""
        return "C" if self.storagetype == "row" else "F"

    @property
    def shape(self):
        return self._array.shape

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return self._array.size

    @property
    def nnz(self):
        return self._array.getnnz()

    def diagonal(self, offset=0, axis1=0, axis2=1):
        """Extract a diagonal as a dense vector, without densifying the matrix."""
        if self.ndim != 2 or (axis1 % 2, axis2 % 2) not in {(0, 1), (1, 0)}:
            raise ValueError("diagonal requires two distinct matrix axes")
        if axis1 % 2 == 1:
            return self.T.diagonal(offset)
        length = max(
            0, min(self.shape[0] + min(offset, 0), self.shape[1] - max(offset, 0))
        )
        rows = numpy.arange(length) + max(-offset, 0)
        cols = numpy.arange(length) + max(offset, 0)
        return self[rows, cols]

    def subarray(self, row_indices, column_indices=None):
        """Take the Cartesian product of index vectors, like NumPy's ix_."""
        if self.ndim == 1:
            if column_indices is not None:
                raise ValueError("Vectors have one axis")
            return self[row_indices]
        return self[numpy.asarray(row_indices), :][:, numpy.asarray(column_indices)]

    subArray = subarray
    nonzeroRowsList = nonzero_rows_list
    nonzeroRowsPtr = nonzero_rows_ptr
    rowInds = row_indices
    toScipyCsc = to_scipy_csc
    toScipyCsr = to_scipy_csr
    parallel_dot = pdot

    def eliminate_zeros(self):
        """Remove explicitly stored zeros in place, retaining all nonzero values."""
        values = self.values()
        mask = values != 0
        result = SparseArray(self.shape, dtype=self.dtype, storagetype=self.storagetype)
        result._put_coordinates(
            values[mask], tuple(index[mask] for index in self.nonzero())
        )
        self._array = result._array

    def column_indices(self, column):
        """Return stored row indices in a matrix column."""
        return self._array.colInds(column)

    colInds = column_indices

    def count_nonzero(self, axis=None):
        """Count nonzero values, excluding explicitly stored zeros."""
        axis = self._normalize_axis(axis)
        mask = self.values() != 0
        if axis is None or self.ndim == 1:
            return numpy.count_nonzero(mask)
        return numpy.bincount(
            self.nonzero()[1 - axis][mask], minlength=self.shape[1 - axis]
        )

    def __repr__(self):
        return f"SparseArray(shape={self.shape}, dtype={self.dtype}, nnz={self.nnz}, order={self.order!r})"

    __array_priority__ = 1000

    dtype = property(_get_dtype)
    T = property(transpose)
    baseTypes = [
        csarray_int_colMajor,
        csarray_double_colMajor,
        csarray_float_colMajor,
        csarray_int64_t_colMajor,
        csarray_short_colMajor,
        csarray_signed_char_colMajor,
    ]
    baseTypes.extend(
        [
            csarray_int_rowMajor,
            csarray_double_rowMajor,
            csarray_float_rowMajor,
            csarray_int64_t_rowMajor,
            csarray_short_rowMajor,
            csarray_signed_char_rowMajor,
        ]
    )


# Backward-compatible constructor name.
csarray = SparseArray
