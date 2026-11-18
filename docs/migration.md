# Migrating to 0.7

The Python interface is ordinary, formatted Python; typed Cython 3 extensions
implement the Eigen kernels. SpPy supports one- and two-dimensional arrays with
signed integer (8/16/32/64 bit) or float32/float64 values. Complex, unsigned, and
boolean storage are not yet supported.

## NumPy-style usage

```python
import numpy as np
import sppy

a = sppy.asarray([[0., 2.], [3., 0.]], order="C")
b = sppy.eye(2, dtype=np.float64)
product = a @ b           # SparseArray
elementwise = a * b      # SparseArray
vector = a @ np.ones(2)  # ndarray
row = a[0]               # one-dimensional SparseArray
reversed_rows = a[::-1]
means = np.mean(a, axis=0, keepdims=True)
single = a.astype(np.float32)
dense = np.asarray(a)    # explicitly allocates dense storage
scipy_csr = a.to_scipy_csr()
```

`asarray` infers the input dtype. Compatible sparse input can be reused with
`copy=False`; conversion from dense input always needs a copy. `zeros`,
`ones`, and `eye` default to C (row) order, as NumPy does. The legacy
`SparseArray(shape)`/`csarray(shape)` constructor retains column order unless
`order` is supplied. A tuple passed to the constructor specifies a shape;
use `asarray` to interpret a tuple as data.

## Renamed APIs

| Preferred | Compatibility alias |
| --- | --- |
| `SparseArray` | `csarray` |
| `to_scipy_csr` | `toScipyCsr` |
| `to_scipy_csc` | `toScipyCsc` |
| `row_indices` | `rowInds` |
| `column_indices` | `colInds` |
| `subarray` | `subArray` |
| `nonzero_rows_list` | `nonzeroRowsList` |
| `nonzero_rows_ptr` | `nonzeroRowsPtr` |
| `parallel_dot` | `pdot` |
| `multiply` | `hadamard` |
| `diagonal` | `diag` (main diagonal) |
| `linalg.bicgstab` | `linalg.biCGSTAB` |
| `as_linear_operator` | `asLinearOperator` |
| `as_linear_operator_sum` | `asLinearOperatorSum` |

The linear operator implementation lives in
`sppy.linalg.general_linear_operator`; the old module remains importable.
`bicgstab` accepts `maxiter` and `rtol`; its legacy wrapper retains
`maxIter` and `tol`. Solver status codes retain SpPy's convention, documented
in the function, rather than SciPy's codes.

## Intentional behavior changes

* Integer indexing drops an axis. Negative assignments, reversed/stepped
  slices, per-axis boolean masks, and broadcasting of paired indices work like
  NumPy. Sparse slices are independent copies, including basic slices.
* `*` is elementwise and `@` is matrix multiplication. Sparse addition,
  subtraction, and elementwise multiplication require matching shapes; general
  arithmetic broadcasting is not implemented. Nonzero scalar addition raises
  `NotImplementedError` because it would fill implicit zeros.
* Reductions accept `axis`, `out`, and `keepdims`. Sum, mean, variance,
  and standard deviation also support `dtype`; variance and standard deviation
  accept `ddof`. Reduction results are dense scalars or arrays. Empty min/max
  retain the historical NaN result.
* Dense conversion preserves dtype, including exact int64 values. Arithmetic
  promotes dtypes, and sum promotes small integers as NumPy does.
* `clip` clamps values rather than discarding out-of-range entries.
  `ceil` and `floor` follow NumPy's output dtype rather than always int64.
  `cos` includes implicit zeros, so its result can store every element.
* `eliminate_zeros` removes only exact stored zeros. `prune` remains the
  tolerance-based operation. `nnz` counts stored entries; `count_nonzero`
  excludes stored zeros.
* SciPy is optional for conversion and Matrix Market IO. The maintained SciPy
  reader handles coordinate formats, comments, symmetry, and compressed files.

## Limits

There is no complete implementation of NumPy's array-function/ufunc dispatch.
`np.asarray` explicitly densifies; NumPy operations that coerce their arguments
may also densify. Prefer sparse methods and operators for large arrays. New-axis
indexing, whole-array boolean masks, more than two dimensions, and the full set
of NumPy reduction keyword arguments are not supported. Stored-entry indices
and individual dimensions use signed 32-bit limits. OpenMP is optional and its
absence makes parallel kernels serial. CI covers Linux; macOS and Windows have
not been validated locally.
