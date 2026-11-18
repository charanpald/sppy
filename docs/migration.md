# Migrating to 1.0

Version 1.0 is a breaking release. The generated Cython backends have been
replaced by a typed pybind11/Eigen implementation, including native complex
storage, direct factorizations, iterative solvers, preconditioners, and orderings.
Rebuild the package after pulling these changes; old binary extensions are not
compatible with the new interface.

```sh
uv sync --locked --reinstall-package sppy
```

## Removed compatibility APIs

| Removed | Use in 1.0 |
| --- | --- |
| `sppy.csarray`, `sppy.csarray` module | `sppy.SparseArray`, `sppy.sparse_array` module |
| `storagetype`, `storage_type`, `storageType` | `order="C"` for rows, `order="F"` for columns |
| `toScipyCsr`, `toScipyCsc` | `to_scipy_csr`, `to_scipy_csc` |
| `rowInds`, `colInds` | `row_indices`, `column_indices` |
| `subArray` | `subarray` |
| `nonzeroRowsList`, `nonzeroRowsPtr` | `nonzero_rows_list`, `nonzero_rows_ptr` |
| `pdot`, `hadamard`, `getnnz`, `diag` method | `parallel_dot`, `multiply`, `nnz`, `diagonal` |
| `biCGSTAB(maxIter=..., tol=...)` | `bicgstab(maxiter=..., rtol=...)` |
| `asLinearOperator`, `asLinearOperatorSum` | `as_linear_operator`, `as_linear_operator_sum` |
| `sppy.linalg.GeneralLinearOperator` module | Import the class from `sppy.linalg` |
| `eye(n, dtype)` positional shorthand | `eye(n, dtype=dtype)` |
| `prune(eps, precision)` | `prune(tolerance=...)`, using an absolute magnitude threshold |
| `sppy.lib.Parameter`, `sppy.lib.Util` | Ordinary Python/NumPy validation and operations |

All factories and the direct constructor now default to **C order**.
`mmread` takes keyword-only `order="C"`, replacing `storagetype`.
Tuple constructor input still denotes a shape; `asarray` treats a tuple as data.
Old pickle state dictionaries are not supported; export arrays before upgrading.

## Stored entries versus nonzero values

`nnz` counts stored entries, including explicit zeros. `nonzero()` now excludes
stored zeros. Code that pairs coordinates with `values()` must use
`stored_indices()` instead:

```python
import sppy

a = sppy.eye(3)
rows, columns = a.stored_indices()
values = a.values()
assert len(values) == len(rows) == len(columns) == a.nnz
```

`compressed_arrays()` returns `(values, indices, indptr)` in the selected
CSR/CSC storage order. Each array is an independent copy.

## Solvers and complex values

The iterative functions return `IterativeResult`. Existing tuple unpacking
`x, status = bicgstab(a, b)` still works. The result also exposes `iterations`
and `error`. Status values use `ComputationInfo`: success=0, numerical issue=1,
no convergence=2, invalid input=3. These are Eigen's values, not SciPy's codes.

Use reusable `SparseLU`, `SparseQR`, `SimplicialLLT`, or `SimplicialLDLT`
factors for repeated solves. Direct numerical failures raise
`numpy.linalg.LinAlgError`. Pattern reuse requires the same shape, dtype,
and stored coordinates. An unsuccessful refactorization invalidates solving.

`T` is a transpose; `H` is a conjugate transpose. Linear operators now use the
adjoint for reverse products. Randomized SVD reconstructs complex inputs as
`(u * s) @ v.conj().T`; it uses a local generator instead of global RNG state.
`rand` also uses a local generator; pass `rng=` to reproduce a workload.

NumPy compatibility remains partial: slicing returns copies, arithmetic requires
matching shapes, and NumPy coercion can allocate a dense array. See
[interoperability](interoperability.md) and the explicit
[Eigen coverage boundaries](eigen-coverage.md).
