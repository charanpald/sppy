# Arrays and constructors

For examples and storage behavior, see the [sparse array guide](sparseops.md).
`SparseArray` is the preferred name; `csarray` remains a compatibility alias.

## Constructors

::: sppy.util
    options:
      members: [asarray, zeros, ones, eye, diag, rand]

## SparseArray

The constructor accepts a shape or existing data. A tuple is interpreted as a
shape; use `sppy.asarray` to treat a tuple as values. The constructor defaults
to column storage; `order="C"` or `order="F"` explicitly selects storage.

Properties include `shape`, `ndim`, `size`, `dtype`, `order`, `nnz`, and `T`.
`nnz` counts stored entries; use `count_nonzero` for nonzero values.
`parallel_dot` aliases `pdot`; OpenMP builds can run its kernels in parallel.
Without OpenMP, the kernels run serially.

::: sppy.csarray.SparseArray
    options:
      merge_init_into_class: true
      filters: ["!^_", "!^(toScipyCsr|toScipyCsc|rowInds|colInds|subArray|nonzeroRowsList|nonzeroRowsPtr|hadamard)$"]
