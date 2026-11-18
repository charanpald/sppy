# Sparse arrays

## Construction, dtypes, and storage

```python
import numpy as np
import sppy

empty = sppy.zeros((5, 7), dtype=np.float32)
vector = sppy.zeros(5, dtype=np.int64)
identity = sppy.eye(5)
diagonal = sppy.diag(np.array([1., 2., 3.]))
random = sppy.rand((5, 7), 0.1, rng=np.random.default_rng(21))
columns = sppy.asarray([[0., 2.], [3., 0.]], order="F")
```

Use `asarray` for data, including tuples. In `SparseArray((5, 7))`, the tuple
specifies a shape instead of values. `asarray` infers the input dtype.
`zeros`, `ones`, and `eye` default to row storage (`order="C"`). The direct
`SparseArray` constructor and `rand` retain the legacy column-storage default.
Choose `order="F"` for column storage and `order="C"` for row storage.

`rand` samples coordinates with replacement, so the resulting density may be
lower than requested. Pass a NumPy `Generator` for reproducible inputs.
Integer dtypes cast the random values, which can produce stored zeros.
`ones` stores every element explicitly.

`asarray(a)` may reuse a compatible sparse array. Request `copy=True` for an
independent object; `copy=False` raises if conversion requires allocation.
Dense inputs always require a copy into Eigen storage. `a.copy()` and
`a.astype(np.float32)` provide explicit copying and dtype conversion.

## Indexing and assignment

Integer indices drop an axis; slices preserve it. Negative indices,
reversed or stepped slices, per-axis boolean masks, and paired array indices
are supported. A slice is a copy, including a basic slice.

```python
a = sppy.asarray([[0., 2., 0.], [3., 0., 4.]])
row = a[0]                       # Shape (3,)
rows = a[:1]                     # Shape (1, 3)
reversed_rows = a[::-1]
masked = a[np.array([True, False]), :]
paired = a[[0, 1], [1, 2]]        # Select (0, 1) and (1, 2)
cartesian = a.subarray([0, 1], [1, 2])  # Select a 2 by 2 submatrix

a[[0, 1], [0, 1]] = [5., 6.]
a[:, 2] = 1.
```

Paired indices select coordinate pairs; `subarray` selects the Cartesian
product of row and column indices. Whole-array boolean masks and `None`
(new-axis) indexing are unsupported.

## Assemble a matrix

Reserve space when the approximate entry count is known, insert entries,
then compress to release spare buffer capacity.

```python
a = sppy.zeros((1000, 1000))
a.reserve(3)
a[[1, 2, 3], [4, 5, 6]] = [0.1, 0.2, 0.4]
a.compress()
```

## Arithmetic and reductions

| Expression | Operation | Result |
| --- | --- | --- |
| `a + b`, `a - b` | Same-shape sparse arithmetic | Sparse array |
| `a * b`, `a.multiply(b)` | Same-shape elementwise product | Sparse array |
| `a * 2`, `-a`, `abs(a)` | Scalar multiplication, negation, absolute value | Sparse array |
| `a @ b`, `a.dot(b)` | Matrix product of sparse matrices | Sparse array |
| `a @ dense_vector` | Matrix-vector product | Dense array |
| `a.T` | Transpose | Sparse array |
| `a.sum(axis=0)` | Column sums | Dense array |
| `a.diagonal()` | Main diagonal | Dense array |

A sparse vector dotted with another sparse vector returns a scalar.
Sparse arithmetic promotes dtypes. General arithmetic broadcasting is
unsupported, and nonzero scalar addition raises `NotImplementedError`.

`sum`, `mean`, `min`, `max`, `var`, and `std` accept `axis`, `out`, and
`keepdims`. Sum, mean, variance, and standard deviation also accept `dtype`;
variance and standard deviation accept `ddof`. Implicit zeros participate in
reductions. Empty min/max retain the historical NaN result.

## Stored zeros and memory

`a.nnz` counts stored entries, including stored zeros. `a.count_nonzero()`
counts nonzero values. `a.eliminate_zeros()` removes exact zeros in place;
`a.prune()` removes entries using a tolerance.

!!! warning "Operations that can fill an array"
    `a.toarray()` and `np.asarray(a)` allocate dense storage. `ones`, `cos`,
    clipping to a positive lower bound, and other operations that map zero to
    a nonzero value can store every position. Estimate the full array size
    before applying these operations to large shapes.

Prefer sparse methods when memory matters. [NumPy interoperability](interoperability.md)
explains the conversion boundaries, and the [API reference](csarray.md)
lists the available methods.
