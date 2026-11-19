# Changelog

## 1.0.0

- Generate sorted sparse products with hierarchical occupancy bits, avoiding sorting and conversion copies; fuse compressed-index copying with validation.
- Add runtime-selected SIMD panel/complex-product kernels with portable fallbacks and smaller LU workspace for narrow-band inputs.
- Correct BiCGSTAB's per-RHS relative-tolerance scaling and diagnostics, including tiny/zero RHS and iteration limits.

- Add native promoted reductions and paired gather, equal-pattern arithmetic, and faster real/complex dense products and norms.
- Pass owned storage snapshots directly to direct and iterative solvers; combine symbolic/numeric setup and solve dense LU RHS blocks directly from supernodal factors.
- Adapt sparse-product assembly to output ordering and avoid temporary triplet objects during construction.

- Accelerate owned copies, canonical compressed imports, basic slicing, contiguous reductions, and sparse/dense products; preserve dtype, stored zeros, and independent results.

- Vectorize compressed-buffer validation and build sparse products directly in CSR/CSC order, retaining canonical indices and explicit zeros with bounded workspace for large sparse axes.
- Define sparse integer matrix products with NumPy-compatible fixed-width overflow.

- Replace generated Cython classes with typed pybind11/Eigen bindings.
- Add complex64/128 arrays, adjoint products, and complex Matrix Market IO.
- Add Eigen LLT/LDLT, LU, and QR sparse factorizations with symbolic reuse.
- Add CG, BiCGSTAB, and least-squares CG with diagnostics and built-in preconditioners.
- Add AMD/COLAMD/natural orderings, triangular solves, and factor access.
- Add triplet/compressed assembly, storage controls, scaling, and sparse coefficient operations.
- Remove compatibility aliases and legacy helpers; default consistently to C order.
- Make stored_indices/values distinct from nonzero-value coordinates.

See [migration](migration.md) and [Eigen coverage](eigen-coverage.md).

## 0.7.0

- Modern isolated builds via `pyproject.toml`, scikit-build-core, and CMake.
- Reproducible development environments with uv and a committed lockfile.
- Eigen 5, Cython 3, NumPy 2, and C++17 support.
- A Pythonic `SparseArray` interface with compatibility aliases.
- NumPy-style indexing, matrix multiplication, dtype conversion, and reductions.
- Optional OpenMP, explicit Eigen discovery, and safer native operations.
- SciPy Matrix Market IO, formatting checks, and pytest CI.


## 0.6.7

* 1d arrays can now be pickled
* Remove string calls for compatibility with Python 3.4
* Remove fromScipySparse method (use constructor instead)

## 0.6.6

* Fix bugs in slicing
* Turn off bounds checking
* Element-wise sin, cos, floor, ceil, sign for 2D arrays

## 0.6.5

* Add methods to support pickling
* nonzeroRowsPtr method
* Several bug fixes

## 0.6.4

* Constructor of csarray accepts scipy.sparse
* Added io module to read and write matrix market files
* In csarray: clip, and submatrix methods
* Various optimisations and bug fixes
* Better documentation

## 0.6.3

* Added prune function to remove nnz elements
* Bug fix for numpy.int32 in csarray.__getitem__

## 0.6.2

* Added sppy.linalg.biCGSTAB which solves linear equations of the form Ax = b with x unknown.

## 0.6.1

* Better documentation
* Added sppy.linalg.rsvd (randomised Singular Value Decomposition) and sppy.linalg.norm

## 0.6

* Dot product with numpy arrays, and parallel version (pdot)
* sppy.linalg.GeneralLinearOperator to work with some scipy.sparse.linalg functions
* Optimisations to put method
* Convert from scipy.sparse matrices (csarray.fromScipySparse)
* Automatic generation of specialised templates during builds (credit: Björn Dahlgren)

