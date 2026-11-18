# Eigen sparse coverage

SpPy 1.0 targets the built-in modules shipped with **Eigen 5.0.1**. The Python
interface materializes owned results instead of exposing C++ expression-template
objects. No SciPy dependency is required for the features in this table.

| Eigen module | SpPy interface |
| --- | --- |
| SparseCore | `SparseArray`, vector/matrix storage in C/F order, indexing and blocks, triplets, CSR/CSC import/export, reserve/compress/uncompress/squeeze, resize/swap, pruning, arithmetic, sparse/dense products, reductions, diagonal scaling, coefficient operations, triangular/Hermitian materialization, adjoint, permutations |
| OrderingMethods | `linalg.ordering(a, "amd"/"colamd"/"natural")`, ordering selection on direct factorizations |
| SparseCholesky | `SimplicialLLT`, `SimplicialLDLT`, symbolic analysis, numerical refactorization, dense/sparse RHS solves, sparse L/U factors, LDLT D, permutations, determinant, diagonal shifts |
| SparseLU | `SparseLU`, symbolic analysis, numerical refactorization, dense/sparse RHS solves, sparse L/U, row/column permutations, determinant/log determinant, pivot threshold |
| SparseQR | `SparseQR`, rectangular least squares, symbolic analysis, numerical refactorization, rank, sparse R, implicit Q/Q.H application, column permutation, pivot threshold |
| IterativeLinearSolvers | `cg`, `bicgstab`, `lscg`, initial guesses, multiple RHS, iteration/error/status diagnostics, identity/diagonal/least-square diagonal/incomplete LU/incomplete Cholesky preconditioners |

See the [array API](csarray.md) and [solver API](linalg.md) for signatures.

## Precision and ownership

Arrays support signed 8/16/32/64-bit integers, float32/64, and complex64/128.
Solvers operate natively in float32/64 or complex64/128; cast integer arrays
explicitly before solving. Sparse vectors use an owned single-column Eigen
matrix internally, with one outer pointer and NumPy-style one-dimensional behavior in Python.
Their native entry storage is proportional to stored entries, even for large
logical vector sizes.

Dimensions and storage indices are signed 32-bit integers. Each matrix has one
storage order; arithmetic converts compatible inputs when needed. Compressed
arrays and factors are copies. Changing them cannot invalidate an Eigen object.

## Deliberate boundaries

This is a Python interface to the built-in modules, not a binding of every C++
template overload. Non-owning `Eigen::Map`/`Ref`, raw pointer mutation,
expression lifetimes, custom C++ scalar types, custom C++ ordering functors,
and custom C++ preconditioner classes are not exposed. Boolean and unsigned
storage, 64-bit storage indices, are not supported. LDLT is intended for positive definite systems, rather than
a general pivoted symmetric-indefinite solver.

External support modules (SuiteSparse, SuperLU, MKL/Pardiso, PaStiX, METIS,
Accelerate) and Eigen's unsupported modules are outside this release's scope.
Newer Eigen APIs beyond the 5.0.1 baseline are not implied by these bindings.

For the upstream module overview, see
[Eigen's sparse tutorial](https://libeigen.gitlab.io/eigen/docs-5.0/group__TutorialSparse.html)
and [built-in sparse solvers](https://libeigen.gitlab.io/eigen/docs-5.0/group__TopicSparseSystems.html).
