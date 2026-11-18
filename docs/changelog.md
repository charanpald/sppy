# Changelog

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

