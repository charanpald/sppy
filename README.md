# SpPy

SpPy provides NumPy-style sparse arrays backed by Eigen's C++ sparse matrix
library. It supports sparse vectors and matrices, arithmetic, slicing, reductions,
and linear algebra operations.

## Quick start

```python
import numpy as np
import sppy

a = sppy.asarray([[0., 2.], [3., 0.]])
b = a.astype(np.float32)
product = a @ a
elementwise = a * a
column_sums = np.sum(a, axis=0)
dense = np.asarray(a)
```

See the [migration guide](docs/migration.rst) for Pythonic names, compatibility
aliases, behavior changes, and NumPy compatibility limits.

## Installation with uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and a C++17
compiler (GCC, Clang, or MSVC). Download and extract
[Eigen 5.0.1 or newer](https://libeigen.gitlab.io/); it is a separate header-only
C++ dependency. `EIGEN_INCLUDE_DIR` must point to the directory containing
`Eigen/Core`.

From the repository root on Linux or macOS:

```sh
export EIGEN_INCLUDE_DIR=/path/to/eigen-5.0.1
uv sync --locked
uv run --locked python -c "import sppy; print(sppy.eye(3).toarray())"
```

`uv sync` creates `.venv`, installs this package in editable mode, and installs
NumPy, SciPy, pytest, and the development tools at the versions in `uv.lock`.
No manual environment activation is required. The `.python-version` file selects
Python 3.14; uv can download it if needed. Python 3.11–3.14 are covered by CI;
to use another supported interpreter, run `uv sync --locked --python 3.12`.

On Windows, run from a PowerShell session with the MSVC C++ build tools available:

```powershell
$env:EIGEN_INCLUDE_DIR = "C:\path\to\eigen-5.0.1"
uv sync --locked
uv run --locked python -c "import sppy; print(sppy.eye(3).toarray())"
```

Installed Eigen CMake packages, standard system include directories, and
`vendor/eigen` are also searched. macOS and Windows have not been validated
locally.

For a minimal environment without development dependencies:

```sh
uv sync --locked --no-dev
uv run --locked --no-dev python -c "import sppy; print(sppy.__version__)"
```

Add `--extra scipy` to both commands when you need SciPy interoperability or
Matrix Market IO. SciPy is included in the default development environment.
Python build dependencies (Cython, NumPy headers, and scikit-build-core) are
installed in an isolated build environment. CMake and Ninja are provisioned by
the backend if suitable tools are not already installed.

### Development and validation

```sh
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check sppy
uv run --locked clang-format --dry-run --Werror sppy/include/*.h sppy/test/*.cpp
uv build
```

The native extensions build with CMake and scikit-build-core, configured by
`pyproject.toml` and `CMakeLists.txt`. Generated Cython sources stay in the build
directory. `uv build` creates a source distribution and wheel in `dist/`.

After changing dependency declarations, run `uv lock` and commit `uv.lock`.
To rebuild native code or pick up externally updated Eigen headers, run:

```sh
uv sync --locked --reinstall-package sppy
```

To enable OpenMP, the compiler and linker must support it and its runtime must
be installed. Set `SPPY_OPENMP=1` before syncing or building:

```sh
SPPY_OPENMP=1 uv sync --locked --reinstall-package sppy
SPPY_OPENMP=1 uv build
```

Keep the variable set for subsequent `uv run` commands to retain that build
configuration. Without OpenMP, parallel kernels run serially. The CMake option
`-Ccmake.define.SPPY_OPENMP=ON` is also available with `uv build`.

For a regular non-editable installation, pip remains supported:

```sh
python -m pip install .
```

Set `EIGEN_INCLUDE_DIR` first as above. See the
[installation guide](docs/install.rst) for more details.

## Changes in version 0.7.0

- Modern isolated builds via `pyproject.toml`, scikit-build-core, and CMake.
- Reproducible development environments with uv and a committed lockfile.
- Eigen 5, Cython 3, NumPy 2, and C++17 support.
- A Pythonic `SparseArray` interface with compatibility aliases.
- NumPy-style indexing, matrix multiplication, dtype conversion, and reductions.
- Optional OpenMP, explicit Eigen discovery, and safer native operations.
- SciPy Matrix Market IO, formatting checks, and pytest CI.


### 0.6.7

* 1d arrays can now be pickled
* Remove string calls for compatibility with Python 3.4
* Remove fromScipySparse method (use constructor instead)

### 0.6.6

* Fix bugs in slicing
* Turn off bounds checking
* Element-wise sin, cos, floor, ceil, sign for 2D arrays

### 0.6.5

* Add methods to support pickling
* nonzeroRowsPtr method
* Several bug fixes

### 0.6.4

* Constructor of csarray accepts scipy.sparse
* Added io module to read and write matrix market files
* In csarray: clip, and submatrix methods
* Various optimisations and bug fixes
* Better documentation

### 0.6.3

* Added prune function to remove nnz elements
* Bug fix for numpy.int32 in csarray.__getitem__

### 0.6.2

* Added sppy.linalg.biCGSTAB which solves linear equations of the form Ax = b with x unknown.

### 0.6.1

* Better documentation
* Added sppy.linalg.rsvd (randomised Singular Value Decomposition) and sppy.linalg.norm

### 0.6

* Dot product with numpy arrays, and parallel version (pdot)
* sppy.linalg.GeneralLinearOperator to work with some scipy.sparse.linalg functions
* Optimisations to put method
* Convert from scipy.sparse matrices (csarray.fromScipySparse)
* Automatic generation of specialised templates during builds (credit: Björn Dahlgren)

