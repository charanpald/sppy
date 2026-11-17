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

## Installation

Building from source requires:

- Python 3.11 or newer.
- A C++17 compiler, such as GCC, Clang, or MSVC.
- Eigen 5.0.1 or newer, installed separately as C++ headers.

Pip installs NumPy 2 and the required Cython 3.3.1+ build dependency automatically.
SciPy is optional for sparse format conversion and Matrix Market IO. OpenMP is
optional for parallel kernels.

### Install from this checkout

First, download and extract Eigen from the [Eigen website](https://libeigen.gitlab.io/).
Use the directory containing `Eigen/Core` as `EIGEN_INCLUDE_DIR`.

On Linux or macOS, run these commands from the repository root:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
export EIGEN_INCLUDE_DIR=/path/to/eigen-5.0.1
python -m pip install .
```

To include SciPy interoperability and Matrix Market IO, replace the last command
with:

```sh
python -m pip install '.[scipy]'
```

On Windows, use a PowerShell session with the MSVC C++ build tools available:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
$env:EIGEN_INCLUDE_DIR = "C:\path\to\eigen-5.0.1"
python -m pip install ".[scipy]"
```

Standard system Eigen include directories and `vendor/eigen` are also searched,
so `EIGEN_INCLUDE_DIR` can be omitted when suitable headers are found there.
The current build has been validated locally on Linux; macOS and Windows have
not been validated locally.

Verify the installation:

```sh
python -c "import sppy; print(sppy.__version__); print(sppy.eye(3).toarray())"
```

### Development installation

With Eigen configured as above, install the editable package and development
tools:

```sh
python -m pip install -e '.[dev]'
python -m pytest
ruff check .
ruff format --check sppy setup.py
clang-format --dry-run --Werror sppy/include/*.h sppy/test/*.cpp
python -m build
```

To enable OpenMP, set `SPPY_OPENMP=1` before installing or building:

```sh
SPPY_OPENMP=1 python -m pip install .
```

The compiler and linker must support OpenMP and its runtime must be available.
Without OpenMP, parallel kernels run serially. See the
[installation guide](docs/install.rst) for additional build details.

## Changes in version 0.7.0

- Modern isolated builds via `pyproject.toml` and setuptools.
- Eigen 5, Cython 3, NumPy 2, and C++17 support.
- A Pythonic `SparseArray` interface with compatibility aliases.
- NumPy-style indexing, matrix multiplication, dtype conversion, and reductions.
- Optional OpenMP, explicit Eigen discovery, and safer native operations.
- SciPy Matrix Market IO, formatting checks, and pytest CI.

## Changelog

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
* Automatic generation of specialised templates in setup.py (credit: Björn Dahlgren)

