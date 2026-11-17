Installation
============

SpPy requires Python 3.11 or newer, NumPy 2, Cython 3.3.1 or newer,
Eigen 5.0.1 or newer, and a compiler supporting C++17. Cython and NumPy
build dependencies are installed automatically by pip in an isolated environment.
Eigen is a header-only C++ dependency and must be installed separately.
Download it from https://libeigen.gitlab.io/ and extract the release archive.

From a source checkout::

    EIGEN_INCLUDE_DIR=/path/to/eigen-5.0.1 python -m pip install .

The include directory must contain ``Eigen/Core``. Standard system Eigen
locations and ``vendor/eigen`` are also searched. On Windows, set the same
environment variable in PowerShell before running pip and use MSVC with C++17.

For development::

    EIGEN_INCLUDE_DIR=/path/to/eigen-5.0.1 python -m pip install -e '.[dev]'
    python -m pytest
    ruff check .
    ruff format --check sppy setup.py
    EIGEN_INCLUDE_DIR=/path/to/eigen-5.0.1 python -m build

SciPy interoperability, Matrix Market IO, and the full test suite require SciPy.
Install ``sppy[scipy]`` when using these features.

OpenMP is optional. Set ``SPPY_OPENMP=1`` to enable it when the compiler and
linker support the corresponding flags. Without it, ``pdot`` runs serially.
On macOS the default build does not require an OpenMP runtime.

The public ``csarray`` API is retained. Removed NumPy aliases such as
``numpy.float`` and ``numpy.int`` must be replaced in user code with ``float``
and ``int`` or explicit NumPy dtypes. Sparse dimensions and stored-entry
indices are limited to signed 32-bit integers.
