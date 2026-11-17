Installation
============

SpPy requires Python 3.11 or newer, Eigen 5.0.1 or newer, and a C++17 compiler.
The native build uses scikit-build-core and CMake, with Cython 3.3.1+ and NumPy 2
headers installed automatically in an isolated build environment. CMake and Ninja
are provisioned by the backend when suitable tools are not installed.

Install uv using https://docs.astral.sh/uv/getting-started/installation/.
Download and extract Eigen from https://libeigen.gitlab.io/ separately.

Creating the development environment
-----------------------------------

From a source checkout on Linux or macOS::

    export EIGEN_INCLUDE_DIR=/path/to/eigen-5.0.1
    uv sync --locked
    uv run --locked pytest

``uv sync`` creates ``.venv`` and installs the project in editable mode, along
with the default ``dev`` dependency group. No activation is needed for ``uv run``.
The committed ``uv.lock`` records the resolved dependency versions;
``.python-version`` selects Python 3.14. Override the interpreter with, for
example, ``uv sync --locked --python 3.12``.

The Eigen include directory must contain ``Eigen/Core``. Installed Eigen CMake
packages, standard system include directories, and ``vendor/eigen`` are searched
when ``EIGEN_INCLUDE_DIR`` is unset. On Windows, set the variable in PowerShell::

    $env:EIGEN_INCLUDE_DIR = "C:\path\to\eigen-5.0.1"
    uv sync --locked
    uv run --locked pytest

Use an MSVC developer shell with C++17 support. Linux is validated locally;
macOS and Windows have not been validated locally.

Minimal environments and SciPy
------------------------------

To omit development dependencies::

    uv sync --locked --no-dev
    uv run --locked --no-dev python -c "import sppy; print(sppy.__version__)"

For SciPy interoperability and Matrix Market IO, add ``--extra scipy`` to sync
and run commands. The default development group already includes SciPy.

Checks and release builds
-------------------------

With the default development environment::

    uv run --locked pytest
    uv run --locked ruff check .
    uv run --locked ruff format --check sppy
    uv run --locked clang-format --dry-run --Werror sppy/include/*.h sppy/test/*.cpp
    uv build

``uv build`` writes the source distribution and wheel to ``dist/``. Native
binding generation runs in the build directory. ``setup.py`` and ``MANIFEST.in``
are no longer needed; package and archive settings live in ``pyproject.toml``.

After editing dependency declarations, run ``uv lock`` and commit the lockfile.
After changing external Eigen headers, force a rebuild::

    uv sync --locked --reinstall-package sppy

For optional OpenMP::

    export SPPY_OPENMP=1
    uv sync --locked --reinstall-package sppy
    uv run --locked pytest
    uv build

The compiler and linker must support OpenMP and its runtime must be available.
Keep ``SPPY_OPENMP`` set for subsequent commands. Alternatively, release builds
accept ``uv build -Ccmake.define.SPPY_OPENMP=ON``. Without OpenMP, parallel kernels
run serially.

Pip interoperability
--------------------

uv is a development convenience, not a runtime dependency. Regular PEP 517
installation remains supported::

    EIGEN_INCLUDE_DIR=/path/to/eigen-5.0.1 python -m pip install .

Use ``'.[scipy]'`` instead of ``.`` to include SciPy support.
