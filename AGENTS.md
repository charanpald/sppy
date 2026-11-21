# Repository guidance

SpPy provides NumPy-style sparse vectors and matrices backed by Eigen 5 and
pybind11. Python 3.11+ and NumPy 2 are required. SciPy is optional at runtime;
native solvers must work without it.

## Layout

- `sppy/sparse_array.py`: public array API, indexing, and reductions.
- `sppy/util.py`: array factories.
- `sppy/linalg/`: Python solver validation, linear operators, and decompositions.
- `sppy/native/`: C++17 storage and numerical kernels, specialized by dtype.
- `sppy/io/`: optional SciPy-backed Matrix Market interoperability.
- `tests/`: deterministic numerical and native-boundary tests.
- `profiling/`: opt-in, numerically validated benchmarks; results are historical data.
- `docs/`: MkDocs source; `mkdocs.yml` defines navigation.

## Setup and commands

Set `EIGEN_INCLUDE_DIR` to a directory containing Eigen 5.0.1+ `Eigen/Core`,
then run `uv sync --locked`. Use the existing `.venv` when dependencies and
the native extension are already installed. Rebuild after native changes with
`uv sync --locked --reinstall-package sppy`. Keep the selected `SPPY_OPENMP`
setting consistent across sync, test, and build commands.

Run the checks relevant to each change, then the full suite before completion:

```sh
uv run --locked pytest
uv run --locked mypy
uv run --locked python -m mypy.stubtest sppy._native --allowlist tools/native-stubtest-allowlist.txt
uv run --locked ruff check .
uv run --locked ruff format --check sppy tests profiling tools
uv run --locked clang-format --dry-run --Werror sppy/native/*.h sppy/native/*.cpp
```

Use `uv build` for packaging changes. For documentation, run
`uv run --locked --only-group docs mkdocs build --strict`; this needs no native
extension. The docs-only dependency group can replace the development environment,
so use a separate environment or restore it with `uv sync --locked` afterward.

After building, run `uv run --locked python tools/validate_wheel.py` to test the
installed wheel in a fresh environment away from checkout imports. Native sanitizer
builds use `SPPY_SANITIZERS=address,undefined`, `SPPY_OPENMP=0`, and a separate
build directory; see `docs/contributing.md` for runtime preloading and commands.
CI covers Linux, macOS, Windows, and a Linux sanitizer build.

## Implementation contracts

- Add useful docstrings and parameter/return annotations to Python functions,
  including private helpers and tests. Document classes and modules. Use
  `numpy.typing` for arrays; retain `Any` at genuinely dynamic native, callback,
  heterogeneous report, or deliberately invalid test-input boundaries.
- Preserve supported signed integer, real, and complex dtypes and C/F storage.
  Distinguish explicit stored zeros (`nnz`) from mathematical nonzeros.
- Sparse results own their buffers. Avoid dense intermediates unless the public
  operation necessarily changes implicit zeros; transform those zeros correctly.
- Match NumPy's documented indexing, scalar promotion, axis, and reduction
  behavior, while preserving intentional SpPy differences documented in tests
  (for example, empty extrema return NaN).
- Validate dimensions, indices, casts, and finite solver inputs before entering
  Eigen. Avoid signed C++ overflow and preserve modular integer arithmetic.
- Keep optional imports inside the operations that require them. Do not add a
  SciPy import to core array or native solver paths.
- Preserve factorization locking and owned matrix snapshots. Never mutate an
  array concurrently with a native operation reading it.
- Add regression tests for bug fixes, comparing against NumPy, SciPy, or known
  residuals. Cover both storage orders, relevant dtypes, empty inputs, and
  explicit versus implicit zeros. Use local seeded generators and temporary files.
  Generated numerical tests use Hypothesis with deterministic CI examples and no
  timing deadlines. Preserve minimized failures as deterministic regressions.
- Keep `sppy/_native.pyi` synchronized with the bindings. Stubtest's allowlist only
  excludes pybind11's runtime metaclass differences, not member signatures.
- Keep performance assertions out of unit tests. Benchmark changes with matched
  inputs, validate outside timed sections, and record the environment and scope.
- Keep `uv.lock` consistent with dependency changes. Do not hand-edit generated
  `build/`, `dist/`, `site/`, caches, or historical benchmark reports.
