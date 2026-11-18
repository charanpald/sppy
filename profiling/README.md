# Profiling and benchmarks

Run these development tools from the repository root after `uv sync --locked`.
They use NumPy/SciPy from the development environment and Python's standard
`cProfile`/`pstats`; no APGL, sandbox, PySparse, or plotting dependency is needed.
Importing a module does not start a workload or change global random state.

```sh
uv run python -m profiling.sparse_array --operation dot --size 1000 --entries 5000 --repeats 5 --output /tmp/sppy-dot.prof
uv run python -m profiling.linear_operator --size 1000 --columns 8 --iterations 2
uv run python -m profiling.matrix_market --size 1000 --entries 5000
uv run python -m profiling.matrix_market /path/to/existing.mtx --repeats 3
uv run python -m profiling.benchmark --operation dot --size 1000 --output /tmp/sppy-timings.json
```

Use `--help` for available operations and settings. Inputs are deterministic
with `--seed` (default 21). Default workloads are modest; increase dimensions and
entry counts explicitly for larger experiments. Input generation is outside the
profiled/timed operation. The construction benchmark intentionally measures
construction; the assignment benchmark includes a copy so repetitions start from
the same data. Timings are descriptive, with no machine-dependent pass threshold.

Inspect a saved profile with:

```sh
uv run python -m pstats /tmp/sppy-dot.prof
```

The older native Eigen timing experiment is in
`native/sparse_matrix_benchmark.cpp`. It is opt-in and uses a larger workload:

```sh
c++ -O3 -std=c++17 -I"$EIGEN_INCLUDE_DIR" profiling/native/sparse_matrix_benchmark.cpp -o /tmp/sppy-native-benchmark
/tmp/sppy-native-benchmark
```

Tests and profiling tools are included in source distributions and excluded from
runtime wheels. The regression suite lives in `tests/` and uses pytest with
Arrange–Act–Assert sections; performance workloads are not collected as tests.
