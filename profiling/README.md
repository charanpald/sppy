# Profiling and SciPy benchmarks

These development tools compare SpPy 1.0 with SciPy's **CSR and CSC sparse arrays**.
Use the development environment (`uv sync --locked`, with Eigen configured).
NumPy, SciPy, and threadpoolctl are development dependencies. Importing a module
runs no workloads and does not change random state or thread settings.

## Benchmark

```sh
uv run --locked python -m profiling.benchmark --operations all --size 1000 --entries 5000 --output /tmp/sppy-random.json --markdown /tmp/sppy-random.md
uv run --locked python -m profiling.benchmark --operations arrays --size 10000 --entries 50000 --output /tmp/sppy-arrays.json
uv run --locked python -m profiling.benchmark --operations solvers --pattern banded --size 10000 --bandwidth 2 --output /tmp/sppy-solvers.json
uv run --locked python -m profiling.benchmark --operations matvec matmat sparse-matmul --dtype complex128 --size 1000 --entries 5000
```

The default runs all 24 workloads in float64, on a 1000 × 1000 random matrix,
with 5000 sampled triplets, one warmup, seven samples, and one numerical thread.
Duplicate triplets are summed; the report records actual nnz and a SHA-256
fingerprint of the canonical input. Banded inputs use `--bandwidth` instead of
`--entries`. Choose float32, float64, complex64, or complex128 with `--dtype`.

The four implementations are `sppy_C`, `sppy_F`, `scipy_csr`, and `scipy_csc`.
Both SciPy representations use signed 32-bit indices, matching SpPy. Initial
values, dtype, dense operands, and mathematical operations match. Each result
is checked numerically before measurement and after each timed sample. Sparse
results are checked with sparse differences, without densifying the whole matrix.
A failed comparison or nonconverged solver aborts the run.

`--min-time` sets the target duration of each sample (default 0.01 seconds).
Calibration chooses a batch size, capped by `--max-batch` (default 4096).
`--repeats` is the number of independently timed batches; raw per-call samples,
median, min/max, and interquartile bounds are saved. Very fast calls can hit
the batch cap before reaching the target duration. Backend order is interleaved
and randomized reproducibly from `--seed` to reduce fixed-order bias.

JSON is written with `--output`; `--markdown` saves the readable table. Use
`--json` to print JSON to stdout. **Ratios are SciPy / SpPy**: a value above one
means SpPy was faster. Compare CSR with C order and CSC with F order. Keep the
individual timings and distributions when comparing different runs.

## What is timed

| Workloads | Scope |
| --- | --- |
| construct-triplets | Construct from the same unsorted triplets, summing duplicates |
| construct-compressed | Copy matching CSR/CSC value/index/pointer buffers |
| copy, copy-assign, copy-prune | Copy, copy plus existing-entry assignment, or copy plus exact-zero removal |
| gather, slice | Paired-coordinate gather or stepped sparse slice |
| transpose, adjoint | An owned result in the input storage order; SciPy views are materialized |
| add, multiply | Same-shape sparse addition or elementwise multiplication |
| matvec, matmat | Sparse times dense vector or a dense matrix with `--columns` RHS columns |
| sparse-matmul | A @ A.H, with the adjoint prepared outside timing |
| sum, column-sum, mean, norm | Reductions and Frobenius norm |
| compressed-export | Independent copies of compressed value/index/pointer arrays |
| lu-factor | Complete public LU factorization using COLAMD and pivot threshold 1 |
| lu-solve | Solve `--columns` dense RHS with factors prepared outside timing |
| cg, bicgstab | Public solve including diagonal-preconditioner setup, zero guess, matched rtol/maxiter |
| triangular-solve | Public lower-triangular solve, with the triangle prepared outside timing |

Solver workloads use a **derived Hermitian positive definite system**: symmetrize
the random/banded matrix, then set a positive strictly dominant diagonal. This
avoids timing failures on singular random matrices. Solver nnz is recorded
separately from array-input nnz. The default iterative rtol is 1e-5 for single
precision and 1e-8 for double precision; override it with `--rtol` and bound
iterations with `--maxiter`.

LU factorization and solve are separate measurements. SciPy's public `splu`
requires CSC, so conversion from its CSR input is included in `lu-factor`.
SpPy's public coordinate extraction and native conversion are also included.
Inputs and any reused factors are prepared outside timing. Public API allocation,
conversion, Python call overhead, and intermediate result destruction remain
inside timing. The final result is kept alive for checking after the timer stops.
Cyclic garbage collection is disabled only inside each batch and restored afterward.

`--threads` controls Eigen and loaded BLAS/OpenMP libraries through threadpoolctl;
previous settings are restored. Metadata records active thread counts, library
versions, CPU, platform, source revision, dirty status, and OpenMP build support.
Thread counts do not guarantee identical parallel algorithms or CPU utilization.

QR/Cholesky factorizations and Eigen's LSCG have no directly matched algorithm
in this comparison and are not timed here. See the upstream
[SciPy sparse linear algebra API](https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html),
[LU interface](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html),
and [CG interface](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html).

## Profile a workload

The same prepared and checked workloads are available to cProfile:

```sh
uv run --locked python -m profiling.profile --operation matmat --backend sppy_C --size 1000 --entries 5000 --repeats 20 --output /tmp/sppy-matmat.prof
uv run --locked python -m profiling.profile --operation lu-factor --backend scipy_csc --size 1000 --entries 5000 --output /tmp/scipy-lu.prof
uv run --locked python -m pstats /tmp/sppy-matmat.prof
```

`--backend` accepts any of the four implementation names. Setup, warmups, and
checks are excluded from the profile. cProfile identifies Python overhead and
native call boundaries; it does not attribute time to individual C++/Eigen kernels.
Use an external native sampling profiler when that detail is needed.

The separate array/operator/IO scripts and the unpaired native timing experiment
have been removed. Benchmarks and profiles now share one workload definition.
Tests verify correctness, isolation, and report structure without asserting
machine-dependent performance thresholds. Tests/profiling ship in source
archives and are excluded from runtime wheels.

See [recorded measurements](results/README.md) for an example run and its limits.

See the [paired optimization results](results/optimization/README.md) for native-copy, basic-slice, and product improvements against the preserved baseline.

See the [compressed-import and sparse-product results](results/import-product/README.md) for the latest paired measurements and remaining SciPy gaps.
