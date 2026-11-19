# Full SpPy/SciPy optimization audit

All 24 benchmark operations were rechecked on the four previously recorded input
families: random float64 at 1,000 and 10,000 rows, banded float64 at 10,000 rows,
and random complex128 at 1,000 rows. The larger random case retains the original
19-array-workload scope; its LU/iterative solvers were not added to this comparison.
This is **91 workloads and 182 matching C/CSR or F/CSC comparisons**.

Measured on 19 November 2026, AMD Ryzen 5 5600X, Linux x86-64, Python 3.14.3,
NumPy 2.5.5, SciPy 1.18.1, Eigen 5.0.1, GCC 16.2, OpenMP enabled. Every backend
used one numerical thread, seven samples and a 20 ms target per sample. No
compilation or tests ran alongside the final public-API timings. All numerical
checks passed. These are local synthetic measurements, not portable rankings.

## Method and result coverage

The original Python package and compiled extension were copied before editing
from revision `7ade79b762eb163d35a07d59cd84fac21b04ebf3`. Baseline subprocesses
loaded that copy after disabling the editable-install redirect. The same harness
and unchanged configuration/input fingerprints were used for both sides.
The [comparison JSON](comparison.json) includes binary hashes, every comparison,
input/configuration metadata and interquartile intervals. Raw reports retain
all samples, interleaved SciPy controls, validation and timing scope. The checkout
revision in their metadata does not identify the separately preserved baseline
extension; the explicit binary hashes do.

Across median timings, SpPy trails SciPy in **77 → 28** of the 182 matching-order comparisons. Using a 10% display threshold, the count is **60 → 22**. This threshold helps flag gaps; it is not a statistical significance test.

## Implementation changes

- **Triplet construction:** Eigen reads a lightweight iterator over the original
  buffers; no temporary vector of triplet objects is built. Globally sorted
  coordinates use Eigen's sorted assembly. Large run/diagonal-ordered COO inputs
  can be counted/scattered directly and reversed where needed. A sample only
  selects a strategy: every outer vector is validated, and duplicate/disordered
  inputs fall back to summing assembly.
- **Compressed import:** vectorizable bounds and adjacent-order scans replace
  short per-segment ordering loops. Distinct boundaries are subtracted exactly
  once, including repeated pointers for empty segments. All validation and owned
  buffer copies remain in the timed public API.
- **Gather and dispatch:** signed paired NumPy indices are broadcast in Python
  and normalized/checked in native code. Same-dtype operands avoid redundant
  result-type and casting dispatch; mixed dtypes/orders retain conversion paths.
- **Coefficient arithmetic:** compressed equal-pattern addition, subtraction and
  elementwise products use contiguous transforms. Self-operations and distinct
  equal-pattern arrays benefit. Different patterns retain the general merge.
  The benchmark's `a+a`/`a.multiply(a)` scope makes this distinction important.
- **Reductions/norms/pruning:** default promoted sums and axis sums operate on
  native storage without value/coordinate copies or NumPy indexed accumulation.
  Other requested dtypes retain NumPy's path. Real Frobenius norms use independent
  accumulators without globally enabling fast-math. Exact-zero pruning avoids
  complex magnitude calculations and retains nonfinite values.
- **Dense products:** fixed eight-column blocks use contiguous RHS operations;
  CSR complex blocks separate real/imaginary values for SIMD. Other RHS widths
  retain Eigen's general paths. Finite complex blocks use the specialization;
  nonfinite data retains the existing IEEE path. A locality probe selects the
  real CSR vector loop; CSC/complex/general vector kernels remain Eigen-based
  after experiments found regressions in custom variants.
- **Sparse products:** denser unordered floating-point outputs can use a symbolic
  counting pass and one final storage-order conversion. Small, ordered/banded,
  integral and extremely wide products retain the single-pass/hash strategies.
  Both strategies produce sorted unique indices and retain structurally touched
  zeros and cancellation entries. Scratch allocation is bounded for huge axes.
- **Solvers:** owned native storage snapshots replace coordinate extraction and
  duplicate-summing reconstruction for direct, iterative and triangular solves.
  Combined direct compute avoids assembling/checking the same pattern twice.
  Iterative solves use CSR snapshots. LU dense RHS blocks with at least four
  columns traverse Eigen's supernodal factors directly and vectorize across RHS
  columns, including separate real/imaginary complex arithmetic. No CSR copies
  of sparse factors or delayed cache setup are needed.

Copies, slices, transpose/adjoint and compressed export were audited as well.
Their existing native paths were retained where experiments/profiling did not
justify a change; small near-parity differences remain visible below.

## Every operation

Counts below flag matching-order comparisons where SpPy's median time exceeds
SciPy by more than 10%. Speedup is old SpPy / new SpPy across **all** inputs and
orders for that operation, including already-fast cases.

| Operation | >10% slower before | >10% slower after | SpPy speedup range |
| --- | ---: | ---: | ---: |
| construct-triplets | 2/8 | 1/8 | 0.94–1.91× |
| construct-compressed | 4/8 | 4/8 | 1.08–2.36× |
| copy | 0/8 | 0/8 | 1.00–1.05× |
| copy-assign | 0/8 | 0/8 | 0.94–1.06× |
| gather | 8/8 | 0/8 | 3.31–3.56× |
| slice | 0/8 | 0/8 | 0.99–1.02× |
| transpose | 0/8 | 0/8 | 0.96–1.05× |
| adjoint | 0/8 | 0/8 | 0.99–1.10× |
| add | 3/8 | 0/8 | 1.76–2.91× |
| multiply | 2/8 | 0/8 | 1.62–2.84× |
| matvec | 4/8 | 2/8 | 1.01–1.24× |
| matmat | 7/8 | 1/8 | 1.65–2.58× |
| sparse-matmul | 6/8 | 6/8 | 1.05–1.30× |
| sum | 5/8 | 0/8 | 1.68–1.88× |
| column-sum | 2/8 | 0/8 | 2.64–6.76× |
| mean | 0/8 | 0/8 | 1.28–1.64× |
| norm | 1/8 | 0/8 | 1.05–1.81× |
| compressed-export | 0/8 | 0/8 | 1.00–1.14× |
| copy-prune | 2/8 | 0/8 | 0.94–4.19× |
| lu-factor | 5/6 | 6/6 | 1.01–1.40× |
| lu-solve | 6/6 | 0/6 | 1.60–1.85× |
| cg | 0/6 | 0/6 | 1.10–1.74× |
| bicgstab | 3/6 | 2/6 | 1.12–1.68× |
| triangular-solve | 0/6 | 0/6 | 2.05–3.47× |

## Remaining gaps and rejected experiments

This does **not** make every operation faster than SciPy. The exact remaining
gaps, including small regressions, are listed in the comparison JSON and full
per-input tables. Larger validated imports, some sparse products and portions of
Eigen's LU/complex iterative kernels still lag. The numerical LU factorization
itself still uses Eigen's general algorithm and the same COLAMD/pivot settings.
A structural-symmetry ordering experiment did not materially improve the exact
solver fixtures, so defaults were retained.

Some work differs at the public API boundary: SpPy keeps canonical sorted sparse
product output and structurally stored zeros; SciPy can return unsorted indices
and discard cancellation zeros. SpPy fully validates imports. Comparisons include
those contracts rather than timing reduced validation or different output semantics.

Rejected prototypes included per-row product sorting, custom complex vector
loops, and lazily converting LU factors to CSR. The latter accelerated warm
solves but substantially penalized the first solve; it was replaced with direct
supernodal traversal. Explicit first/warm solve checks below prevent that cost
from being hidden by benchmark warmups.

## First versus reused LU solves

Factorization is excluded. Each first-call sample creates a fresh factor first;
seven independent factors were tested per input, followed by seven warm solves
each. Eight dense RHS columns, C order, one numerical thread. All residuals were
checked. No sparse-factor conversion cache is present in the final code.

| Input | Before first ms | After first ms | Before warm ms | After warm ms |
| --- | ---: | ---: | ---: | ---: |
| Random 1,000 float64 | 2.233 | 1.445 | 2.173 | 1.380 |
| Banded 10,000 float64 | 0.753 | 0.450 | 0.730 | 0.414 |
| Random 1,000 complex128 | 5.051 | 3.052 | 4.946 | 2.887 |

To repeat a first/warm measurement (also supports `--backend scipy_csr` or CSC/F):

```sh
.venv/bin/python -m profiling.solve_latency --size 1000 --entries 5000 --columns 8 --repeats 7 --warm-solves 7 --output /tmp/solve-latency.json
```

Use the banded/complex options from the full benchmark commands below for the
other cases. This tool times individual calls without solve warmups or batching;
factorization and validation remain outside timing.

Raw first/warm samples: [before](before-cold-solves.json), [after](after-cold-solves.json).
These independent measurements need not exactly match the interleaved benchmark.

## Full reports and reproduction

| Input | Baseline | Optimized |
| --- | --- | --- |
| Random 1,000 float64 | [table](before-random-1000.md) · [JSON](before-random-1000.json) | [table](after-random-1000.md) · [JSON](after-random-1000.json) |
| Random 10,000 float64 | [table](before-random-10000.md) · [JSON](before-random-10000.json) | [table](after-random-10000.md) · [JSON](after-random-10000.json) |
| Banded 10,000 float64 | [table](before-banded-10000.md) · [JSON](before-banded-10000.json) | [table](after-banded-10000.md) · [JSON](after-banded-10000.json) |
| Random 1,000 complex128 | [table](before-complex-1000.md) · [JSON](before-complex-1000.json) | [table](after-complex-1000.md) · [JSON](after-complex-1000.json) |

Use the same development environment after building the extension:

```sh
.venv/bin/python -m profiling.benchmark --operations all --size 1000 --entries 5000 --threads 1 --repeats 7 --min-time 0.02 --output /tmp/random-1000.json --markdown /tmp/random-1000.md
.venv/bin/python -m profiling.benchmark --operations arrays --size 10000 --entries 50000 --threads 1 --repeats 7 --min-time 0.02 --output /tmp/random-10000.json --markdown /tmp/random-10000.md
.venv/bin/python -m profiling.benchmark --operations all --pattern banded --size 10000 --bandwidth 2 --threads 1 --repeats 7 --min-time 0.02 --output /tmp/banded-10000.json --markdown /tmp/banded-10000.md
.venv/bin/python -m profiling.benchmark --operations all --size 1000 --entries 5000 --dtype complex128 --threads 1 --repeats 7 --min-time 0.02 --output /tmp/complex-1000.json --markdown /tmp/complex-1000.md
```

To reproduce the baseline, build the revision above separately with the same
compiler/Eigen/thread settings and run the same harness there. See the
[methodology](../../README.md) for setup, tolerances and timed scope.

The [baseline cProfile output](profile-before.txt) and
[optimized cProfile output](profile-after.txt) cover assembly, indexing,
reductions, arithmetic, products, pruning and solvers. Their absolute times are
not benchmark measurements. They identify Python/native boundaries and show the
elimination of casting/coordinate round-trips; native prototypes informed kernel
selection, followed by the complete public-API measurements above.

Validation includes every dtype/order, mixed operands, broadcasting and extreme
indices, malformed buffers and empty segments, sorted/unsorted/duplicate assembly,
stored zeros, promoted/overflowing integer sums, real/complex rectangular products,
huge sparse axes, pivoted real/complex LU blocks, refactorization and failure
invalidation, nonfinite values, threaded products and installed-wheel smoke checks.
No runtime SciPy dependency was added.

Final checks: **2,667 tests passed**; Ruff lint/format, clang-format and strict MkDocs builds passed. Wheel/source distributions were built, their contents verified, and the installed wheel passed storage/product/reduction/LU smoke checks without importing SciPy.
