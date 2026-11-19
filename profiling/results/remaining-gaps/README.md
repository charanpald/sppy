# Remaining-gap optimization results

The baseline is the previous optimized implementation, revision `9325b56`.
Its Python package and compiled extension were preserved before changes;
baseline subprocesses explicitly loaded that copy. The same harness and
configuration were used for both versions, with matching input fingerprints.
All numerical checks passed. Binary hashes and every matched comparison are
in [comparison.json](comparison.json).

This repeats all **24 operations**, **91 workloads** and **182 C/CSR or F/CSC
comparisons** from the [previous audit](../all-operations/README.md): random
float64 at 1,000/10,000 rows, banded float64 at 10,000, and random complex128 at
1,000. The 10,000-row random case retains its 19-array-operation scope.

Recorded on 19 November 2026, AMD Ryzen 5 5600X, Linux x86-64, Python 3.14.3,
NumPy 2.5.5, SciPy 1.18.1, Eigen 5.0.1 and GCC 16.2. Each backend used one
numerical thread, seven samples and a 20 ms target per batch. No compiler or
test process ran alongside final timings. Runtime dispatch was **AVX2/FMA** on
this CPU; other CPUs/compiler targets retain Eigen fallbacks. The active path
is recorded as `environment.runtime_simd` in raw reports.

SpPy trails SciPy in **30 → 17** median comparisons. Gaps exceeding 10% fall from **19 → 10**. The 10% threshold is a display aid, not a statistical significance test.

## Changes

- **Sparse multiplication:** three levels of occupancy bits accumulate values
  in traversal order and enumerate only touched indices in sorted order. This
  removes result sorting and transpose/conversion copies. Stored zeros,
  cancellation, integer overflow semantics and canonical output are preserved.
  Scratch remains bounded; very wide products retain the hash fallback.
- **Compressed imports:** index copying, bounds checks and adjacent-order checks
  share a streaming pass. Pointers are validated before use; duplicate/unsorted
  input still uses summing assembly. Buffer ownership and all validation remain.
- **LU panels:** small untransposed real/complex GEMM panels use guarded AVX2/FMA
  kernels on supported x86 GCC/Clang builds. General shapes, scaling,
  conjugation and threaded GEMM retain Eigen's implementation. SIMD changes
  floating-point rounding; it does not enable global fast-math or native-CPU
  compiler flags. Narrow-band LU uses smaller panel workspace. COLAMD and the
  pivot threshold remain unchanged. General blocking sweeps and extra panel
  packing did not help and were not retained.
- **Complex products:** compressed CSR vector and CSR/CSC eight-RHS products use
  runtime-selected vector kernels. Nonfinite vector results and unsupported
  CPUs/storage use Eigen. General vector evaluation is kept in a focused helper.
- **BiCGSTAB correctness:** Eigen 5's internal loop compares an absolute residual
  while SpPy promises relative `rtol`. The wrapper now scales the threshold per
  RHS and classifies the returned relative error against the requested `rtol`.
  Previously large RHS were oversolved and sufficiently small RHS could fail
  before taking a step. Benchmarked `rtol` is unchanged; independent residual
  checks pass. This speedup includes removing unintended extra iterations.
- **Existing paths:** COO assembly selects its checked count/scatter path from
  outer-run counts instead of an extra sample scan. Triangular bindings are
  compiled separately; a single CSR RHS is solved in its owned buffer. Other
  generic paths and input validation remain available.

No SciPy runtime dependency was added. Kernels are selected automatically;
set `SPPY_DISABLE_SIMD=1` **before starting Python** to force the baseline CPU
path. The complete test suite was also run with that setting.

Validation: **3,162 tests passed in each CPU mode**. Python/native style checks,
the strict MkDocs build, wheel and source builds, source archive completeness,
and isolated wheel numerical smoke tests in both CPU modes also passed. The
wheel smoke tests did not import SciPy.

## Every operation

Speedup is previous SpPy / new SpPy across all input/order combinations.
Counts flag SpPy medians more than 10% above matching SciPy controls.

| Operation | >10% slower before | >10% slower after | SpPy speedup range |
| --- | ---: | ---: | ---: |
| construct-triplets | 1/8 | 1/8 | 0.98–1.04× |
| construct-compressed | 4/8 | 4/8 | 0.97–1.17× |
| copy | 0/8 | 0/8 | 0.99–1.02× |
| copy-assign | 0/8 | 0/8 | 0.94–1.00× |
| gather | 0/8 | 0/8 | 0.97–1.03× |
| slice | 0/8 | 0/8 | 0.98–0.99× |
| transpose | 0/8 | 0/8 | 0.98–1.03× |
| adjoint | 0/8 | 0/8 | 0.97–1.00× |
| add | 0/8 | 0/8 | 0.94–1.00× |
| multiply | 0/8 | 0/8 | 0.93–1.21× |
| matvec | 2/8 | 1/8 | 0.92–1.42× |
| matmat | 1/8 | 0/8 | 0.91–1.74× |
| sparse-matmul | 4/8 | 0/8 | 1.08–1.71× |
| sum | 0/8 | 0/8 | 0.98–1.01× |
| column-sum | 0/8 | 0/8 | 0.93–1.08× |
| mean | 0/8 | 0/8 | 0.97–1.01× |
| norm | 0/8 | 0/8 | 0.96–1.01× |
| compressed-export | 0/8 | 0/8 | 0.94–1.04× |
| copy-prune | 0/8 | 0/8 | 0.88–1.00× |
| lu-factor | 5/6 | 4/6 | 0.97–1.20× |
| lu-solve | 0/6 | 0/6 | 0.93–0.98× |
| cg | 0/6 | 0/6 | 0.89–0.98× |
| bicgstab | 2/6 | 0/6 | 1.14–1.23× |
| triangular-solve | 0/6 | 0/6 | 1.06–1.19× |

## Remaining gaps

These measurements do not establish a universal ranking. All remaining gaps
above 10%, with matching current medians in microseconds, are listed below.
SpPy validates imports and preserves canonical sorted sparse products with
structural zeros; SciPy can do less work at those boundaries. The libraries also
use different numerical LU implementations. Small near-parity results should
be read with the raw samples and interquartile intervals.

| Input | Operation | Order | SpPy µs | SciPy µs | SpPy/SciPy time |
| --- | --- | --- | ---: | ---: | ---: |
| random-10000 | construct-compressed | C | 30.32 | 23.11 | 1.31× |
| random-10000 | construct-compressed | F | 30.49 | 23.24 | 1.31× |
| random-10000 | matvec | F | 66.70 | 57.84 | 1.15× |
| banded-10000 | construct-triplets | C | 194.15 | 167.21 | 1.16× |
| banded-10000 | construct-compressed | C | 30.88 | 23.37 | 1.32× |
| banded-10000 | construct-compressed | F | 30.40 | 23.66 | 1.28× |
| banded-10000 | lu-factor | C | 3168.72 | 2844.16 | 1.11× |
| banded-10000 | lu-factor | F | 3087.45 | 2659.33 | 1.16× |
| complex-1000 | lu-factor | C | 99313.84 | 80124.20 | 1.24× |
| complex-1000 | lu-factor | F | 100154.72 | 80893.30 | 1.24× |

## Regressions and variation

The following medians were more than 10% slower than the previous SpPy run; these are retained explicitly instead of hiding already-fast workloads.

| Input | Operation | Order | Before µs | After µs |
| --- | --- | --- | ---: | ---: |
| banded-10000 | copy-prune | C | 45.66 | 50.99 |
| banded-10000 | copy-prune | F | 45.48 | 51.11 |
| banded-10000 | cg | F | 1082.22 | 1213.27 |
| complex-1000 | copy-prune | F | 7.70 | 8.72 |

## Raw reports and reproduction

| Input | Before | After |
| --- | --- | --- |
| Random 1,000 float64 | [table](before-random-1000.md) · [JSON](before-random-1000.json) | [table](after-random-1000.md) · [JSON](after-random-1000.json) |
| Random 10,000 float64 | [table](before-random-10000.md) · [JSON](before-random-10000.json) | [table](after-random-10000.md) · [JSON](after-random-10000.json) |
| Banded 10,000 float64 | [table](before-banded-10000.md) · [JSON](before-banded-10000.json) | [table](after-banded-10000.md) · [JSON](after-banded-10000.json) |
| Random 1,000 complex128 | [table](before-complex-1000.md) · [JSON](before-complex-1000.json) | [table](after-complex-1000.md) · [JSON](after-complex-1000.json) |

Use the same [full benchmark commands](../all-operations/README.md#full-reports-and-reproduction)
for the four configurations. Build revision `9325b56` separately for the
baseline; keep Eigen/compiler/threads fixed and compare fingerprints before
comparing timings. Current raw reports include CPU dispatch metadata. The
[benchmark methodology](../../README.md) documents allocation, conversion,
solver tolerances and timed scope.

Validation covers all eight dtypes and both orders, occupancy word/group/cap
boundaries, stored zeros/cancellation, ownership, malformed compressed buffers,
nonfinite values, panel tails/strides/conjugation/scaling, pivoted LU systems,
relative-tolerance scaling per RHS, zero RHS and iteration-limit status. SIMD
and forced-fallback suites passed. Packaging and strict documentation checks
were also performed; see the task's final validation note.
