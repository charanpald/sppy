# Randomized SVD simplification

Local measurements on 21 November 2026 compare the original implementation
against reduced QR calls with unused outputs released promptly and explicit cyclic
garbage collections removed. Inputs, RNG seeds, and all singular-value
decomposition settings match; every reconstruction residual was unchanged.
The new implementation also back-projects only the requested singular vectors.
These measurements include the updated operator wrappers, which fetch a live
adjoint once per reverse call; they do not isolate individual optimizations.

Each case uses a 600 × 300 matrix with approximately 2% nonzeros, rank 20,
oversampling 10, one numerical thread, one warmup, and seven timed samples.
Sparse inputs use C storage. Dense inputs contain the same coefficients.

| dtype | storage | q | before (ms) | after (ms) | time ratio | tracked peak before / after (KiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| float64 | dense | 0 | 3.143 | 1.097 | 2.86× | 785.2 / 785.2 |
| float64 | dense | 2 | 14.609 | 2.558 | 5.71× | 792.7 / 784.7 |
| float64 | sparse | 0 | 2.960 | 0.709 | 4.18× | 785.0 / 784.7 |
| float64 | sparse | 2 | 12.746 | 1.459 | 8.74× | 792.8 / 784.7 |
| complex128 | dense | 0 | 5.273 | 2.494 | 2.11× | 3236.9 / 3236.7 |
| complex128 | dense | 2 | 22.178 | 6.537 | 3.39× | 3532.7 / 3236.7 |
| complex128 | sparse | 0 | 3.761 | 1.393 | 2.70× | 1496.1 / 1495.8 |
| complex128 | sparse | 2 | 15.135 | 3.099 | 4.88× | 1581.2 / 1566.1 |

Time ratio is before / after; above 1 means the new implementation was faster.
Memory is measured in a separate call using tracemalloc. It includes tracked
Python/NumPy allocations and excludes untracked native allocations; these figures
are not total process peak RSS. Small memory differences may reflect allocator noise.

The JSON files contain raw samples and environment metadata. These synthetic
results are local observations, not performance guarantees.

Reproduce the current implementation:

```sh
uv run --locked python -m profiling.rsvd --output /tmp/rsvd.json
```

To compare implementations, run the same command in separate checkouts before
and after the change, using the same dependencies, numerical thread settings,
hardware, and command-line options.
