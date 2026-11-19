# SpPy versus SciPy sparse

AMD Ryzen 5 5600X 6-Core Processor · Python 3.14.3 · NumPy 2.5.5 · SciPy 1.18.1 · SpPy 1.0.0

Pattern: random; shape: [1000, 1000]; nnz: 4993; dtype: complex128; threads: 1; samples: 7.

Median microseconds per call. Ratios are SciPy / SpPy in matching storage order; **above 1 means SpPy is faster**.

| Operation | SpPy C µs | SciPy CSR µs | CSR / C | SpPy F µs | SciPy CSC µs | CSC / F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| construct-triplets | 50.88 | 73.96 | 1.45× | 50.73 | 74.66 | 1.47× |
| construct-compressed | 8.16 | 10.88 | 1.33× | 8.11 | 11.06 | 1.36× |
| copy | 2.88 | 12.08 | 4.19× | 2.97 | 12.31 | 4.15× |
| copy-assign | 28.81 | 53.47 | 1.86× | 29.23 | 54.01 | 1.85× |
| gather | 5.53 | 12.55 | 2.27× | 5.69 | 12.81 | 2.25× |
| slice | 10.23 | 53.78 | 5.26× | 9.76 | 53.90 | 5.52× |
| transpose | 12.71 | 38.83 | 3.06× | 12.68 | 38.79 | 3.06× |
| adjoint | 13.02 | 55.81 | 4.29× | 12.88 | 55.83 | 4.34× |
| add | 12.07 | 35.31 | 2.92× | 11.04 | 35.78 | 3.24× |
| multiply | 12.54 | 39.71 | 3.17× | 12.51 | 41.30 | 3.3× |
| matvec | 8.68 | 6.84 | 0.788× | 7.57 | 7.29 | 0.963× |
| matmat | 33.95 | 39.31 | 1.16× | 40.77 | 35.42 | 0.869× |
| sparse-matmul | 263.90 | 228.40 | 0.865× | 277.22 | 228.12 | 0.823× |
| sum | 2.37 | 3.84 | 1.62× | 2.37 | 3.91 | 1.65× |
| column-sum | 4.11 | 30.53 | 7.42× | 3.35 | 22.33 | 6.66× |
| mean | 4.84 | 30.43 | 6.28× | 4.84 | 30.75 | 6.36× |
| norm | 1.85 | 4.58 | 2.48× | 1.85 | 4.53 | 2.45× |
| compressed-export | 2.26 | 2.28 | 1.01× | 2.28 | 2.30 | 1.01× |
| copy-prune | 7.80 | 19.46 | 2.49× | 7.70 | 19.70 | 2.56× |
| lu-factor | 118923.45 | 79799.25 | 0.671× | 119302.57 | 80835.14 | 0.678× |
| lu-solve | 2927.10 | 3075.88 | 1.05× | 2922.82 | 3066.74 | 1.05× |
| cg | 342.65 | 548.32 | 1.6× | 366.87 | 558.68 | 1.52× |
| bicgstab | 768.68 | 675.80 | 0.879× | 791.01 | 687.42 | 0.869× |
| triangular-solve | 34.91 | 370.15 | 10.6× | 45.36 | 291.33 | 6.42× |

Each result passed numerical checks outside timing. Setup, calibration, warmups, and validation are excluded. Public API allocation/conversion is included; inspect each backend's scope in the JSON report.

LU/iterative/triangular workloads use a derived Hermitian diagonally dominant system; its nnz is recorded per benchmark. These are synthetic, in-process measurements, not universal performance guarantees.
