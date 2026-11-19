# SpPy versus SciPy sparse

AMD Ryzen 5 5600X 6-Core Processor · Python 3.14.3 · NumPy 2.5.5 · SciPy 1.18.1 · SpPy 1.0.0

Pattern: random; shape: [1000, 1000]; nnz: 4993; dtype: float64; threads: 1; samples: 7.

Median microseconds per call. Ratios are SciPy / SpPy in matching storage order; **above 1 means SpPy is faster**.

| Operation | SpPy C µs | SciPy CSR µs | CSR / C | SpPy F µs | SciPy CSC µs | CSC / F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| construct-triplets | 49.95 | 67.94 | 1.36× | 50.19 | 67.83 | 1.35× |
| construct-compressed | 7.97 | 9.89 | 1.24× | 8.34 | 9.91 | 1.19× |
| copy | 2.16 | 11.16 | 5.16× | 2.17 | 11.20 | 5.15× |
| copy-assign | 27.77 | 52.74 | 1.9× | 28.14 | 54.35 | 1.93× |
| gather | 18.82 | 12.39 | 0.659× | 18.78 | 12.67 | 0.675× |
| slice | 10.17 | 53.82 | 5.29× | 9.63 | 54.86 | 5.7× |
| transpose | 10.91 | 37.56 | 3.44× | 10.89 | 37.24 | 3.42× |
| adjoint | 10.66 | 53.16 | 4.99× | 10.49 | 52.90 | 5.05× |
| add | 23.61 | 32.28 | 1.37× | 25.94 | 32.33 | 1.25× |
| multiply | 25.91 | 36.85 | 1.42× | 24.07 | 37.33 | 1.55× |
| matvec | 4.60 | 4.60 | 1× | 4.90 | 5.42 | 1.11× |
| matmat | 24.41 | 19.21 | 0.787× | 21.84 | 19.91 | 0.912× |
| sparse-matmul | 268.59 | 231.31 | 0.861× | 270.29 | 232.47 | 0.86× |
| sum | 2.75 | 3.91 | 1.42× | 2.67 | 3.55 | 1.33× |
| column-sum | 15.64 | 28.65 | 1.83× | 17.84 | 20.06 | 1.12× |
| mean | 5.30 | 29.98 | 5.66× | 5.12 | 34.65 | 6.77× |
| norm | 1.06 | 2.12 | 2× | 1.05 | 2.12 | 2.02× |
| compressed-export | 1.56 | 1.66 | 1.07× | 1.57 | 1.60 | 1.02× |
| copy-prune | 5.50 | 17.61 | 3.2× | 5.49 | 17.27 | 3.15× |
| lu-factor | 45137.67 | 41479.15 | 0.919× | 48319.26 | 39871.02 | 0.825× |
| lu-solve | 2225.62 | 1719.31 | 0.773× | 2265.43 | 1791.50 | 0.791× |
| cg | 234.10 | 401.17 | 1.71× | 228.36 | 437.03 | 1.91× |
| bicgstab | 329.65 | 523.35 | 1.59× | 313.66 | 559.33 | 1.78× |
| triangular-solve | 71.20 | 372.12 | 5.23× | 69.57 | 302.75 | 4.35× |

Each result passed numerical checks outside timing. Setup, calibration, warmups, and validation are excluded. Public API allocation/conversion is included; inspect each backend's scope in the JSON report.

LU/iterative/triangular workloads use a derived Hermitian diagonally dominant system; its nnz is recorded per benchmark. These are synthetic, in-process measurements, not universal performance guarantees.
