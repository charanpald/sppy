# SpPy versus SciPy sparse

AMD Ryzen 5 5600X 6-Core Processor · Python 3.14.3 · NumPy 2.5.5 · SciPy 1.18.1 · SpPy 1.0.0

Pattern: random; shape: [1000, 1000]; nnz: 4993; dtype: float64; threads: 1; samples: 9.

Median microseconds per call. Ratios are SciPy / SpPy in matching storage order; **above 1 means SpPy is faster**.

| Operation | SpPy C µs | SciPy CSR µs | CSR / C | SpPy F µs | SciPy CSC µs | CSC / F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| construct-triplets | 49.86 | 68.44 | 1.37× | 50.77 | 69.43 | 1.37× |
| construct-compressed | 9.23 | 9.69 | 1.05× | 9.30 | 10.00 | 1.08× |
| copy | 2.27 | 10.99 | 4.83× | 2.20 | 11.07 | 5.02× |
| copy-assign | 27.55 | 51.49 | 1.87× | 28.21 | 51.83 | 1.84× |
| gather | 18.71 | 12.24 | 0.654× | 18.97 | 12.44 | 0.656× |
| slice | 10.15 | 52.22 | 5.14× | 9.51 | 52.76 | 5.55× |
| transpose | 11.48 | 36.52 | 3.18× | 11.43 | 36.01 | 3.15× |
| adjoint | 10.85 | 51.23 | 4.72× | 10.53 | 51.35 | 4.88× |
| add | 24.42 | 31.91 | 1.31× | 25.80 | 31.97 | 1.24× |
| multiply | 29.11 | 35.05 | 1.2× | 28.14 | 35.33 | 1.26× |
| matvec | 4.67 | 4.44 | 0.951× | 4.97 | 5.42 | 1.09× |
| matmat | 22.42 | 18.39 | 0.82× | 21.03 | 17.78 | 0.846× |
| sparse-matmul | 248.86 | 218.37 | 0.878× | 514.18 | 220.16 | 0.428× |
| sum | 2.65 | 3.43 | 1.3× | 2.60 | 3.47 | 1.33× |
| column-sum | 15.80 | 27.91 | 1.77× | 17.80 | 20.19 | 1.13× |
| mean | 5.07 | 28.46 | 5.61× | 4.99 | 28.51 | 5.71× |
| norm | 1.05 | 2.29 | 2.18× | 1.05 | 2.10 | 2.01× |
| compressed-export | 1.64 | 1.60 | 0.978× | 1.59 | 1.62 | 1.02× |
| copy-prune | 5.61 | 16.49 | 2.94× | 5.66 | 16.63 | 2.94× |
| lu-factor | 44873.85 | 40847.77 | 0.91× | 45323.73 | 40889.53 | 0.902× |
| lu-solve | 2222.20 | 1714.69 | 0.772× | 2224.32 | 1724.72 | 0.775× |
| cg | 233.90 | 400.69 | 1.71× | 229.33 | 423.82 | 1.85× |
| bicgstab | 313.69 | 503.43 | 1.6× | 308.80 | 535.54 | 1.73× |
| triangular-solve | 69.92 | 339.71 | 4.86× | 67.42 | 265.70 | 3.94× |

Each result passed numerical checks outside timing. Setup, calibration, warmups, and validation are excluded. Public API allocation/conversion is included; inspect each backend's scope in the JSON report.

LU/iterative/triangular workloads use a derived Hermitian diagonally dominant system; its nnz is recorded per benchmark. These are synthetic, in-process measurements, not universal performance guarantees.
