# SpPy versus SciPy sparse

AMD Ryzen 5 5600X 6-Core Processor · Python 3.14.3 · NumPy 2.5.5 · SciPy 1.18.1 · SpPy 1.0.0

Pattern: banded; shape: [10000, 10000]; nnz: 49994; dtype: float64; threads: 1; samples: 7.

Median microseconds per call. Ratios are SciPy / SpPy in matching storage order; **above 1 means SpPy is faster**.

| Operation | SpPy C µs | SciPy CSR µs | CSR / C | SpPy F µs | SciPy CSC µs | CSC / F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| construct-triplets | 201.99 | 157.62 | 0.78× | 235.29 | 306.48 | 1.3× |
| construct-compressed | 33.70 | 22.95 | 0.681× | 34.25 | 23.34 | 0.682× |
| copy | 14.13 | 25.07 | 1.77× | 14.30 | 25.39 | 1.78× |
| copy-assign | 40.67 | 66.06 | 1.62× | 40.97 | 65.29 | 1.59× |
| gather | 5.29 | 12.03 | 2.27× | 5.30 | 12.22 | 2.31× |
| slice | 75.20 | 127.04 | 1.69× | 70.65 | 127.74 | 1.81× |
| transpose | 90.84 | 139.57 | 1.54× | 91.16 | 138.96 | 1.52× |
| adjoint | 89.57 | 171.03 | 1.91× | 89.69 | 170.40 | 1.9× |
| add | 92.77 | 108.95 | 1.17× | 116.42 | 109.53 | 0.941× |
| multiply | 93.92 | 114.83 | 1.22× | 115.59 | 115.22 | 0.997× |
| matvec | 23.83 | 26.24 | 1.1× | 30.30 | 35.34 | 1.17× |
| matmat | 80.19 | 126.23 | 1.57× | 93.40 | 113.99 | 1.22× |
| sparse-matmul | 519.35 | 543.16 | 1.05× | 517.60 | 542.94 | 1.05× |
| sum | 9.26 | 10.51 | 1.13× | 9.33 | 10.50 | 1.13× |
| column-sum | 29.39 | 63.24 | 2.15× | 19.94 | 127.89 | 6.41× |
| mean | 12.18 | 69.23 | 5.69× | 12.15 | 70.20 | 5.78× |
| norm | 4.94 | 6.42 | 1.3× | 4.88 | 7.86 | 1.61× |
| compressed-export | 12.54 | 12.78 | 1.02× | 12.74 | 12.96 | 1.02× |
| copy-prune | 45.66 | 56.43 | 1.24× | 45.48 | 56.19 | 1.24× |
| lu-factor | 3126.08 | 2795.17 | 0.894× | 3006.61 | 2602.53 | 0.866× |
| lu-solve | 398.91 | 505.67 | 1.27× | 395.39 | 506.50 | 1.28× |
| cg | 1014.79 | 1302.51 | 1.28× | 1082.22 | 1505.61 | 1.39× |
| bicgstab | 1844.42 | 1892.27 | 1.03× | 1900.21 | 2145.74 | 1.13× |
| triangular-solve | 153.83 | 778.99 | 5.06× | 204.89 | 618.26 | 3.02× |

Each result passed numerical checks outside timing. Setup, calibration, warmups, and validation are excluded. Public API allocation/conversion is included; inspect each backend's scope in the JSON report.

LU/iterative/triangular workloads use a derived Hermitian diagonally dominant system; its nnz is recorded per benchmark. These are synthetic, in-process measurements, not universal performance guarantees.
