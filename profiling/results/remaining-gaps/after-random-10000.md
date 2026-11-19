# SpPy versus SciPy sparse

AMD Ryzen 5 5600X 6-Core Processor · Python 3.14.3 · NumPy 2.5.5 · SciPy 1.18.1 · SpPy 1.0.0

Pattern: random; shape: [10000, 10000]; nnz: 49985; dtype: float64; threads: 1; samples: 7.

Median microseconds per call. Ratios are SciPy / SpPy in matching storage order; **above 1 means SpPy is faster**.

| Operation | SpPy C µs | SciPy CSR µs | CSR / C | SpPy F µs | SciPy CSC µs | CSC / F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| construct-triplets | 581.15 | 681.28 | 1.17× | 573.94 | 684.17 | 1.19× |
| construct-compressed | 30.32 | 23.11 | 0.762× | 30.49 | 23.24 | 0.762× |
| copy | 14.05 | 24.53 | 1.75× | 13.92 | 24.88 | 1.79× |
| copy-assign | 43.12 | 68.40 | 1.59× | 44.37 | 67.53 | 1.52× |
| gather | 5.52 | 12.50 | 2.26× | 5.86 | 13.03 | 2.22× |
| slice | 224.70 | 253.64 | 1.13× | 235.09 | 256.48 | 1.09× |
| transpose | 186.04 | 181.51 | 0.976× | 185.87 | 177.48 | 0.955× |
| adjoint | 183.51 | 216.32 | 1.18× | 183.27 | 213.60 | 1.17× |
| add | 94.21 | 247.30 | 2.62× | 117.23 | 244.84 | 2.09× |
| multiply | 94.47 | 252.00 | 2.67× | 117.53 | 245.70 | 2.09× |
| matvec | 68.98 | 63.13 | 0.915× | 66.70 | 57.84 | 0.867× |
| matmat | 148.04 | 211.59 | 1.43× | 159.24 | 199.69 | 1.25× |
| sparse-matmul | 2250.06 | 2566.85 | 1.14× | 2323.58 | 2536.41 | 1.09× |
| sum | 9.30 | 10.67 | 1.15× | 9.43 | 10.73 | 1.14× |
| column-sum | 59.45 | 86.04 | 1.45× | 57.44 | 161.69 | 2.81× |
| mean | 12.42 | 115.97 | 9.34× | 12.55 | 115.72 | 9.22× |
| norm | 4.88 | 7.71 | 1.58× | 5.07 | 7.97 | 1.57× |
| compressed-export | 12.74 | 13.11 | 1.03× | 12.95 | 12.96 | 1× |
| copy-prune | 82.63 | 90.28 | 1.09× | 84.10 | 89.72 | 1.07× |

Each result passed numerical checks outside timing. Setup, calibration, warmups, and validation are excluded. Public API allocation/conversion is included; inspect each backend's scope in the JSON report.

LU/iterative/triangular workloads use a derived Hermitian diagonally dominant system; its nnz is recorded per benchmark. These are synthetic, in-process measurements, not universal performance guarantees.
