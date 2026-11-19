# SpPy versus SciPy sparse

AMD Ryzen 5 5600X 6-Core Processor · Python 3.14.3 · NumPy 2.5.5 · SciPy 1.18.1 · SpPy 1.0.0

Pattern: random; shape: [1000, 1000]; nnz: 4993; dtype: complex128; threads: 1; samples: 7.

Median microseconds per call. Ratios are SciPy / SpPy in matching storage order; **above 1 means SpPy is faster**.

| Operation | SpPy C µs | SciPy CSR µs | CSR / C | SpPy F µs | SciPy CSC µs | CSC / F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| construct-triplets | 51.26 | 76.80 | 1.5× | 51.92 | 75.62 | 1.46× |
| construct-compressed | 8.28 | 11.05 | 1.33× | 8.54 | 11.05 | 1.29× |
| copy | 2.84 | 12.80 | 4.5× | 2.82 | 12.95 | 4.59× |
| copy-assign | 33.26 | 58.86 | 1.77× | 33.70 | 59.04 | 1.75× |
| gather | 5.68 | 12.78 | 2.25× | 5.59 | 12.73 | 2.28× |
| slice | 10.20 | 54.44 | 5.34× | 9.71 | 54.32 | 5.6× |
| transpose | 12.61 | 39.27 | 3.11× | 12.72 | 39.36 | 3.09× |
| adjoint | 12.79 | 56.24 | 4.4× | 12.71 | 56.32 | 4.43× |
| add | 11.96 | 35.98 | 3.01× | 10.87 | 36.41 | 3.35× |
| multiply | 12.73 | 41.96 | 3.3× | 12.68 | 42.49 | 3.35× |
| matvec | 8.71 | 6.79 | 0.779× | 7.41 | 7.02 | 0.947× |
| matmat | 33.57 | 38.77 | 1.15× | 39.94 | 34.68 | 0.868× |
| sparse-matmul | 263.17 | 228.09 | 0.867× | 267.40 | 229.53 | 0.858× |
| sum | 2.37 | 3.81 | 1.61× | 2.37 | 3.85 | 1.62× |
| column-sum | 4.08 | 30.91 | 7.57× | 3.36 | 22.27 | 6.63× |
| mean | 4.73 | 30.47 | 6.44× | 4.84 | 30.64 | 6.33× |
| norm | 1.85 | 4.53 | 2.46× | 1.85 | 4.62 | 2.5× |
| compressed-export | 2.15 | 2.25 | 1.05× | 2.17 | 2.23 | 1.03× |
| copy-prune | 7.69 | 19.13 | 2.49× | 7.70 | 19.53 | 2.54× |
| lu-factor | 119175.15 | 78760.27 | 0.661× | 119527.35 | 79529.80 | 0.665× |
| lu-solve | 2928.50 | 3106.23 | 1.06× | 2922.79 | 3077.16 | 1.05× |
| cg | 353.28 | 559.88 | 1.58× | 375.70 | 576.39 | 1.53× |
| bicgstab | 778.21 | 678.47 | 0.872× | 801.43 | 696.39 | 0.869× |
| triangular-solve | 35.16 | 375.98 | 10.7× | 45.30 | 292.87 | 6.47× |

Each result passed numerical checks outside timing. Setup, calibration, warmups, and validation are excluded. Public API allocation/conversion is included; inspect each backend's scope in the JSON report.

LU/iterative/triangular workloads use a derived Hermitian diagonally dominant system; its nnz is recorded per benchmark. These are synthetic, in-process measurements, not universal performance guarantees.
