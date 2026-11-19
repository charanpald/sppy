# SpPy versus SciPy sparse

AMD Ryzen 5 5600X 6-Core Processor · Python 3.14.3 · NumPy 2.5.5 · SciPy 1.18.1 · SpPy 1.0.0

Pattern: banded; shape: [10000, 10000]; nnz: 49994; dtype: float64; threads: 1; samples: 7.

Median microseconds per call. Ratios are SciPy / SpPy in matching storage order; **above 1 means SpPy is faster**.

| Operation | SpPy C µs | SciPy CSR µs | CSR / C | SpPy F µs | SciPy CSC µs | CSC / F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| construct-triplets | 206.58 | 159.16 | 0.77× | 237.57 | 314.25 | 1.32× |
| construct-compressed | 34.56 | 22.95 | 0.664× | 35.02 | 23.65 | 0.675× |
| copy | 13.86 | 25.36 | 1.83× | 14.01 | 25.10 | 1.79× |
| copy-assign | 41.46 | 64.68 | 1.56× | 41.17 | 64.05 | 1.56× |
| gather | 5.30 | 12.00 | 2.27× | 5.27 | 11.99 | 2.28× |
| slice | 74.72 | 127.10 | 1.7× | 70.97 | 127.81 | 1.8× |
| transpose | 90.12 | 138.90 | 1.54× | 90.14 | 137.83 | 1.53× |
| adjoint | 89.85 | 172.35 | 1.92× | 89.16 | 169.67 | 1.9× |
| add | 93.42 | 108.75 | 1.16× | 115.99 | 109.22 | 0.942× |
| multiply | 93.59 | 111.53 | 1.19× | 116.18 | 112.06 | 0.965× |
| matvec | 23.75 | 26.13 | 1.1× | 30.17 | 35.22 | 1.17× |
| matmat | 79.45 | 125.37 | 1.58× | 90.73 | 111.88 | 1.23× |
| sparse-matmul | 516.98 | 535.13 | 1.04× | 508.85 | 539.24 | 1.06× |
| sum | 9.15 | 10.33 | 1.13× | 9.13 | 10.40 | 1.14× |
| column-sum | 29.41 | 62.29 | 2.12× | 19.71 | 123.60 | 6.27× |
| mean | 12.04 | 64.03 | 5.32× | 12.09 | 64.59 | 5.34× |
| norm | 4.70 | 6.22 | 1.32× | 4.80 | 7.78 | 1.62× |
| compressed-export | 12.49 | 12.76 | 1.02× | 12.62 | 12.78 | 1.01× |
| copy-prune | 45.06 | 55.39 | 1.23× | 45.57 | 55.92 | 1.23× |
| lu-factor | 3097.02 | 2785.94 | 0.9× | 3000.97 | 2556.27 | 0.852× |
| lu-solve | 392.46 | 494.56 | 1.26× | 393.61 | 499.94 | 1.27× |
| cg | 999.54 | 1284.98 | 1.29× | 1067.71 | 1484.36 | 1.39× |
| bicgstab | 1708.23 | 1731.27 | 1.01× | 1782.64 | 1989.87 | 1.12× |
| triangular-solve | 148.73 | 763.07 | 5.13× | 198.02 | 584.72 | 2.95× |

Each result passed numerical checks outside timing. Setup, calibration, warmups, and validation are excluded. Public API allocation/conversion is included; inspect each backend's scope in the JSON report.

LU/iterative/triangular workloads use a derived Hermitian diagonally dominant system; its nnz is recorded per benchmark. These are synthetic, in-process measurements, not universal performance guarantees.
