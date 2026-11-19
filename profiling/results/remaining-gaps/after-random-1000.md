# SpPy versus SciPy sparse

AMD Ryzen 5 5600X 6-Core Processor · Python 3.14.3 · NumPy 2.5.5 · SciPy 1.18.1 · SpPy 1.0.0

Pattern: random; shape: [1000, 1000]; nnz: 4993; dtype: float64; threads: 1; samples: 7.

Median microseconds per call. Ratios are SciPy / SpPy in matching storage order; **above 1 means SpPy is faster**.

| Operation | SpPy C µs | SciPy CSR µs | CSR / C | SpPy F µs | SciPy CSC µs | CSC / F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| construct-triplets | 47.91 | 68.73 | 1.43× | 47.83 | 68.45 | 1.43× |
| construct-compressed | 6.69 | 10.11 | 1.51× | 6.88 | 10.27 | 1.49× |
| copy | 2.16 | 11.07 | 5.12× | 2.21 | 11.25 | 5.08× |
| copy-assign | 27.82 | 52.74 | 1.9× | 28.73 | 53.20 | 1.85× |
| gather | 5.38 | 12.66 | 2.35× | 5.41 | 12.86 | 2.38× |
| slice | 10.23 | 53.53 | 5.23× | 9.64 | 53.79 | 5.58× |
| transpose | 10.76 | 37.41 | 3.48× | 11.04 | 37.53 | 3.4× |
| adjoint | 10.48 | 52.83 | 5.04× | 10.55 | 52.73 | 5× |
| add | 10.17 | 32.70 | 3.22× | 12.37 | 32.93 | 2.66× |
| multiply | 10.65 | 36.15 | 3.4× | 12.91 | 36.31 | 2.81× |
| matvec | 3.99 | 4.43 | 1.11× | 4.22 | 5.34 | 1.26× |
| matmat | 9.55 | 18.16 | 1.9× | 11.84 | 17.54 | 1.48× |
| sparse-matmul | 177.39 | 220.75 | 1.24× | 179.37 | 220.88 | 1.23× |
| sum | 1.61 | 3.47 | 2.15× | 1.62 | 3.52 | 2.18× |
| column-sum | 3.43 | 28.17 | 8.2× | 3.24 | 19.85 | 6.12× |
| mean | 4.13 | 28.74 | 6.95× | 4.12 | 28.51 | 6.93× |
| norm | 0.67 | 2.10 | 3.11× | 0.66 | 2.30 | 3.47× |
| compressed-export | 1.58 | 1.67 | 1.06× | 1.60 | 1.65 | 1.03× |
| copy-prune | 5.50 | 16.59 | 3.02× | 5.45 | 16.71 | 3.07× |
| lu-factor | 41721.38 | 40146.00 | 0.962× | 42213.70 | 40485.91 | 0.959× |
| lu-solve | 1413.42 | 1707.33 | 1.21× | 1416.75 | 1709.26 | 1.21× |
| cg | 135.42 | 415.49 | 3.07× | 158.36 | 447.61 | 2.83× |
| bicgstab | 163.54 | 527.50 | 3.23× | 188.33 | 561.26 | 2.98× |
| triangular-solve | 18.67 | 343.83 | 18.4× | 27.55 | 265.27 | 9.63× |

Each result passed numerical checks outside timing. Setup, calibration, warmups, and validation are excluded. Public API allocation/conversion is included; inspect each backend's scope in the JSON report.

LU/iterative/triangular workloads use a derived Hermitian diagonally dominant system; its nnz is recorded per benchmark. These are synthetic, in-process measurements, not universal performance guarantees.
