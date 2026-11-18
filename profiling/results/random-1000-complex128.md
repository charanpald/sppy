# SpPy versus SciPy sparse

AMD Ryzen 5 5600X 6-Core Processor · Python 3.14.3 · NumPy 2.5.5 · SciPy 1.18.1 · SpPy 1.0.0

Pattern: random; shape: [1000, 1000]; nnz: 4993; dtype: complex128; threads: 1; samples: 7.

Median microseconds per call. Ratios are SciPy / SpPy in matching storage order; **above 1 means SpPy is faster**.

| Operation | SpPy C µs | SciPy CSR µs | CSR / C | SpPy F µs | SciPy CSC µs | CSC / F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| construct-triplets | 67.19 | 74.94 | 1.12× | 55.03 | 76.10 | 1.38× |
| construct-compressed | 91.43 | 10.91 | 0.119× | 79.98 | 11.09 | 0.139× |
| copy | 77.42 | 12.28 | 0.159× | 64.78 | 12.32 | 0.19× |
| copy-assign | 112.73 | 54.51 | 0.484× | 99.39 | 55.04 | 0.554× |
| gather | 18.55 | 12.41 | 0.669× | 18.88 | 12.66 | 0.67× |
| slice | 3100.37 | 55.77 | 0.018× | 3091.28 | 56.29 | 0.0182× |
| transpose | 12.92 | 38.80 | 3× | 12.94 | 38.89 | 3.01× |
| adjoint | 12.73 | 56.91 | 4.47× | 12.83 | 56.89 | 4.43× |
| add | 25.07 | 36.43 | 1.45× | 26.17 | 36.95 | 1.41× |
| multiply | 27.43 | 41.83 | 1.52× | 27.83 | 42.15 | 1.51× |
| matvec | 11.26 | 6.76 | 0.6× | 9.39 | 7.15 | 0.761× |
| matmat | 76.14 | 38.83 | 0.51× | 61.99 | 35.36 | 0.57× |
| sparse-matmul | 1085.78 | 229.35 | 0.211× | 345.32 | 234.64 | 0.679× |
| sum | 5.41 | 3.81 | 0.704× | 5.30 | 3.83 | 0.722× |
| column-sum | 19.87 | 32.04 | 1.61× | 21.04 | 22.66 | 1.08× |
| mean | 10.45 | 32.36 | 3.1× | 10.30 | 31.10 | 3.02× |
| norm | 3.53 | 4.50 | 1.28× | 3.47 | 4.54 | 1.31× |
| compressed-export | 4.10 | 2.35 | 0.573× | 4.11 | 2.35 | 0.571× |
| copy-prune | 133.22 | 19.40 | 0.146× | 101.62 | 20.32 | 0.2× |
| lu-factor | 119202.46 | 79656.53 | 0.668× | 119609.06 | 79841.53 | 0.668× |
| lu-solve | 4994.37 | 3056.04 | 0.612× | 4960.82 | 3072.60 | 0.619× |
| cg | 419.88 | 543.57 | 1.29× | 417.43 | 555.51 | 1.33× |
| bicgstab | 901.75 | 671.73 | 0.745× | 891.56 | 686.96 | 0.771× |
| triangular-solve | 102.87 | 372.82 | 3.62× | 101.88 | 292.37 | 2.87× |

Each result passed numerical checks outside timing. Setup, calibration, warmups, and validation are excluded. Public API allocation/conversion is included; inspect each backend's scope in the JSON report.

LU/iterative/triangular workloads use a derived Hermitian diagonally dominant system; its nnz is recorded per benchmark. These are synthetic, in-process measurements, not universal performance guarantees.
