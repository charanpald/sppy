# SpPy versus SciPy sparse

AMD Ryzen 5 5600X 6-Core Processor · Python 3.14.3 · NumPy 2.5.5 · SciPy 1.18.1 · SpPy 1.0.0

Pattern: random; shape: [1000, 1000]; nnz: 4993; dtype: float64; threads: 1; samples: 9.

Median microseconds per call. Ratios are SciPy / SpPy in matching storage order; **above 1 means SpPy is faster**.

| Operation | SpPy C µs | SciPy CSR µs | CSR / C | SpPy F µs | SciPy CSC µs | CSC / F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| construct-triplets | 60.56 | 67.99 | 1.12× | 49.70 | 68.24 | 1.37× |
| construct-compressed | 83.08 | 10.00 | 0.12× | 72.03 | 10.11 | 0.14× |
| copy | 69.76 | 11.15 | 0.16× | 58.46 | 11.11 | 0.19× |
| copy-assign | 100.89 | 51.98 | 0.515× | 90.12 | 52.43 | 0.582× |
| gather | 18.09 | 12.43 | 0.687× | 18.41 | 12.55 | 0.682× |
| slice | 2964.60 | 53.01 | 0.0179× | 2952.59 | 53.29 | 0.018× |
| transpose | 11.34 | 37.02 | 3.26× | 11.26 | 37.17 | 3.3× |
| adjoint | 10.31 | 52.12 | 5.05× | 10.39 | 52.35 | 5.04× |
| add | 23.38 | 32.10 | 1.37× | 25.24 | 32.46 | 1.29× |
| multiply | 25.83 | 35.14 | 1.36× | 26.05 | 35.36 | 1.36× |
| matvec | 5.82 | 4.42 | 0.759× | 6.01 | 5.25 | 0.873× |
| matmat | 31.64 | 18.13 | 0.573× | 32.90 | 17.61 | 0.535× |
| sparse-matmul | 916.85 | 220.59 | 0.241× | 514.10 | 218.93 | 0.426× |
| sum | 4.83 | 3.39 | 0.701× | 4.85 | 3.43 | 0.706× |
| column-sum | 17.79 | 27.91 | 1.57× | 20.02 | 19.68 | 0.983× |
| mean | 9.81 | 29.14 | 2.97× | 9.61 | 28.85 | 3× |
| norm | 3.51 | 2.09 | 0.595× | 3.48 | 2.09 | 0.599× |
| compressed-export | 2.65 | 1.63 | 0.616× | 2.66 | 1.58 | 0.595× |
| copy-prune | 74.28 | 16.70 | 0.225× | 62.82 | 16.79 | 0.267× |
| lu-factor | 45179.87 | 40220.86 | 0.89× | 44496.05 | 40025.45 | 0.9× |
| lu-solve | 2205.20 | 1698.98 | 0.77× | 2204.42 | 1703.33 | 0.773× |
| cg | 243.56 | 410.88 | 1.69× | 237.06 | 437.92 | 1.85× |
| bicgstab | 324.50 | 524.12 | 1.62× | 320.30 | 550.20 | 1.72× |
| triangular-solve | 76.55 | 348.66 | 4.55× | 72.52 | 269.91 | 3.72× |

Each result passed numerical checks outside timing. Setup, calibration, warmups, and validation are excluded. Public API allocation/conversion is included; inspect each backend's scope in the JSON report.

LU/iterative/triangular workloads use a derived Hermitian diagonally dominant system; its nnz is recorded per benchmark. These are synthetic, in-process measurements, not universal performance guarantees.
