# SpPy versus SciPy sparse

AMD Ryzen 5 5600X 6-Core Processor · Python 3.14.3 · NumPy 2.5.5 · SciPy 1.18.1 · SpPy 1.0.0

Pattern: random; shape: [10000, 10000]; nnz: 49985; dtype: float64; threads: 1; samples: 7.

Median microseconds per call. Ratios are SciPy / SpPy in matching storage order; **above 1 means SpPy is faster**.

| Operation | SpPy C µs | SciPy CSR µs | CSR / C | SpPy F µs | SciPy CSC µs | CSC / F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| construct-triplets | 708.85 | 703.68 | 0.993× | 534.42 | 687.01 | 1.29× |
| construct-compressed | 800.69 | 23.63 | 0.0295× | 810.10 | 23.05 | 0.0285× |
| copy | 873.31 | 24.51 | 0.0281× | 906.74 | 24.83 | 0.0274× |
| copy-assign | 896.85 | 67.30 | 0.075× | 884.37 | 69.13 | 0.0782× |
| gather | 18.61 | 12.47 | 0.67× | 19.19 | 12.97 | 0.676× |
| slice | 30627.25 | 250.11 | 0.00817× | 30731.46 | 252.92 | 0.00823× |
| transpose | 185.78 | 175.46 | 0.944× | 181.14 | 178.00 | 0.983× |
| adjoint | 183.81 | 215.16 | 1.17× | 179.79 | 206.93 | 1.15× |
| add | 259.80 | 242.14 | 0.932× | 281.12 | 241.04 | 0.857× |
| multiply | 264.91 | 245.70 | 0.927× | 263.32 | 244.13 | 0.927× |
| matvec | 77.57 | 62.57 | 0.807× | 74.40 | 56.64 | 0.761× |
| matmat | 651.80 | 202.80 | 0.311× | 669.39 | 192.87 | 0.288× |
| sparse-matmul | 14562.07 | 2474.13 | 0.17× | 5338.10 | 2495.94 | 0.468× |
| sum | 71.39 | 10.26 | 0.144× | 70.09 | 10.38 | 0.148× |
| column-sum | 220.11 | 86.94 | 0.395× | 231.01 | 163.41 | 0.707× |
| mean | 77.47 | 112.67 | 1.45× | 76.49 | 112.37 | 1.47× |
| norm | 59.19 | 7.77 | 0.131× | 58.76 | 7.83 | 0.133× |
| compressed-export | 24.27 | 12.78 | 0.526× | 24.30 | 12.75 | 0.525× |
| copy-prune | 914.84 | 92.94 | 0.102× | 762.27 | 92.81 | 0.122× |

Each result passed numerical checks outside timing. Setup, calibration, warmups, and validation are excluded. Public API allocation/conversion is included; inspect each backend's scope in the JSON report.

LU/iterative/triangular workloads use a derived Hermitian diagonally dominant system; its nnz is recorded per benchmark. These are synthetic, in-process measurements, not universal performance guarantees.
