# SpPy versus SciPy sparse

AMD Ryzen 5 5600X 6-Core Processor · Python 3.14.3 · NumPy 2.5.5 · SciPy 1.18.1 · SpPy 1.0.0

Pattern: random; shape: [10000, 10000]; nnz: 49985; dtype: float64; threads: 1; samples: 7.

Median microseconds per call. Ratios are SciPy / SpPy in matching storage order; **above 1 means SpPy is faster**.

| Operation | SpPy C µs | SciPy CSR µs | CSR / C | SpPy F µs | SciPy CSC µs | CSC / F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| construct-triplets | 601.70 | 703.60 | 1.17× | 543.93 | 697.17 | 1.28× |
| construct-compressed | 80.19 | 23.68 | 0.295× | 74.20 | 23.18 | 0.312× |
| copy | 14.47 | 24.47 | 1.69× | 13.94 | 25.83 | 1.85× |
| copy-assign | 41.71 | 69.93 | 1.68× | 45.60 | 68.12 | 1.49× |
| gather | 19.38 | 13.02 | 0.672× | 19.90 | 12.82 | 0.644× |
| slice | 225.64 | 258.01 | 1.14× | 235.33 | 260.75 | 1.11× |
| transpose | 186.59 | 196.80 | 1.05× | 191.27 | 181.58 | 0.949× |
| adjoint | 184.16 | 213.58 | 1.16× | 182.90 | 213.49 | 1.17× |
| add | 269.12 | 247.15 | 0.918× | 283.66 | 247.47 | 0.872× |
| multiply | 263.13 | 244.33 | 0.929× | 240.85 | 244.89 | 1.02× |
| matvec | 65.72 | 62.54 | 0.952× | 65.85 | 55.92 | 0.849× |
| matmat | 265.09 | 215.18 | 0.812× | 245.83 | 199.69 | 0.812× |
| sparse-matmul | 5038.66 | 2501.14 | 0.496× | 5119.16 | 2455.72 | 0.48× |
| sum | 16.90 | 11.69 | 0.692× | 16.93 | 11.62 | 0.686× |
| column-sum | 156.03 | 85.65 | 0.549× | 169.67 | 164.96 | 0.972× |
| mean | 19.47 | 111.68 | 5.74× | 19.31 | 109.60 | 5.68× |
| norm | 8.44 | 8.34 | 0.988× | 8.57 | 8.40 | 0.981× |
| compressed-export | 13.11 | 13.42 | 1.02× | 13.27 | 13.38 | 1.01× |
| copy-prune | 78.44 | 93.91 | 1.2× | 79.74 | 89.69 | 1.12× |

Each result passed numerical checks outside timing. Setup, calibration, warmups, and validation are excluded. Public API allocation/conversion is included; inspect each backend's scope in the JSON report.

LU/iterative/triangular workloads use a derived Hermitian diagonally dominant system; its nnz is recorded per benchmark. These are synthetic, in-process measurements, not universal performance guarantees.
