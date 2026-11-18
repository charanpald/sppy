# SpPy versus SciPy sparse

AMD Ryzen 5 5600X 6-Core Processor · Python 3.14.3 · NumPy 2.5.5 · SciPy 1.18.1 · SpPy 1.0.0

Pattern: random; shape: [10000, 10000]; nnz: 49985; dtype: float64; threads: 1; samples: 7.

Median microseconds per call. Ratios are SciPy / SpPy in matching storage order; **above 1 means SpPy is faster**.

| Operation | SpPy C µs | SciPy CSR µs | CSR / C | SpPy F µs | SciPy CSC µs | CSC / F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| construct-triplets | 729.11 | 686.74 | 0.942× | 514.00 | 683.63 | 1.33× |
| construct-compressed | 799.13 | 23.56 | 0.0295× | 642.26 | 23.16 | 0.0361× |
| copy | 1075.68 | 25.27 | 0.0235× | 925.24 | 25.32 | 0.0274× |
| copy-assign | 1079.17 | 71.36 | 0.0661× | 952.91 | 69.71 | 0.0732× |
| gather | 19.74 | 12.36 | 0.626× | 19.43 | 14.11 | 0.726× |
| slice | 31404.69 | 249.40 | 0.00794× | 30437.98 | 250.78 | 0.00824× |
| transpose | 182.30 | 175.01 | 0.96× | 180.42 | 175.30 | 0.972× |
| adjoint | 187.78 | 217.01 | 1.16× | 183.10 | 216.84 | 1.18× |
| add | 260.57 | 244.37 | 0.938× | 279.36 | 242.39 | 0.868× |
| multiply | 264.60 | 246.81 | 0.933× | 263.35 | 245.74 | 0.933× |
| matvec | 78.26 | 63.34 | 0.809× | 74.83 | 57.23 | 0.765× |
| matmat | 652.59 | 207.26 | 0.318× | 669.27 | 199.68 | 0.298× |
| sparse-matmul | 14134.80 | 2495.52 | 0.177× | 6217.02 | 2486.86 | 0.4× |
| sum | 71.30 | 10.77 | 0.151× | 70.01 | 10.51 | 0.15× |
| column-sum | 219.08 | 87.30 | 0.398× | 231.01 | 161.15 | 0.698× |
| mean | 78.00 | 113.02 | 1.45× | 76.97 | 113.08 | 1.47× |
| norm | 59.43 | 7.91 | 0.133× | 58.54 | 7.94 | 0.136× |
| compressed-export | 24.59 | 13.12 | 0.534× | 24.76 | 12.91 | 0.521× |
| copy-prune | 911.85 | 95.60 | 0.105× | 767.55 | 93.94 | 0.122× |

Each result passed numerical checks outside timing. Setup, calibration, warmups, and validation are excluded. Public API allocation/conversion is included; inspect each backend's scope in the JSON report.

LU/iterative/triangular workloads use a derived Hermitian diagonally dominant system; its nnz is recorded per benchmark. These are synthetic, in-process measurements, not universal performance guarantees.
