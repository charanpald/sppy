# SpPy versus SciPy sparse

AMD Ryzen 5 5600X 6-Core Processor · Python 3.14.3 · NumPy 2.5.5 · SciPy 1.18.1 · SpPy 1.0.0

Pattern: random; shape: [10000, 10000]; nnz: 49985; dtype: float64; threads: 1; samples: 7.

Median microseconds per call. Ratios are SciPy / SpPy in matching storage order; **above 1 means SpPy is faster**.

| Operation | SpPy C µs | SciPy CSR µs | CSR / C | SpPy F µs | SciPy CSC µs | CSC / F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| construct-triplets | 656.37 | 688.28 | 1.05× | 549.16 | 679.30 | 1.24× |
| construct-compressed | 104.77 | 24.03 | 0.229× | 105.03 | 23.50 | 0.224× |
| copy | 14.20 | 26.65 | 1.88× | 14.75 | 25.66 | 1.74× |
| copy-assign | 47.71 | 81.78 | 1.71× | 50.08 | 73.78 | 1.47× |
| gather | 19.17 | 12.40 | 0.647× | 19.95 | 12.89 | 0.646× |
| slice | 229.19 | 251.65 | 1.1× | 233.68 | 254.68 | 1.09× |
| transpose | 187.47 | 177.68 | 0.948× | 185.53 | 179.37 | 0.967× |
| adjoint | 182.45 | 215.11 | 1.18× | 181.62 | 211.49 | 1.16× |
| add | 262.11 | 248.34 | 0.947× | 278.86 | 246.36 | 0.883× |
| multiply | 281.34 | 249.31 | 0.886× | 280.09 | 250.74 | 0.895× |
| matvec | 65.58 | 62.87 | 0.959× | 65.70 | 56.91 | 0.866× |
| matmat | 259.40 | 206.03 | 0.794× | 237.78 | 195.32 | 0.821× |
| sparse-matmul | 5191.66 | 2489.12 | 0.479× | 5344.20 | 2553.81 | 0.478× |
| sum | 16.34 | 10.38 | 0.635× | 16.12 | 10.50 | 0.651× |
| column-sum | 157.52 | 88.46 | 0.562× | 170.08 | 162.86 | 0.958× |
| mean | 20.52 | 115.98 | 5.65× | 19.69 | 113.44 | 5.76× |
| norm | 8.54 | 6.24 | 0.731× | 8.47 | 6.09 | 0.719× |
| compressed-export | 12.74 | 12.79 | 1× | 12.66 | 12.74 | 1.01× |
| copy-prune | 78.45 | 92.67 | 1.18× | 83.71 | 92.42 | 1.1× |

Each result passed numerical checks outside timing. Setup, calibration, warmups, and validation are excluded. Public API allocation/conversion is included; inspect each backend's scope in the JSON report.

LU/iterative/triangular workloads use a derived Hermitian diagonally dominant system; its nnz is recorded per benchmark. These are synthetic, in-process measurements, not universal performance guarantees.
