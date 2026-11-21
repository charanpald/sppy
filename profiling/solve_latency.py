"""Measure first and repeated LU solves separately, excluding factorization."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

from profiling._common import (
    BACKENDS,
    environment,
    numerical_threads,
    parser,
    positive_int,
    validate_options,
)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Parse command-line options and execute the profiling tool."""
    arguments = parser(__doc__)
    arguments.add_argument("--backend", choices=BACKENDS, default="sppy_C")
    arguments.add_argument("--warm-solves", type=positive_int, default=7)
    arguments.add_argument("--output", type=Path)
    options = arguments.parse_args(argv)
    validate_options(arguments, options)
    from profiling.workloads import Fixture, _lu

    with numerical_threads(options.threads):
        fixture = Fixture(options)
        arrays = fixture.system()
        matrix, rhs = arrays[options.backend], fixture.dense
        cold, warm = [], []
        for _ in range(options.repeats):
            factor = _lu(options.backend, matrix)
            start = perf_counter()
            result = factor.solve(rhs)
            cold.append(perf_counter() - start)
            fixture.check_solution(result, arrays["scipy_csr"], rhs)
            for _ in range(options.warm_solves):
                start = perf_counter()
                result = factor.solve(rhs)
                warm.append(perf_counter() - start)
            fixture.check_solution(result, arrays["scipy_csr"], rhs)
        report = {
            "environment": environment(),
            "configuration": {
                key: value for key, value in vars(options).items() if key != "output"
            },
            "input": {
                "shape": list(matrix.shape),
                "nnz": int(matrix.nnz),
                "source_sha256": fixture.fingerprint(),
            },
            "scope": "Fresh factor per first solve; factorization and checks excluded. No solve warmups or batching.",
            "cold_median_seconds": median(cold),
            "warm_median_seconds": median(warm),
            "cold_samples_seconds": cold,
            "warm_samples_seconds": warm,
            "validation": {"passed": True},
        }
    encoded = json.dumps(report, indent=2) + "\n"
    if options.output is None:
        print(encoded, end="")
    else:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(encoded)
    return report


if __name__ == "__main__":
    main()
