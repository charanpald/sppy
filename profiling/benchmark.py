"""Benchmark SpPy 1.0 against SciPy CSR/CSC arrays on matched, checked inputs."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import statistics
from collections.abc import Callable, Sequence
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

from profiling._common import (
    ARRAY_OPERATIONS,
    BACKENDS,
    OPERATIONS,
    SOLVER_OPERATIONS,
    environment,
    nonnegative_float,
    numerical_threads,
    parser,
    positive_int,
    validate_options,
)

if TYPE_CHECKING:
    from profiling.workloads import Workload


def time_batch(operation: Callable[[], Any], count: int) -> tuple[float, Any]:
    """Time calls and intermediate result destruction; keep the last result to validate."""
    enabled = gc.isenabled()
    gc.disable()
    try:
        start = perf_counter()
        for _ in range(count):
            result = operation()
        elapsed = perf_counter() - start
    finally:
        if enabled:
            gc.enable()
    return elapsed, result


def measure(
    workloads: dict[str, Workload],
    *,
    repeats: int,
    warmups: int,
    min_time: float,
    max_batch: int,
    seed: int,
) -> tuple[dict[str, Any], list[list[str]]]:
    """Warm, calibrate, then interleave implementations in seeded random order."""
    rng = random.Random(seed)
    names = list(workloads)
    rng.shuffle(names)
    batches, validations = {}, {}
    for name in names:
        workload = workloads[name]
        validations[name] = workload.check(workload.run())
        for _ in range(warmups):
            workload.run()
        elapsed, last = time_batch(workload.run, 1)
        workload.check(last)
        batches[name] = min(
            max_batch, max(1, math.ceil(min_time / max(elapsed, 1e-12)))
        )
    samples = {name: [] for name in names}
    order = []
    for _ in range(repeats):
        rng.shuffle(names)
        order.append(names.copy())
        for name in names:
            elapsed, last = time_batch(workloads[name].run, batches[name])
            samples[name].append(elapsed / batches[name])
            validations[name] = workloads[name].check(last)
    result = {}
    for name in workloads:
        values = samples[name]
        if len(values) == 1:
            q1 = q3 = values[0]
        else:
            q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
        result[name] = {
            "median_seconds": statistics.median(values),
            "min_seconds": min(values),
            "max_seconds": max(values),
            "p25_seconds": q1,
            "p75_seconds": q3,
            "samples_seconds": values,
            "calls_per_sample": batches[name],
            "validation": validations[name],
            "scope": workloads[name].description,
        }
    return result, order


def markdown_report(report: dict[str, Any]) -> str:
    """Render benchmark timings as a Markdown comparison table."""
    config, env = report["configuration"], report["environment"]
    lines = [
        "# SpPy versus SciPy sparse",
        "",
        f"{env['cpu']} · Python {env['python']} · NumPy {env['numpy']} · SciPy {env['scipy']} · SpPy {env['sppy']}",
        "",
        f"Pattern: {config['pattern']}; shape: {report['input']['shape']}; nnz: {report['input']['nnz']}; dtype: {config['dtype']}; threads: {config['threads']}; samples: {config['repeats']}.",
        "",
        "Median microseconds per call. Ratios are SciPy / SpPy in matching storage order; **above 1 means SpPy is faster**.",
        "",
        "| Operation | SpPy C µs | SciPy CSR µs | CSR / C | SpPy F µs | SciPy CSC µs | CSC / F |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in report["benchmarks"]:
        values = {name: case["backends"][name]["median_seconds"] for name in BACKENDS}
        ratio_c = values["scipy_csr"] / values["sppy_C"]
        ratio_f = values["scipy_csc"] / values["sppy_F"]
        lines.append(
            f"| {case['operation']} | {values['sppy_C'] * 1e6:.2f} | {values['scipy_csr'] * 1e6:.2f} | {ratio_c:.3g}× | {values['sppy_F'] * 1e6:.2f} | {values['scipy_csc'] * 1e6:.2f} | {ratio_f:.3g}× |"
        )
    lines += [
        "",
        "Each result passed numerical checks outside timing. Setup, calibration, warmups, and validation are excluded. Public API allocation/conversion is included; inspect each backend's scope in the JSON report.",
        "",
        "LU/iterative/triangular workloads use a derived Hermitian diagonally dominant system; its nnz is recorded per benchmark. These are synthetic, in-process measurements, not universal performance guarantees.",
        "",
    ]
    return "\n".join(lines)


def run(options: argparse.Namespace, operations: Sequence[str]) -> dict[str, Any]:
    """Run the prepared operation and return its result."""
    from profiling.workloads import Fixture

    with numerical_threads(options.threads):
        fixture = Fixture(options)
        report = {
            "schema_version": 1,
            "environment": environment(),
            "configuration": {
                key: value
                for key, value in vars(options).items()
                if key not in {"output", "markdown", "json"}
            },
            "input": {
                "shape": list(fixture.csr.shape),
                "nnz": int(fixture.csr.nnz),
                "sampled_triplets": len(fixture.values),
                "index_dtype": str(fixture.csr.indices.dtype),
                "sha256": fixture.fingerprint(),
            },
            "benchmarks": [],
        }
        report["configuration"]["operations"] = list(operations)
        report["configuration"]["rtol"] = fixture.rtol
        for index, operation in enumerate(operations):
            workloads = fixture.workloads(operation)
            timings, order = measure(
                workloads,
                repeats=options.repeats,
                warmups=options.warmups,
                min_time=options.min_time,
                max_batch=options.max_batch,
                seed=options.seed + index,
            )
            input_array = (
                fixture.system()["scipy_csr"]
                if operation in SOLVER_OPERATIONS
                else fixture.csr
            )
            report["benchmarks"].append(
                {
                    "operation": operation,
                    "shape": list(input_array.shape),
                    "nnz": int(input_array.nnz),
                    "backends": timings,
                    "round_order": order,
                    "scipy_over_sppy": {
                        "csr_over_C": timings["scipy_csr"]["median_seconds"]
                        / timings["sppy_C"]["median_seconds"],
                        "csc_over_F": timings["scipy_csc"]["median_seconds"]
                        / timings["sppy_F"]["median_seconds"],
                    },
                }
            )
        return report


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Parse command-line options and execute the profiling tool."""
    arguments = parser(__doc__)
    arguments.add_argument(
        "--operations",
        nargs="+",
        choices=["all", "arrays", "solvers", *OPERATIONS],
        default=["all"],
    )
    arguments.add_argument(
        "--min-time",
        type=nonnegative_float,
        default=0.01,
        help="Target seconds per sample, subject to --max-batch",
    )
    arguments.add_argument("--max-batch", type=positive_int, default=4096)
    arguments.add_argument("--output", type=Path, help="Save complete JSON results")
    arguments.add_argument(
        "--markdown", type=Path, help="Save a readable Markdown table"
    )
    arguments.add_argument(
        "--json", action="store_true", help="Print JSON instead of the summary table"
    )
    options = arguments.parse_args(argv)
    validate_options(arguments, options)
    operations = []
    for selected in options.operations:
        expanded = (
            OPERATIONS
            if selected == "all"
            else ARRAY_OPERATIONS
            if selected == "arrays"
            else SOLVER_OPERATIONS
            if selected == "solvers"
            else (selected,)
        )
        operations.extend(
            operation for operation in expanded if operation not in operations
        )
    report = run(options, operations)
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    summary = markdown_report(report)
    for path, text in ((options.output, encoded), (options.markdown, summary)):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    print(encoded if options.json else summary, end="")
    return report


if __name__ == "__main__":
    main()
