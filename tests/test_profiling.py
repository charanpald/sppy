"""Benchmark correctness, isolation, metadata, and CLI contracts (not speed thresholds)."""

from __future__ import annotations

import argparse
import gc
import json
import pstats
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import sparse

import sppy
from profiling._common import BACKENDS, OPERATIONS, numerical_threads, parser
from profiling.benchmark import time_batch
from profiling.workloads import Fixture, SolveResult

ROOT = Path(__file__).resolve().parents[1]


def options(*arguments: str) -> argparse.Namespace:
    """Create small deterministic benchmark options."""
    return parser("test").parse_args(["--size", "7", "--entries", "20", *arguments])


@pytest.mark.parametrize("module", ["benchmark", "profile", "workloads"])
def test_import_has_no_output_or_global_rng_changes(module: str) -> None:
    """Verify import has no output or global rng changes."""
    script = f"""
import numpy as np
before = np.random.get_state()
import profiling.{module}
after = np.random.get_state()
assert before[0] == after[0]
np.testing.assert_array_equal(before[1], after[1])
assert before[2:] == after[2:]
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == result.stderr == ""


@pytest.mark.parametrize("module", ["benchmark", "profile"])
def test_help(module: str) -> None:
    """Verify help."""
    result = subprocess.run(
        [sys.executable, "-m", f"profiling.{module}", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--repeats" in result.stdout and "--dtype" in result.stdout


@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
@pytest.mark.parametrize("pattern", ["random", "banded"])
@pytest.mark.parametrize("operation", OPERATIONS)
def test_workloads_pass_numeric_validation_and_preserve_inputs(
    dtype: str, pattern: str, operation: str
) -> None:
    """Verify workloads pass numeric validation and preserve inputs."""
    fixture = Fixture(options("--dtype", dtype, "--pattern", pattern))
    before = fixture.fingerprint()
    with numerical_threads(1):
        workloads = fixture.workloads(operation)
        assert tuple(workloads) == BACKENDS
        for workload in workloads.values():
            actual = workload.run()
            assert workload.check(actual)["passed"]
    assert fixture.fingerprint() == before
    # Assignment and pruning are copy workloads; no backend source may change.
    for a in fixture.arrays.values():
        fixture.check_equal(a, fixture.csr)


@pytest.mark.parametrize("operation", OPERATIONS)
def test_zero_entry_inputs(operation: str) -> None:
    """Verify zero entry inputs."""
    fixture = Fixture(options("--entries", "0"))
    with numerical_threads(1):
        for workload in fixture.workloads(operation).values():
            assert workload.check(workload.run())["passed"]


def test_fixture_is_reproducible_and_uses_matched_data() -> None:
    """Verify fixture is reproducible and uses matched data."""
    first, second = Fixture(options()), Fixture(options())
    assert first.fingerprint() == second.fingerprint()
    for backend in BACKENDS:
        first.check_equal(first.arrays[backend], second.arrays[backend])
        assert first.arrays[backend].dtype == np.dtype("float64")
    assert first.csr.indices.dtype == first.csc.indices.dtype == np.dtype("int32")
    assert Fixture(options("--seed", "22")).fingerprint() != first.fingerprint()


def test_validation_rejects_bad_results_and_nonconvergence() -> None:
    """Verify validation rejects bad results and nonconvergence."""
    fixture = Fixture(options())
    with pytest.raises(AssertionError, match="differ"):
        fixture.check_equal(sparse.eye_array(7), fixture.csr)
    with pytest.raises(AssertionError, match="converge"):
        fixture.check_solution(
            SolveResult(np.ones(7), 2), sparse.eye_array(7), np.ones(7)
        )
    with pytest.raises(AssertionError, match="residual"):
        fixture.check_solution(np.zeros(7), sparse.eye_array(7), np.ones(7))


def test_time_batch_restores_gc_after_failure() -> None:
    """Verify time batch restores gc after failure."""

    def fails() -> None:
        """Raise an error to exercise cleanup after a failed benchmark."""
        raise RuntimeError("intentional")

    previous = gc.isenabled()
    with pytest.raises(RuntimeError, match="intentional"):
        time_batch(fails, 1)
    assert gc.isenabled() == previous


def test_thread_context_restores_eigen_settings() -> None:
    """Verify thread context restores eigen settings."""
    previous = sppy.get_num_threads()
    with numerical_threads(1):
        assert sppy.get_num_threads() == 1
    assert sppy.get_num_threads() == previous


def test_benchmark_cli_saves_checked_samples_and_summary(tmp_path: Path) -> None:
    """Verify benchmark cli saves checked samples and summary."""
    output, summary = tmp_path / "report.json", tmp_path / "report.md"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "profiling.benchmark",
            "--size",
            "8",
            "--entries",
            "12",
            "--operations",
            "matmat",
            "cg",
            "--repeats",
            "2",
            "--warmups",
            "0",
            "--min-time",
            "0",
            "--output",
            str(output),
            "--markdown",
            str(summary),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["schema_version"] == 1
    assert report["environment"]["eigen_threads"] == 1
    assert report["environment"]["scipy"]
    assert len(report["input"]["sha256"]) == 64
    for case in report["benchmarks"]:
        assert set(case["backends"]) == set(BACKENDS)
        assert len(case["round_order"]) == 2
        for timing in case["backends"].values():
            assert len(timing["samples_seconds"]) == 2
            assert timing["median_seconds"] > 0
            assert timing["calls_per_sample"] == 1
            assert timing["validation"]["passed"]
    assert summary.read_text() == result.stdout
    assert "above 1 means SpPy is faster" in result.stdout


@pytest.mark.parametrize("backend", BACKENDS)
def test_profile_cli_saves_readable_stats(backend: str, tmp_path: Path) -> None:
    """Verify profile cli saves readable stats."""
    output = tmp_path / "workload.prof"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "profiling.profile",
            "--backend",
            backend,
            "--operation",
            "lu-solve",
            "--size",
            "8",
            "--entries",
            "12",
            "--repeats",
            "1",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert pstats.Stats(str(output)).total_calls > 0


@pytest.mark.parametrize(
    "argument,value",
    [
        ("--threads", "0"),
        ("--repeats", "0"),
        ("--min-time", "nan"),
        ("--rtol", "0"),
        ("--size", str(2**31)),
    ],
)
def test_invalid_options_are_cli_errors(argument: Any, value: Any) -> None:
    """Verify invalid options are cli errors."""
    result = subprocess.run(
        [sys.executable, "-m", "profiling.benchmark", argument, value],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "error:" in result.stderr


def test_rsvd_benchmark_saves_validated_time_and_memory(tmp_path: Path) -> None:
    """The SVD benchmark must record each dtype/storage/power case and memory scope."""
    from profiling.rsvd import main

    output = tmp_path / "rsvd.json"
    report = main(
        [
            "--rows",
            "8",
            "--columns",
            "5",
            "--rank",
            "2",
            "--repeats",
            "1",
            "--output",
            str(output),
        ]
    )
    assert json.loads(output.read_text()) == report
    assert len(report["cases"]) == 8
    for case in report["cases"]:
        assert len(case["samples_seconds"]) == 1
        assert case["median_seconds"] >= 0
        assert case["tracked_peak_bytes"] > 0
        assert np.isfinite(case["residual_norm"])
    assert "untracked native" in report["scope"]
