"""Profiling tools must be import-safe and runnable with small inputs."""

import json
from pathlib import Path
import pstats
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULES = ["sparse_array", "linear_operator", "matrix_market", "benchmark"]


@pytest.mark.parametrize("module", MODULES)
def test_profiling_import_has_no_side_effects(module):
    # Arrange
    command = [sys.executable, "-c", f"import profiling.{module}"]

    # Act
    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=30
    )

    # Assert
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert result.stderr == ""


@pytest.mark.parametrize("module", MODULES)
def test_profiling_help(module):
    # Arrange
    command = [sys.executable, "-m", f"profiling.{module}", "--help"]

    # Act
    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=30
    )

    # Assert
    assert result.returncode == 0, result.stderr
    assert "--repeats" in result.stdout


@pytest.mark.parametrize("module", MODULES)
def test_profiling_small_workload(module, tmp_path):
    # Arrange
    output = tmp_path / ("timings.json" if module == "benchmark" else "profile.prof")
    command = [
        sys.executable,
        "-m",
        f"profiling.{module}",
        "--size",
        "8",
        "--entries",
        "12",
        "--repeats",
        "1",
        "--output",
        str(output),
    ]

    # Act
    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=30
    )

    # Assert
    assert result.returncode == 0, result.stderr
    assert output.is_file()
    if module == "benchmark":
        report = json.loads(output.read_text())
        assert set(report["seconds_per_call"]) == {"sppy", "scipy_csr", "scipy_csc"}
        assert all(seconds >= 0 for seconds in report["seconds_per_call"].values())
    else:
        assert pstats.Stats(str(output)).total_calls > 0
