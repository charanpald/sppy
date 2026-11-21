"""Install a wheel in a fresh environment and run copied tests away from checkout imports."""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    """Validate package contents, optional SciPy isolation, and the installed test suite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", nargs="?", default="dist/*.whl")
    parser.add_argument(
        "--asan-runtime",
        type=Path,
        help="Preload this ASan runtime for test processes only",
    )
    parser.add_argument(
        "--offline", action="store_true", help="Install using cached dependencies only"
    )
    parser.add_argument(
        "--pytest-args",
        nargs=argparse.REMAINDER,
        default=["-q", "tests"],
        help="Arguments passed to pytest (must be last)",
    )
    options = parser.parse_args(argv)
    wheels = glob.glob(options.wheel)
    if len(wheels) != 1:
        parser.error(
            f"Expected exactly one wheel matching {options.wheel!r}, found {len(wheels)}"
        )
    wheel = Path(wheels[0]).resolve()
    root = Path(__file__).resolve().parents[1]
    uv = shutil.which("uv")
    if uv is None:
        parser.error("uv must be on PATH")
    with tempfile.TemporaryDirectory(prefix="sppy-wheel-") as directory:
        temporary = Path(directory)
        environment = temporary / "env"
        subprocess.run(
            [uv, "venv", "--python", sys.executable, str(environment)], check=True
        )
        python = environment / (
            "Scripts/python.exe" if os.name == "nt" else "bin/python"
        )
        requirements = temporary / "requirements.txt"
        subprocess.run(
            [
                uv,
                "export",
                "--locked",
                "--extra",
                "test",
                "--no-dev",
                "--no-emit-project",
                "--no-hashes",
                "--output-file",
                str(requirements),
            ],
            cwd=root,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        command = [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            "--requirement",
            str(requirements),
            str(wheel),
        ]
        if options.offline:
            command.append("--offline")
        subprocess.run(command, check=True)
        suite = temporary / "suite"
        suite.mkdir()
        for name in ("tests", "profiling"):
            shutil.copytree(
                root / name,
                suite / name,
                ignore=shutil.ignore_patterns("__pycache__", "results"),
            )
        shutil.copy2(root / "pyproject.toml", suite / "pyproject.toml")
        test_environment = os.environ.copy()
        test_environment.pop("PYTHONPATH", None)
        test_environment.pop("PYTHONHOME", None)
        if options.asan_runtime is not None:
            runtime = options.asan_runtime.resolve()
            if not runtime.is_file():
                parser.error(f"ASan runtime does not exist: {runtime}")
            test_environment["LD_PRELOAD"] = str(runtime) + (
                ":" + test_environment["LD_PRELOAD"]
                if test_environment.get("LD_PRELOAD")
                else ""
            )
            # CPython/third-party shutdown allocations are outside SpPy's ownership.
            test_environment["ASAN_OPTIONS"] = "detect_leaks=0:halt_on_error=1"
            test_environment["UBSAN_OPTIONS"] = "halt_on_error=1:print_stacktrace=1"
        probe = """
import pathlib
import sys
import sppy
package = pathlib.Path(sppy.__file__).resolve()
assert package.is_relative_to(pathlib.Path(sys.prefix).resolve()), package
assert package.with_name('py.typed').is_file()
assert package.with_name('_native.pyi').is_file()
assert not any(name == 'scipy' or name.startswith('scipy.') for name in sys.modules)
print('Testing installed wheel:', package, flush=True)
import pytest
raise SystemExit(pytest.main(sys.argv[1:]))
"""
        subprocess.run(
            [
                str(python),
                "-c",
                probe,
                *(["-s"] if options.asan_runtime is not None else []),
                *options.pytest_args,
            ],
            cwd=suite,
            env=test_environment,
            check=True,
        )


if __name__ == "__main__":
    main()
