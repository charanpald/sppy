"""Build the Eigen-backed Cython extensions."""

import importlib.util
import os
import re
from pathlib import Path
import sys

from Cython.Build import cythonize
import numpy
from setuptools import Extension, setup

ROOT = Path(__file__).parent.resolve()


def eigen_include():
    """Find Eigen without relying on a platform-specific include spelling."""
    candidates = [
        Path(os.environ.get("EIGEN_INCLUDE_DIR", ROOT / "vendor" / "eigen")),
        Path(sys.prefix) / "include" / "eigen3",
        Path("/usr/local/include/eigen3"),
        Path("/usr/include/eigen3"),
        Path("/opt/homebrew/include/eigen3"),
    ]
    for candidate in candidates:
        if (candidate / "Eigen" / "Core").is_file():
            version_header = candidate / "Eigen/Version"
            if not version_header.is_file():
                raise RuntimeError(
                    f"Eigen >=5.0.1 required; older headers found at {candidate}"
                )
            macros = version_header.read_text(encoding="utf-8")

            version = tuple(
                int(re.search(rf"#define EIGEN_{part}_VERSION\s+(\d+)", macros)[1])
                for part in ("MAJOR", "MINOR", "PATCH")
            )
            if version < (5, 0, 1):
                raise RuntimeError(
                    f"Eigen >=5.0.1 required; found {version} at {candidate}"
                )
            return str(candidate)
    raise RuntimeError(
        "Eigen >=5.0.1 headers not found. Set EIGEN_INCLUDE_DIR to the directory containing Eigen/Core."
    )


spec = importlib.util.spec_from_file_location(
    "template_expander", ROOT / "sppy/TemplateExpander.py"
)
expander = importlib.util.module_from_spec(spec)
spec.loader.exec_module(expander)
expander.expand_base(str(ROOT / "sppy"), force=True)

compile_args = ["/std:c++17"] if sys.platform == "win32" else ["-std=c++17"]
link_args = []
if os.environ.get("SPPY_OPENMP") == "1":
    compile_args += ["/openmp"] if sys.platform == "win32" else ["-fopenmp"]
    if sys.platform != "win32":
        link_args.append("-fopenmp")

include_dirs = [numpy.get_include(), eigen_include()]
extensions = [
    Extension(
        f"sppy.{name}",
        [f"sppy/{name}.pyx"],
        language="c++",
        include_dirs=include_dirs,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    )
    for name in ("csarray_sub", "csarray1d_sub")
]
setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": 3, "embedsignature": True},
        build_dir="build/cython",
    )
)
