"""Generate Cython extension classes for supported scalar/storage combinations.

Cython does not provide templated extension classes. These private templates
keep one implementation for each operation while C++ retains typed Eigen storage.
"""

from itertools import product
from pathlib import Path
import re

SCALARS = ("signed char", "short", "int", "int64_t", "float", "double")
STORAGE = ("colMajor", "rowMajor")
TEMPLATE = re.compile(r"^cdef template\[([^]]+)\] class (\w+):", re.MULTILINE)


def expand_template(source: Path, destination: Path, combinations) -> None:
    """Expand exactly one extension-class template, preserving its module preamble."""
    text = source.read_text(encoding="utf-8")
    matches = list(TEMPLATE.finditer(text))
    if len(matches) != 1:
        raise ValueError(f"Expected one class template in {source}")
    match = matches[0]
    parameters = [parameter.strip() for parameter in match[1].split(",")]
    class_name = match[2]
    body = text[match.end() :]
    output = [text[: match.start()]]
    notation = re.compile(rf"\b{class_name}\[([^]]+)\]")
    for combination in combinations:
        substitutions = dict(zip(parameters, combination, strict=True))
        if "StorageType" in substitutions:
            substitutions["NotStorageType"] = (
                "rowMajor" if substitutions["StorageType"] == "colMajor" else "colMajor"
            )
        specialized = body
        for parameter, value in substitutions.items():
            specialized = re.sub(rf"\b{parameter}\b", value, specialized)
        specialized = notation.sub(
            lambda found: class_name + "_" + re.sub(r"[,\s]+", "_", found[1]),
            specialized,
        )
        suffix = "_".join(combination).replace(" ", "_")
        output.append(f"cdef class {class_name}_{suffix}:{specialized}")
    generated = "".join(output)
    if not destination.exists() or destination.read_text(encoding="utf-8") != generated:
        destination.write_text(generated, encoding="utf-8")


def expand_base(workdir=".", force=False):
    """Generate sources deterministically; retain force for old build callers."""
    directory = Path(workdir)
    for stem, combinations in (
        ("csarray", tuple(product(SCALARS, STORAGE))),
        ("csarray1d", tuple((scalar,) for scalar in SCALARS)),
    ):
        for suffix in ("pyx", "pxd"):
            expand_template(
                directory / f"{stem}_base.{suffix}",
                directory / f"{stem}_sub.{suffix}",
                combinations,
            )


if __name__ == "__main__":
    expand_base(Path(__file__).parent)
