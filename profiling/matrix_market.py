"""Profile Matrix Market reading from a supplied file or a generated fixture."""

from pathlib import Path
from tempfile import TemporaryDirectory

from profiling._common import make_matrix, parser, run_profile
from sppy.io import mmread, mmwrite


def main(argv=None):
    arguments = parser(__doc__)
    arguments.add_argument(
        "path", nargs="?", type=Path, help="Optional existing Matrix Market file"
    )
    options = arguments.parse_args(argv)
    if options.path is not None and not options.path.is_file():
        arguments.error(f"Matrix Market file does not exist: {options.path}")
    with TemporaryDirectory(prefix="sppy-profile-") as directory:
        path = options.path
        if path is None:
            matrix, _ = make_matrix(
                options.size, options.entries, options.seed, options.order
            )
            path = Path(directory) / "matrix.mtx"
            mmwrite(path, matrix)
        run_profile(
            lambda: mmread(path, order=options.order),
            options.repeats,
            options.output,
        )


if __name__ == "__main__":
    main()
