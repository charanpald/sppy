"""Profile sparse insertion, slicing, reductions, and products.

Run with ``uv run python -m profiling.sparse_array --help``.
"""

import sppy
from profiling._common import make_matrix, parser, run_profile


def main(argv=None):
    arguments = parser(__doc__)
    arguments.add_argument(
        "--operation",
        choices=["put", "slice", "sum", "nonzero", "dot", "sparse-dot", "block"],
        default="dot",
    )
    options = arguments.parse_args(argv)
    matrix, rng = make_matrix(
        options.size, options.entries, options.seed, options.order
    )
    operand = rng.normal(size=(options.size, 8))
    rows, columns = matrix.nonzero()
    values = matrix.values()
    transposed = matrix.T

    def put():
        result = sppy.zeros(matrix.shape, order=options.order)
        result.put(values, rows, columns, init=True)
        return result

    workloads = {
        "put": put,
        "slice": lambda: matrix[::2, ::2],
        "sum": matrix.sum,
        "nonzero": matrix.nonzero,
        "dot": lambda: matrix @ operand,
        "sparse-dot": lambda: matrix @ transposed,
        "block": lambda: matrix.submatrix(
            0, 0, max(1, options.size // 2), options.size
        ),
    }
    run_profile(workloads[options.operation], options.repeats, options.output)


if __name__ == "__main__":
    main()
