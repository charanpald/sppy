"""Compare SpPy and SciPy using the same inputs; emit timings as JSON.

This is an opt-in performance measurement, not a pass/fail regression test.
"""

import json
from time import perf_counter

import sppy
from profiling._common import make_matrix, parser


def main(argv=None):
    arguments = parser(__doc__)
    arguments.add_argument(
        "--operation",
        choices=[
            "construct",
            "set",
            "get",
            "slice",
            "add",
            "multiply",
            "dot",
            "sum",
            "mean",
        ],
        default="dot",
    )
    options = arguments.parse_args(argv)
    matrix, rng = make_matrix(
        options.size, options.entries, options.seed, options.order
    )
    csr = matrix.to_scipy_csr()
    csc = matrix.to_scipy_csc()
    operand = rng.normal(size=(options.size, 8))
    rows, columns = matrix.nonzero()
    values = matrix.values()
    sample_rows = rows[:100]
    sample_columns = columns[:100]

    def set_values(array):
        result = array.copy()
        result[sample_rows, sample_columns] = 1.0
        return result

    def workload(name, array):
        operations = {
            "construct": lambda: (
                sppy.asarray(csr, order=options.order)
                if name == "sppy"
                else type(array)(csr, copy=True)
            ),
            "set": lambda: set_values(array),
            "get": lambda: array[sample_rows, sample_columns],
            "slice": lambda: array[::2, ::2],
            "add": lambda: array + array,
            "multiply": lambda: array.multiply(array),
            "dot": lambda: array @ operand,
            "sum": array.sum,
            "mean": array.mean,
        }
        return operations[options.operation]

    timings = {}
    for name, array in (("sppy", matrix), ("scipy_csr", csr), ("scipy_csc", csc)):
        operation = workload(name, array)
        start = perf_counter()
        for _ in range(options.repeats):
            operation()
        timings[name] = (perf_counter() - start) / options.repeats
    report = {
        "operation": options.operation,
        "size": options.size,
        "nnz": len(values),
        "repeats": options.repeats,
        "seconds_per_call": timings,
    }
    encoded = json.dumps(report, indent=2)
    if options.output is not None:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
