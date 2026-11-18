"""Profile one checked SpPy/SciPy workload; inspect saved output with pstats."""

import cProfile
from pathlib import Path
import pstats

from profiling._common import (
    BACKENDS,
    OPERATIONS,
    numerical_threads,
    parser,
    validate_options,
)


def main(argv=None):
    arguments = parser(__doc__)
    arguments.add_argument("--operation", choices=OPERATIONS, default="matmat")
    arguments.add_argument("--backend", choices=BACKENDS, default="sppy_C")
    arguments.add_argument("--output", type=Path, help="Save a cProfile .prof file")
    options = arguments.parse_args(argv)
    validate_options(arguments, options)
    from profiling.workloads import Fixture

    with numerical_threads(options.threads):
        workload = Fixture(options).workloads(options.operation)[options.backend]
        workload.check(workload.run())
        for _ in range(options.warmups):
            workload.run()
        profiler = cProfile.Profile()
        for _ in range(options.repeats):
            last = profiler.runcall(workload.run)
        workload.check(last)
        if options.output is not None:
            options.output.parent.mkdir(parents=True, exist_ok=True)
            profiler.dump_stats(str(options.output))
        pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(30)


if __name__ == "__main__":
    main()
