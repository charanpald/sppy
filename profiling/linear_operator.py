"""Profile linear-operator power iterations with configurable, bounded inputs."""

import numpy as np

from profiling._common import make_matrix, parser, positive_int, run_profile
from sppy.linalg import GeneralLinearOperator


def main(argv=None):
    arguments = parser(__doc__)
    arguments.add_argument("--columns", type=positive_int, default=8)
    arguments.add_argument("--iterations", type=positive_int, default=2)
    arguments.add_argument("--parallel", action="store_true")
    options = arguments.parse_args(argv)
    matrix, rng = make_matrix(
        options.size,
        options.entries,
        options.seed,
        order="C" if options.parallel else options.order,
    )
    operator = GeneralLinearOperator.as_linear_operator(
        matrix, parallel=options.parallel
    )
    omega = rng.normal(size=(options.size, options.columns))

    def power_iteration():
        result = operator.matmat(omega)
        for _ in range(options.iterations):
            result = operator.matmat(operator.rmatmat(result))
            magnitude = np.linalg.norm(result)
            if magnitude:
                result /= magnitude
        return result

    run_profile(power_iteration, options.repeats, options.output)


if __name__ == "__main__":
    main()
