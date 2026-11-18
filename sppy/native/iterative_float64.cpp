#include "iterative.h"
void bind_iterative_float64(py::module_ &m) { sppy::bind_iterative<double>(m, "float64"); }
