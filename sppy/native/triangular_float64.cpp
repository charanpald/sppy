#include "triangular.h"
void bind_triangular_float64(py::module_ &m) { sppy::bind_triangular<double>(m, "float64"); }
