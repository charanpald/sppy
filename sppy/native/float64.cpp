#include "array.h"
void bind_float64(py::module_ &m) { sppy::bind_scalar<double>(m, "float64"); }
