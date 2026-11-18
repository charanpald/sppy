#include "direct.h"
void bind_direct_float64(py::module_ &m) { sppy::bind_direct<double>(m, "float64"); }
