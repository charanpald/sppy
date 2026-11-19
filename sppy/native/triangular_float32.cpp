#include "triangular.h"
void bind_triangular_float32(py::module_ &m) { sppy::bind_triangular<float>(m, "float32"); }
