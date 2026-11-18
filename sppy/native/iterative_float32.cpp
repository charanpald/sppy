#include "iterative.h"
void bind_iterative_float32(py::module_ &m) { sppy::bind_iterative<float>(m, "float32"); }
