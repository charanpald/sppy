#include "array.h"
void bind_float32(py::module_ &m) { sppy::bind_scalar<float>(m, "float32"); }
