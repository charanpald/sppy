#include "array.h"
void bind_int8(py::module_ &m) { sppy::bind_scalar<int8_t>(m, "int8"); }
