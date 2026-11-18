#include "array.h"
void bind_int32(py::module_ &m) { sppy::bind_scalar<int32_t>(m, "int32"); }
