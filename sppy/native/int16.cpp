#include "array.h"
void bind_int16(py::module_ &m) { sppy::bind_scalar<int16_t>(m, "int16"); }
