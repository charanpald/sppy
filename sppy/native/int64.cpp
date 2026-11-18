#include "array.h"
void bind_int64(py::module_ &m) { sppy::bind_scalar<int64_t>(m, "int64"); }
