#include "direct.h"
void bind_direct_float32(py::module_ &m) { sppy::bind_direct<float>(m, "float32"); }
