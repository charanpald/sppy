#include "preconditioner.h"
void bind_preconditioner_float32(py::module_ &m) { sppy::bind_preconditioner<float>(m, "float32"); }
