#include "array.h"
void bind_complex64(py::module_ &m) { sppy::bind_scalar<std::complex<float>>(m, "complex64"); }
