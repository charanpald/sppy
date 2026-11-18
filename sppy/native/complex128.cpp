#include "array.h"
void bind_complex128(py::module_ &m) { sppy::bind_scalar<std::complex<double>>(m, "complex128"); }
