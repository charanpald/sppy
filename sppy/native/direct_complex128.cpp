#include "direct.h"
void bind_direct_complex128(py::module_ &m) {
    sppy::bind_direct<std::complex<double>>(m, "complex128");
}
