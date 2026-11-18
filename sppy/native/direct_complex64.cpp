#include "direct.h"
void bind_direct_complex64(py::module_ &m) {
    sppy::bind_direct<std::complex<float>>(m, "complex64");
}
