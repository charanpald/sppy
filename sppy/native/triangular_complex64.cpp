#include "triangular.h"
void bind_triangular_complex64(py::module_ &m) {
    sppy::bind_triangular<std::complex<float>>(m, "complex64");
}
