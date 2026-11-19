#include "triangular.h"
void bind_triangular_complex128(py::module_ &m) {
    sppy::bind_triangular<std::complex<double>>(m, "complex128");
}
