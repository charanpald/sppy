#include "iterative.h"
void bind_iterative_complex128(py::module_ &m) {
    sppy::bind_iterative<std::complex<double>>(m, "complex128");
}
