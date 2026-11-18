#include "iterative.h"
void bind_iterative_complex64(py::module_ &m) {
    sppy::bind_iterative<std::complex<float>>(m, "complex64");
}
