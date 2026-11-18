#include "preconditioner.h"
void bind_preconditioner_complex64(py::module_ &m) {
    sppy::bind_preconditioner<std::complex<float>>(m, "complex64");
}
