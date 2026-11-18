#include "preconditioner.h"
void bind_preconditioner_complex128(py::module_ &m) {
    sppy::bind_preconditioner<std::complex<double>>(m, "complex128");
}
