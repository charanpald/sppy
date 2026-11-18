#include "preconditioner.h"
void bind_preconditioner_float64(py::module_ &m) {
    sppy::bind_preconditioner<double>(m, "float64");
}
