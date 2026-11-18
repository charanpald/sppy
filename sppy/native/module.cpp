#include "common.h"
void bind_int8(py::module_ &);
void bind_int16(py::module_ &);
void bind_int32(py::module_ &);
void bind_int64(py::module_ &);
void bind_float32(py::module_ &);
void bind_float64(py::module_ &);
void bind_complex64(py::module_ &);
void bind_complex128(py::module_ &);
void bind_direct_float32(py::module_ &);
void bind_iterative_float32(py::module_ &);
void bind_direct_float64(py::module_ &);
void bind_iterative_float64(py::module_ &);
void bind_direct_complex64(py::module_ &);
void bind_iterative_complex64(py::module_ &);
void bind_direct_complex128(py::module_ &);
void bind_iterative_complex128(py::module_ &);
void bind_preconditioner_float32(py::module_ &);
void bind_preconditioner_float64(py::module_ &);
void bind_preconditioner_complex64(py::module_ &);
void bind_preconditioner_complex128(py::module_ &);
PYBIND11_MODULE(_native, m) {
    bind_int8(m);
    bind_int16(m);
    bind_int32(m);
    bind_int64(m);
    bind_float32(m);
    bind_float64(m);
    bind_complex64(m);
    bind_complex128(m);
    bind_direct_float32(m);
    bind_iterative_float32(m);
    bind_direct_float64(m);
    bind_iterative_float64(m);
    bind_direct_complex64(m);
    bind_iterative_complex64(m);
    bind_direct_complex128(m);
    bind_iterative_complex128(m);
    bind_preconditioner_float32(m);
    bind_preconditioner_float64(m);
    bind_preconditioner_complex64(m);
    bind_preconditioner_complex128(m);
#ifdef _OPENMP
    m.attr("openmp_enabled") = true;
#else
    m.attr("openmp_enabled") = false;
#endif
    m.def("set_num_threads", [](int n) {
        sppy::require(n > 0, "Thread count must be positive");
        Eigen::setNbThreads(n);
    });
    m.def("get_num_threads", [] { return Eigen::nbThreads(); });
}
