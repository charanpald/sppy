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
void bind_triangular_float32(py::module_ &);
void bind_triangular_float64(py::module_ &);
void bind_triangular_complex64(py::module_ &);
void bind_triangular_complex128(py::module_ &);
// Internal numerical probe: exercise the same Eigen dispatch used by solver
// panels, including non-unit scaling and conjugated fallback expressions.
template <class T> void bind_gemm_probe(py::module_ &m, const char *name) {
    m.def(name, [](sppy::Array<T> a, sppy::Array<T> b, sppy::Array<T> initial, T alpha,
                   bool adjoint, int padding) {
        auto left = sppy::dense_copy<T>(a), right = sppy::dense_copy<T>(b),
             original = sppy::dense_copy<T>(initial);
        sppy::require(padding >= 0 && padding < 16, "Invalid padding");
        sppy::require((adjoint ? left.rows() : left.cols()) == right.rows() &&
                          original.rows() == (adjoint ? left.cols() : left.rows()) &&
                          original.cols() == right.cols(),
                      "Product shape mismatch");
        sppy::ColDense<T> left_storage =
            sppy::ColDense<T>::Zero(left.rows() + padding, left.cols());
        sppy::ColDense<T> right_storage =
            sppy::ColDense<T>::Zero(right.rows() + padding, right.cols());
        sppy::ColDense<T> result_storage =
            sppy::ColDense<T>::Constant(original.rows() + padding, original.cols(), T(7));
        left_storage.topRows(left.rows()) = left;
        right_storage.topRows(right.rows()) = right;
        auto lhs = left_storage.topRows(left.rows());
        auto rhs = right_storage.topRows(right.rows());
        auto result = result_storage.topRows(original.rows());
        result = original;
        {
            py::gil_scoped_release release;
            if (adjoint)
                result.noalias() += alpha * (lhs.adjoint() * rhs);
            else
                result.noalias() += alpha * (lhs * rhs);
        }
        sppy::require((result_storage.bottomRows(padding).array() == T(7)).all(),
                      "Panel overwrote padding");
        return sppy::array_copy(result);
    });
}
PYBIND11_MODULE(_native, m) {
    bind_gemm_probe<float>(m, "_gemm_float32");
    bind_gemm_probe<double>(m, "_gemm_float64");
    bind_gemm_probe<std::complex<float>>(m, "_gemm_complex64");
    bind_gemm_probe<std::complex<double>>(m, "_gemm_complex128");
    m.attr("runtime_simd") = sppy::panel_simd_available() ? "avx2-fma" : "baseline";
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
    bind_triangular_float32(m);
    bind_triangular_float64(m);
    bind_triangular_complex64(m);
    bind_triangular_complex128(m);
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
