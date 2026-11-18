#pragma once
#include "common.h"
#include <type_traits>
namespace sppy {
template <class T, int Order> struct Matrix {
    using Storage = Eigen::SparseMatrix<T, Order, int>;
    Storage a;
    Matrix(int rows, int cols) {
        require(rows >= 0 && cols >= 0, "Negative dimensions");
        a.resize(rows, cols);
    }
    explicit Matrix(Storage value) : a(std::move(value)) {}
    auto coordinates() const {
        py::array_t<int> rows(a.nonZeros()), cols(a.nonZeros());
        int k = 0;
        for (int j = 0; j < a.outerSize(); ++j)
            for (typename Storage::InnerIterator it(a, j); it; ++it, ++k) {
                rows.mutable_data()[k] = it.row();
                cols.mutable_data()[k] = it.col();
            }
        return py::make_tuple(rows, cols);
    }
    auto values() const {
        py::array_t<T> result(a.nonZeros());
        int k = 0;
        for (int j = 0; j < a.outerSize(); ++j)
            for (typename Storage::InnerIterator it(a, j); it; ++it)
                result.mutable_data()[k++] = it.value();
        return result;
    }
};
template <class T, int O> void bind_matrix(py::module_ &m, const std::string &name) {
    using M = Matrix<T, O>;
    using S = typename M::Storage;
    py::class_<M>(m, name.c_str())
        .def(py::init<int, int>())
        .def_property_readonly("shape",
                               [](const M &m) { return py::make_tuple(m.a.rows(), m.a.cols()); })
        .def_property_readonly("nnz", [](const M &m) { return m.a.nonZeros(); })
        .def_property_readonly("is_compressed", [](const M &m) { return m.a.isCompressed(); })
        .def_property_readonly("inner_size", [](const M &m) { return m.a.innerSize(); })
        .def_property_readonly("outer_size", [](const M &m) { return m.a.outerSize(); })
        .def("coeff",
             [](const M &m, int r, int c) {
                 require(r >= 0 && r < m.a.rows() && c >= 0 && c < m.a.cols(),
                         "Coordinate out of bounds");
                 return m.a.coeff(r, c);
             })
        .def("set",
             [](M &m, int r, int c, T v) {
                 require(r >= 0 && r < m.a.rows() && c >= 0 && c < m.a.cols(),
                         "Coordinate out of bounds");
                 m.a.coeffRef(r, c) = v;
             })
        .def("gather",
             [](const M &m, Array<int> rows, Array<int> columns) {
                 require(rows.ndim() == 1 && columns.ndim() == 1 && rows.size() == columns.size(),
                         "Coordinate lengths differ");
                 py::array_t<T> result(rows.size());
                 for (py::ssize_t k = 0; k < rows.size(); ++k) {
                     int r = rows.data()[k], c = columns.data()[k];
                     require(r >= 0 && r < m.a.rows() && c >= 0 && c < m.a.cols(),
                             "Coordinate out of bounds");
                     result.mutable_data()[k] = m.a.coeff(r, c);
                 }
                 return result;
             })
        .def("assign",
             [](M &m, Array<int> rows, Array<int> columns, Array<T> values) {
                 require(rows.ndim() == 1 && columns.ndim() == 1 && values.ndim() == 1 &&
                             rows.size() == columns.size() && rows.size() == values.size(),
                         "Coordinate lengths differ");
                 for (py::ssize_t k = 0; k < rows.size(); ++k)
                     require(rows.data()[k] >= 0 && rows.data()[k] < m.a.rows() &&
                                 columns.data()[k] >= 0 && columns.data()[k] < m.a.cols(),
                             "Coordinate out of bounds");
                 for (py::ssize_t k = 0; k < rows.size(); ++k)
                     m.a.coeffRef(rows.data()[k], columns.data()[k]) = values.data()[k];
             })
        .def("assemble", [](M &m, Array<int> r, Array<int> c,
                            Array<T> v) { m.a = sparse_copy<T>(m.a.rows(), m.a.cols(), r, c, v); })
        .def("coordinates", &M::coordinates)
        .def("values", &M::values)
        .def("copy", [](const M &m) { return M(m.a); })
        .def("add",
             [](const M &x, const M &y) {
                 require(x.a.rows() == y.a.rows() && x.a.cols() == y.a.cols(), "Shape mismatch");
                 return M(S(x.a + y.a));
             })
        .def("subtract",
             [](const M &x, const M &y) {
                 require(x.a.rows() == y.a.rows() && x.a.cols() == y.a.cols(), "Shape mismatch");
                 return M(S(x.a - y.a));
             })
        .def("multiply",
             [](const M &x, const M &y) {
                 require(x.a.rows() == y.a.rows() && x.a.cols() == y.a.cols(), "Shape mismatch");
                 return M(S(x.a.cwiseProduct(y.a)));
             })
        .def("matmul",
             [](const M &x, const M &y) {
                 require(x.a.cols() == y.a.rows(), "Product shape mismatch");
                 return M(S(x.a * y.a));
             })
        .def("dense_matmul",
             [](const M &x, Array<T> y) {
                 auto b = dense_copy<T>(y);
                 require(x.a.cols() == b.rows(), "Product shape mismatch");
                 Dense<T> result;
                 {
                     py::gil_scoped_release release;
                     result = x.a * b;
                 }
                 return array_copy(result);
             })
        .def("transpose", [](const M &m) { return M(S(m.a.transpose())); })
        .def("adjoint", [](const M &m) { return M(S(m.a.adjoint())); })
        .def("compress", [](M &m) { m.a.makeCompressed(); })
        .def("uncompress", [](M &m) { m.a.uncompress(); })
        .def("squeeze",
             [](M &m) {
                 m.a.makeCompressed();
                 m.a.data().squeeze();
             })
        .def("reserve",
             [](M &m, int count) {
                 require(count >= 0, "Capacity must be nonnegative");
                 m.a.makeCompressed();
                 m.a.reserve(count);
             })
        .def("reserve_per_outer",
             [](M &m, Array<int> counts) {
                 require(counts.ndim() == 1 && counts.size() == m.a.outerSize(),
                         "Expected one capacity per outer vector");
                 for (auto i = 0; i < counts.size(); ++i)
                     require(counts.data()[i] >= 0, "Capacity must be nonnegative");
                 Eigen::Map<const Eigen::VectorXi> v(counts.data(), counts.size());
                 m.a.reserve(v);
             })
        .def("set_zero", [](M &m) { m.a.setZero(); })
        .def("resize",
             [](M &m, int r, int c, bool preserve) {
                 require(r >= 0 && c >= 0, "Negative dimensions");
                 if (preserve)
                     m.a.conservativeResize(r, c);
                 else
                     m.a.resize(r, c);
             })
        .def("prune",
             [](M &m, double tolerance) {
                 require(std::isfinite(tolerance) && tolerance >= 0, "Invalid tolerance");
                 m.a.prune([tolerance](int, int, const T &v) {
                     if constexpr (std::is_integral_v<T>) {
                         using U = std::make_unsigned_t<T>;
                         U magnitude = v < 0 ? U(0) - U(v) : U(v);
                         return tolerance < double(std::numeric_limits<U>::max()) &&
                                magnitude > static_cast<U>(tolerance);
                     } else {
                         return !(std::abs(v) <= tolerance);
                     }
                 });
             })
        .def("norm",
             [](const M &m) {
                 return m.a.template cast<typename Eigen::NumTraits<T>::NonInteger>().norm();
             })
        .def("compressed_arrays",
             [](const M &m) {
                 S a = m.a;
                 a.makeCompressed();
                 py::array_t<int> inner(a.nonZeros()), outer(a.outerSize() + 1);
                 py::array_t<T> values(a.nonZeros());
                 std::copy_n(a.innerIndexPtr(), a.nonZeros(), inner.mutable_data());
                 std::copy_n(a.valuePtr(), a.nonZeros(), values.mutable_data());
                 std::copy_n(a.outerIndexPtr(), a.outerSize() + 1, outer.mutable_data());
                 return py::make_tuple(values, inner, outer);
             })
        .def("triangular",
             [](const M &m, bool lower, bool unit) {
                 require(m.a.rows() == m.a.cols() || !unit,
                         "Unit triangular matrices must be square");
                 S result = m.a;
                 result.prune([lower](int r, int c, const T &) { return lower ? r >= c : r <= c; });
                 if (unit)
                     for (int i = 0; i < result.rows(); ++i)
                         result.coeffRef(i, i) = T(1);
                 return M(std::move(result));
             })
        .def("selfadjoint", [](const M &m, bool lower) {
            require(m.a.rows() == m.a.cols(), "Self-adjoint matrices must be square");
            Sparse<T> col = m.a;
            Sparse<T> result;
            if (lower)
                result = col.template selfadjointView<Eigen::Lower>();
            else
                result = col.template selfadjointView<Eigen::Upper>();
            return M(S(result));
        });
}
template <class T> void bind_scalar(py::module_ &m, const char *name) {
    bind_matrix<T, Eigen::RowMajor>(m, std::string(name) + "_C");
    bind_matrix<T, Eigen::ColMajor>(m, std::string(name) + "_F");
}
} // namespace sppy
