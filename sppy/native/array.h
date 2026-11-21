#pragma once
#include "array_algorithms.h"
#include "array_ops.h"
#include "dense_kernels.h"
#include "matrix_storage.h"
#include "sparse_product.h"
#include <type_traits>
namespace sppy {
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
        .def("gather_signed",
             [](const M &m, Array<int64_t> rows, Array<int64_t> columns) {
                 require(rows.ndim() == 1 && columns.ndim() == 1 && rows.size() == columns.size(),
                         "Coordinate lengths differ");
                 py::array_t<T> result(rows.size());
                 for (py::ssize_t k = 0; k < rows.size(); ++k) {
                     int64_t r = rows.data()[k], c = columns.data()[k];
                     if (r < 0)
                         r += m.a.rows();
                     if (c < 0)
                         c += m.a.cols();
                     require(r >= 0 && r < m.a.rows() && c >= 0 && c < m.a.cols(),
                             "Index out of range");
                     result.mutable_data()[k] = m.a.coeff(int(r), int(c));
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
        .def("assemble",
             [](M &m, Array<int> r, Array<int> c, Array<T> v) {
                 m.a = assemble_triplets<S>(m.a.rows(), m.a.cols(), r, c, v);
             })
        .def_static("from_compressed",
                    [](int rows, int cols, Array<T> values, Array<int> indices, Array<int> indptr) {
                        return M(import_compressed<S>(rows, cols, values, indices, indptr));
                    })
        .def("slice_basic",
             [](const M &m, int64_t rs, int64_t rt, int rn, int64_t cs, int64_t ct, int cn,
                int mode) -> py::object {
                 if (mode == 0)
                     return py::cast(M(basic_slice<S>(m.a, {rs, rt, rn}, {cs, ct, cn}, mode)));
                 using V = Matrix<T, Eigen::ColMajor>;
                 return py::cast(
                     V(basic_slice<typename V::Storage>(m.a, {rs, rt, rn}, {cs, ct, cn}, mode)));
             })
        .def("to_row_major",
             [](const M &m) {
                 return Matrix<T, Eigen::RowMajor>(
                     Eigen::SparseMatrix<T, Eigen::RowMajor, int>(m.a));
             })
        .def("to_column_major",
             [](const M &m) { return Matrix<T, Eigen::ColMajor>(Sparse<T>(m.a)); })
        .def("coordinates", &M::coordinates)
        .def("values", &M::values)
        .def("copy", [](const M &m) { return M(m.a); })
        .def("add",
             [](const M &x, const M &y) {
                 require(x.a.rows() == y.a.rows() && x.a.cols() == y.a.cols(), "Shape mismatch");
                 return M(coefficient_binary<0>(x.a, y.a));
             })
        .def("subtract",
             [](const M &x, const M &y) {
                 require(x.a.rows() == y.a.rows() && x.a.cols() == y.a.cols(), "Shape mismatch");
                 return M(coefficient_binary<1>(x.a, y.a));
             })
        .def("multiply",
             [](const M &x, const M &y) {
                 require(x.a.rows() == y.a.rows() && x.a.cols() == y.a.cols(), "Shape mismatch");
                 return M(coefficient_binary<2>(x.a, y.a));
             })
        .def("matmul",
             [](const M &x, const M &y) {
                 require(x.a.cols() == y.a.rows(), "Product shape mismatch");
                 return M(sparse_product(x.a, y.a));
             })
        .def("dense_matmul",
             [](const M &x, Array<T> y) {
                 require(y.ndim() == 2 && x.a.cols() == y.shape(0), "Product shape mismatch");
                 py::array_t<T> output({x.a.rows(), Eigen::Index(y.shape(1))});
                 const Eigen::Index columns = y.shape(1);
                 if (!columns)
                     return output;
                 const T *input_data = y.data();
                 T *output_data = output.mutable_data();
                 if constexpr (std::is_integral_v<T>) {
                     py::gil_scoped_release release;
                     if (columns == 8) {
                         dense_block_product<8>(x.a, input_data, output_data);
                     } else {
                         std::fill_n(output_data, x.a.rows() * columns, T(0));
                         for (int outer = 0; outer < x.a.outerSize(); ++outer)
                             for (typename S::InnerIterator it(x.a, outer); it; ++it)
                                 for (Eigen::Index k = 0; k < columns; ++k) {
                                     auto &value = output_data[it.row() * columns + k];
                                     value = sum_value(
                                         value, product_value(it.value(),
                                                              input_data[it.col() * columns + k]));
                                 }
                     }
                     return output;
                 }
                 Eigen::Map<const Dense<T>> input(input_data, y.shape(0), columns);
                 Eigen::Map<Dense<T>> result(output_data, x.a.rows(), columns);
                 bool block = columns == 8;
                 if constexpr (Eigen::NumTraits<T>::IsComplex)
                     block = block && all_finite(x.a) && input.allFinite();
                 if (block) {
                     py::gil_scoped_release release;
                     if constexpr (Eigen::NumTraits<T>::IsComplex) {
                         if (x.a.isCompressed() &&
                             complex_sparse_block(x.a.rows(), x.a.cols(), O == Eigen::RowMajor,
                                                  x.a.valuePtr(), x.a.innerIndexPtr(),
                                                  x.a.outerIndexPtr(), input_data, output_data))
                             return output;
                     }
                     if constexpr (Eigen::NumTraits<T>::IsComplex && O == Eigen::RowMajor) {
                         complex_row_block<8>(x.a, input_data, output_data);
                     } else {
                         dense_block_product<8>(x.a, input_data, output_data);
                     }
                     return output;
                 }
                 if (columns == 1) {
                     if constexpr (Eigen::NumTraits<T>::IsComplex && O == Eigen::RowMajor) {
                         if (x.a.isCompressed() && panel_simd_available()) {
                             bool complete;
                             {
                                 py::gil_scoped_release release;
                                 complete = complex_csr_vector(
                                     x.a.rows(), x.a.valuePtr(), x.a.innerIndexPtr(),
                                     x.a.outerIndexPtr(), input_data, output_data);
                             }
                             if (complete)
                                 return output;
                         }
                     }
                     if constexpr (std::is_floating_point_v<T> && O == Eigen::RowMajor) {
                         if (narrow_vector_rows(x.a)) {
                             py::gil_scoped_release release;
                             raw_vector_product(x.a, input_data, output_data);
                             return output;
                         }
                     }
                     {
                         py::gil_scoped_release release;
                         mapped_vector_product(x.a, input_data, output_data);
                     }
                     return output;
                 }
                 if constexpr (Eigen::NumTraits<T>::IsComplex) {
                     if (columns > 1) {
                         // Eigen's column-major dense operand gives its complex
                         // product kernel better SIMD access. Write the result
                         // directly to NumPy while retaining that specialization.
                         auto right = dense_copy<T>(y);
                         {
                             py::gil_scoped_release release;
                             result.noalias() = x.a * right;
                         }
                         return output;
                     }
                 }
                 {
                     py::gil_scoped_release release;
                     if constexpr (O == Eigen::RowMajor) {
                         if (columns != 1)
                             result.setZero();
                         const int threads = Eigen::nbThreads();
                         (void)threads;
                         auto row_product = [&](int row) {
                             for (typename S::InnerIterator it(x.a, row); it; ++it)
                                 result.row(row) += it.value() * input.row(it.col());
                         };
#ifdef _OPENMP
                         if (threads > 1 && x.a.rows() >= 256) {
#pragma omp parallel for num_threads(threads)
                             for (int row = 0; row < x.a.rows(); ++row)
                                 row_product(row);
                         } else
#endif
                         {
                             for (int row = 0; row < x.a.rows(); ++row)
                                 row_product(row);
                         }
                     } else {
                         result.setZero();
                         for (int col = 0; col < x.a.cols(); ++col)
                             for (typename S::InnerIterator it(x.a, col); it; ++it)
                                 result.row(it.row()) += it.value() * input.row(col);
                     }
                 }
                 return output;
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
                 if (tolerance == 0) {
                     m.a.prune([](int, int, const T &v) { return v != T(0); });
                     return;
                 }
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
        .def("all_finite", [](const M &m) { return all_finite(m.a); })
        .def("sum", [](const M &m, int axis) { return sparse_sum(m.a, axis); })
        .def("norm",
             [](const M &m) {
                 using P = typename Eigen::NumTraits<T>::NonInteger;
                 if constexpr (std::is_floating_point_v<T>) {
                     if (m.a.isCompressed())
                         return contiguous_norm(m.a.valuePtr(), m.a.nonZeros());
                 }
                 if (m.a.isCompressed()) {
                     Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> values(m.a.valuePtr(),
                                                                                  m.a.nonZeros());
                     return values.template cast<P>().norm();
                 }
                 return m.a.template cast<P>().norm();
             })
        .def("compressed_arrays",
             [](const M &m) {
                 S compressed;
                 const S *source = &m.a;
                 if (!m.a.isCompressed()) {
                     compressed = m.a;
                     compressed.makeCompressed();
                     source = &compressed;
                 }
                 const S &a = *source;
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
