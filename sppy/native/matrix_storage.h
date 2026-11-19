#pragma once
#include "common.h"
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
        if (a.isCompressed()) {
            std::copy_n(a.valuePtr(), a.nonZeros(), result.mutable_data());
        } else {
            int k = 0;
            for (int j = 0; j < a.outerSize(); ++j)
                for (typename Storage::InnerIterator it(a, j); it; ++it)
                    result.mutable_data()[k++] = it.value();
        }
        return result;
    }
};

// Solvers snapshot existing storage instead of exporting coordinates and
// rebuilding it through a duplicate-summing triplet assembly.
template <class T, int Order = Eigen::ColMajor>
Eigen::SparseMatrix<T, Order, int> solver_matrix(py::handle backend) {
    using S = Eigen::SparseMatrix<T, Order, int>;
    if (py::isinstance<Matrix<T, Eigen::ColMajor>>(backend)) {
        S result = py::cast<const Matrix<T, Eigen::ColMajor> &>(backend).a;
        result.makeCompressed();
        return result;
    }
    if (py::isinstance<Matrix<T, Eigen::RowMajor>>(backend)) {
        S result = py::cast<const Matrix<T, Eigen::RowMajor> &>(backend).a;
        result.makeCompressed();
        return result;
    }
    throw py::type_error("Matrix storage has an incompatible scalar type");
}
} // namespace sppy
