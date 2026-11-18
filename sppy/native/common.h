#pragma once
#include <Eigen/Sparse>
#include <complex>
#include <limits>
#include <memory>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
namespace py = pybind11;
namespace sppy {
template <class T> using Array = py::array_t<T, py::array::c_style | py::array::forcecast>;
template <class T> using Dense = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
template <class T> using ColDense = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template <class T> using Sparse = Eigen::SparseMatrix<T, Eigen::ColMajor, int>;
inline void require(bool condition, const char *message) {
    if (!condition)
        throw py::value_error(message);
}
template <class Derived> auto array_copy(const Eigen::MatrixBase<Derived> &value) {
    using T = typename Derived::Scalar;
    py::array_t<T> result({value.rows(), value.cols()});
    Eigen::Map<Dense<T>>(result.mutable_data(), value.rows(), value.cols()) = value;
    return result;
}
template <class T> ColDense<T> dense_copy(Array<T> value) {
    require(value.ndim() == 2, "Expected a two-dimensional dense array");
    return Eigen::Map<const Dense<T>>(value.data(), value.shape(0), value.shape(1));
}
template <class T>
Sparse<T> sparse_copy(int rows, int cols, Array<int> ri, Array<int> ci, Array<T> values) {
    require(rows >= 0 && cols >= 0, "Negative matrix dimensions");
    require(ri.ndim() == 1 && ci.ndim() == 1 && values.ndim() == 1,
            "Coordinates and values must be vectors");
    require(ri.size() == ci.size() && ri.size() == values.size(),
            "Coordinate and value lengths differ");
    require(values.size() <= std::numeric_limits<int>::max(), "Too many stored entries");
    std::vector<Eigen::Triplet<T>> entries;
    entries.reserve(values.size());
    for (py::ssize_t k = 0; k < values.size(); ++k) {
        require(ri.data()[k] >= 0 && ri.data()[k] < rows && ci.data()[k] >= 0 &&
                    ci.data()[k] < cols,
                "Coordinate out of bounds");
        entries.emplace_back(ri.data()[k], ci.data()[k], values.data()[k]);
    }
    Sparse<T> result(rows, cols);
    result.setFromTriplets(entries.begin(), entries.end());
    result.makeCompressed();
    return result;
}
template <class T> py::tuple export_sparse(const Sparse<T> &a) {
    py::array_t<int> rows(a.nonZeros()), cols(a.nonZeros());
    py::array_t<T> values(a.nonZeros());
    int k = 0;
    for (int j = 0; j < a.outerSize(); ++j)
        for (typename Sparse<T>::InnerIterator it(a, j); it; ++it, ++k) {
            rows.mutable_data()[k] = it.row();
            cols.mutable_data()[k] = it.col();
            values.mutable_data()[k] = it.value();
        }
    return py::make_tuple(a.rows(), a.cols(), rows, cols, values);
}
} // namespace sppy
