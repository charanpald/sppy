#pragma once
#include "matrix_storage.h"
namespace sppy {
template <bool Lower, bool Unit, class S, class T>
void triangular_vector_in_place(const S &a, T *values) {
    for (int step = 0; step < a.rows(); ++step) {
        const int row = Lower ? step : int(a.rows()) - 1 - step;
        T value = values[row], diagonal = Unit ? T(1) : T(0);
        for (int k = a.outerIndexPtr()[row]; k < a.outerIndexPtr()[row + 1]; ++k) {
            const int col = a.innerIndexPtr()[k];
            if (col == row) {
                if constexpr (!Unit)
                    diagonal = a.valuePtr()[k];
                if constexpr (Lower)
                    break;
            } else if constexpr (Lower) {
                if (col > row)
                    break;
                value -= a.valuePtr()[k] * values[col];
            } else {
                if (col > row)
                    value -= a.valuePtr()[k] * values[col];
            }
        }
        if constexpr (Unit)
            values[row] = value;
        else {
            if (diagonal == T(0))
                throw std::invalid_argument("Singular triangular diagonal");
            values[row] = value / diagonal;
        }
    }
}
template <class T, class S>
auto triangular_solve_storage(const S &a, Array<T> rhs, bool lower, bool unit) {
    const auto r = a.rows(), c = a.cols();
    auto b = dense_copy<T>(rhs);
    require(r == c && r == b.rows(), "Invalid triangular system dimensions");
    require(b.allFinite(), "Right-hand side must be finite");
    for (int j = 0; j < a.nonZeros(); ++j)
        require(Eigen::numext::isfinite(a.valuePtr()[j]), "Matrix contains nonfinite values");
    if constexpr (S::IsRowMajor) {
        if (a.isCompressed() && b.cols() == 1) {
            {
                py::gil_scoped_release release;
                if (lower) {
                    if (unit)
                        triangular_vector_in_place<true, true>(a, b.data());
                    else
                        triangular_vector_in_place<true, false>(a, b.data());
                } else {
                    if (unit)
                        triangular_vector_in_place<false, true>(a, b.data());
                    else
                        triangular_vector_in_place<false, false>(a, b.data());
                }
            }
            return array_copy(b);
        }
    }
    if (!unit)
        for (int i = 0; i < r; ++i)
            require(a.coeff(i, i) != T(0), "Singular triangular diagonal");
    ColDense<T> x;
    {
        py::gil_scoped_release release;
        if (lower) {
            if (unit)
                x = a.template triangularView<Eigen::UnitLower>().solve(b);
            else
                x = a.template triangularView<Eigen::Lower>().solve(b);
        } else {
            if (unit)
                x = a.template triangularView<Eigen::UnitUpper>().solve(b);
            else
                x = a.template triangularView<Eigen::Upper>().solve(b);
        }
    }
    return array_copy(x);
}
template <class T> void bind_triangular(py::module_ &m, const char *name) {
    m.def((std::string("triangular_solve_matrix_") + name).c_str(),
          [](py::handle backend, Array<T> rhs, bool lower, bool unit) {
              return triangular_solve_storage<T>(solver_matrix<T, Eigen::RowMajor>(backend), rhs,
                                                 lower, unit);
          });
    m.def((std::string("triangular_solve_") + name).c_str(),
          [](int r, int c, Array<int> ri, Array<int> ci, Array<T> values, Array<T> rhs, bool lower,
             bool unit) {
              return triangular_solve_storage<T>(sparse_copy<T>(r, c, ri, ci, values), rhs, lower,
                                                 unit);
          });
}
} // namespace sppy
