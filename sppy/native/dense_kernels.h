#pragma once
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <algorithm>
#include <cmath>
namespace sppy {
// Multiple independent accumulators reduce the dependency chain without
// enabling reassociation/fast-math for the rest of the extension.
template <class T> T contiguous_norm(const T *v, Eigen::Index size) {
    T a = 0, b = 0, c = 0, d = 0, e = 0, f = 0, g = 0, h = 0;
    Eigen::Index k = 0;
    for (; k + 7 < size; k += 8) {
        a += v[k] * v[k];
        b += v[k + 1] * v[k + 1];
        c += v[k + 2] * v[k + 2];
        d += v[k + 3] * v[k + 3];
        e += v[k + 4] * v[k + 4];
        f += v[k + 5] * v[k + 5];
        g += v[k + 6] * v[k + 6];
        h += v[k + 7] * v[k + 7];
    }
    T total = ((a + b) + (c + d)) + ((e + f) + (g + h));
    for (; k < size; ++k)
        total += v[k] * v[k];
    return std::sqrt(total);
}
// Keep the generic vector evaluator out of the larger block-dispatch lambda:
// it gives the compiler a small, focused loop on the fallback path.
template <class S>
EIGEN_DONT_INLINE void mapped_vector_product(const S &a, const typename S::Scalar *input,
                                             typename S::Scalar *output) {
    using T = typename S::Scalar;
    using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    Eigen::Map<const Vector> rhs(input, a.cols());
    Eigen::Map<Vector> result(output, a.rows());
    result.noalias() = a * rhs;
}
template <class S> bool narrow_vector_rows(const S &a) {
    for (int row = 0; row < std::min<int>(16, int(a.rows())); ++row) {
        int first = -1, last = -1;
        for (typename S::InnerIterator it(a, row); it; ++it) {
            if (first < 0)
                first = int(it.index());
            last = int(it.index());
        }
        if (last - first > 32)
            return false;
    }
    return true;
}
template <class S>
void raw_vector_product(const S &a, const typename S::Scalar *input, typename S::Scalar *output) {
    static_assert(S::IsRowMajor, "This kernel traverses CSR rows");
    using T = typename S::Scalar;
    for (int row = 0; row < a.rows(); ++row) {
        T total = 0;
        for (typename S::InnerIterator it(a, row); it; ++it)
            total += it.value() * input[it.col()];
        output[row] = total;
    }
}
template <int Columns, class S>
void complex_row_block(const S &a, const typename S::Scalar *input, typename S::Scalar *output) {
    using T = typename S::Scalar;
    using R = typename Eigen::NumTraits<T>::Real;
    using Dense = Eigen::Matrix<T, Eigen::Dynamic, Columns, Eigen::RowMajor>;
    using RealDense = Eigen::Matrix<R, Eigen::Dynamic, Columns, Eigen::RowMajor>;
    using Row = Eigen::Matrix<R, 1, Columns>;
    Eigen::Map<const Dense> source(input, a.cols(), Columns);
    Eigen::Map<Dense> result(output, a.rows(), Columns);
    RealDense real = source.real(), imag = source.imag();
    for (int row = 0; row < a.rows(); ++row) {
        Row real_sum = Row::Zero(), imag_sum = Row::Zero();
        for (typename S::InnerIterator it(a, row); it; ++it) {
            const auto r = it.value().real(), i = it.value().imag();
            real_sum += r * real.row(it.col()) - i * imag.row(it.col());
            imag_sum += r * imag.row(it.col()) + i * real.row(it.col());
        }
        result.row(row).real() = real_sum;
        result.row(row).imag() = imag_sum;
    }
}
} // namespace sppy
