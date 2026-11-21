#pragma once
#include "matrix_storage.h"
#include "sparse_product.h"
namespace sppy {
template <class S> bool all_finite(const S &a) {
    using T = typename S::Scalar;
    if constexpr (std::is_integral_v<T>)
        return true;
    if (a.isCompressed())
        return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(a.valuePtr(), a.nonZeros())
            .allFinite();
    for (int j = 0; j < a.outerSize(); ++j)
        for (typename S::InnerIterator it(a, j); it; ++it)
            if (!Eigen::numext::isfinite(it.value()))
                return false;
    return true;
}

template <class S> auto sparse_sum(const S &a, int axis) {
    using T = typename S::Scalar;
    using R = std::conditional_t<std::is_integral_v<T>, int64_t, T>;
    require(axis >= -1 && axis <= 1, "Invalid sum axis");
    const Eigen::Index size = axis == -1 ? 1 : axis == 0 ? a.cols() : a.rows();
    py::array_t<R> output(size);
    R *values = output.mutable_data();
    std::fill_n(values, size, R(0));
    if (axis == -1 && a.isCompressed() && !std::is_integral_v<T>) {
        values[0] =
            Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(a.valuePtr(), a.nonZeros()).sum();
    } else {
        for (int j = 0; j < a.outerSize(); ++j)
            for (typename S::InnerIterator it(a, j); it; ++it) {
                const auto index = axis == -1 ? 0 : axis == 0 ? it.col() : it.row();
                values[index] = sum_value(values[index], R(it.value()));
            }
    }
    return output;
}

// Equal-pattern coefficient operations are a contiguous transform rather than
// a merge. The result still owns its buffers and keeps explicitly stored zeros.
template <int Operation, class S> S coefficient_binary(const S &a, const S &b) {
    using T = typename S::Scalar;
    const bool same =
        &a == &b ||
        (a.isCompressed() && b.isCompressed() && a.nonZeros() == b.nonZeros() &&
         std::equal(a.outerIndexPtr(), a.outerIndexPtr() + a.outerSize() + 1, b.outerIndexPtr()) &&
         std::equal(a.innerIndexPtr(), a.innerIndexPtr() + a.nonZeros(), b.innerIndexPtr()));
    if (same && a.isCompressed() && b.isCompressed()) {
        S result = a;
        for (Eigen::Index k = 0; k < a.nonZeros(); ++k) {
            if constexpr (Operation == 0)
                result.valuePtr()[k] = sum_value(a.valuePtr()[k], b.valuePtr()[k]);
            else if constexpr (Operation == 1) {
                if constexpr (std::is_integral_v<T>)
                    result.valuePtr()[k] =
                        sum_value(a.valuePtr()[k], product_value(T(-1), b.valuePtr()[k]));
                else
                    result.valuePtr()[k] = a.valuePtr()[k] - b.valuePtr()[k];
            } else
                result.valuePtr()[k] = product_value(a.valuePtr()[k], b.valuePtr()[k]);
        }
        return result;
    }
    if constexpr (std::is_integral_v<T>) {
        // Eigen's default coefficient expressions can overflow signed integers.
        // Merge mismatched/uncompressed patterns using modular scalar helpers.
        S result(a.rows(), a.cols());
        result.reserve(Eigen::Index(std::min<int64_t>(int64_t(a.nonZeros()) + b.nonZeros(),
                                                      std::numeric_limits<int>::max())));
        for (int outer = 0; outer < a.outerSize(); ++outer) {
            result.startVec(outer);
            typename S::InnerIterator left(a, outer), right(b, outer);
            while (left || right) {
                if (left && right && left.index() == right.index()) {
                    T value;
                    if constexpr (Operation == 0)
                        value = sum_value(left.value(), right.value());
                    else if constexpr (Operation == 1)
                        value = sum_value(left.value(), product_value(T(-1), right.value()));
                    else
                        value = product_value(left.value(), right.value());
                    result.insertBackByOuterInner(outer, left.index()) = value;
                    ++left;
                    ++right;
                } else if (left && (!right || left.index() < right.index())) {
                    if constexpr (Operation != 2)
                        result.insertBackByOuterInner(outer, left.index()) = left.value();
                    ++left;
                } else {
                    if constexpr (Operation != 2)
                        result.insertBackByOuterInner(outer, right.index()) =
                            Operation == 0 ? right.value() : product_value(T(-1), right.value());
                    ++right;
                }
            }
        }
        result.finalize();
        return result;
    } else if constexpr (Operation == 0)
        return S(a + b);
    else if constexpr (Operation == 1)
        return S(a - b);
    else
        return S(a.cwiseProduct(b));
}

// A fixed RHS block avoids dynamic inner-loop setup and makes complex packet
// arithmetic available without converting both operands to column-major.
template <int Columns, class S>
void dense_block_product(const S &a, const typename S::Scalar *input, typename S::Scalar *output) {
    using T = typename S::Scalar;
    using Row = Eigen::Matrix<T, 1, Columns>;
    using ConstRow = Eigen::Map<const Row>;
    using OutputRow = Eigen::Map<Row>;
    std::fill_n(output, a.rows() * Columns, T(0));
    if constexpr (S::IsRowMajor) {
        auto row_product = [&](int row) {
            Row destination = Row::Zero();
            for (typename S::InnerIterator it(a, row); it; ++it) {
                if constexpr (std::is_integral_v<T>) {
                    for (int k = 0; k < Columns; ++k)
                        destination[k] =
                            sum_value(destination[k],
                                      product_value(it.value(), input[it.col() * Columns + k]));
                } else if constexpr (Eigen::NumTraits<T>::IsComplex) {
                    const auto real = it.value().real(), imag = it.value().imag();
                    for (int k = 0; k < Columns; ++k) {
                        const auto value = input[it.col() * Columns + k];
                        destination[k] += T(real * value.real() - imag * value.imag(),
                                            real * value.imag() + imag * value.real());
                    }
                } else {
                    destination += it.value() * ConstRow(input + it.col() * Columns);
                }
            }
            OutputRow(output + row * Columns) = destination;
        };
        const int threads = Eigen::nbThreads();
        (void)threads;
#ifdef _OPENMP
        if (threads > 1 && a.rows() >= 256) {
#pragma omp parallel for num_threads(threads)
            for (int row = 0; row < a.rows(); ++row)
                row_product(row);
            return;
        }
#endif
        for (int row = 0; row < a.rows(); ++row)
            row_product(row);
    } else {
        for (int col = 0; col < a.cols(); ++col)
            for (typename S::InnerIterator it(a, col); it; ++it) {
                OutputRow destination(output + it.row() * Columns);
                if constexpr (std::is_integral_v<T>) {
                    for (int k = 0; k < Columns; ++k)
                        destination[k] = sum_value(
                            destination[k], product_value(it.value(), input[col * Columns + k]));
                } else if constexpr (Eigen::NumTraits<T>::IsComplex) {
                    const auto real = it.value().real(), imag = it.value().imag();
                    for (int k = 0; k < Columns; ++k) {
                        const auto value = input[col * Columns + k];
                        destination[k] += T(real * value.real() - imag * value.imag(),
                                            real * value.imag() + imag * value.real());
                    }
                } else {
                    destination += it.value() * ConstRow(input + col * Columns);
                }
            }
    }
}
} // namespace sppy
