#pragma once
#include "scalar_arithmetic.h"
#include <Eigen/SparseCore>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>
namespace sppy {
// Probe a few outer vectors. Ordered products keep the single-pass builder;
// larger unordered products amortize a symbolic pass that avoids one full
// output conversion. This choice affects work only, never stored coefficients.
template <class S> bool product_needs_sorting(const S &outer, const S &inner, int width) {
    std::vector<uint8_t> seen(width, 0);
    std::vector<int> touched;
    for (int j = 0; j < std::min<int>(16, int(outer.outerSize())); ++j) {
        int previous = -1;
        touched.clear();
        for (typename S::InnerIterator a(outer, j); a; ++a)
            for (typename S::InnerIterator b(inner, a.index()); b; ++b) {
                const int index = b.index();
                if (!seen[index]) {
                    if (index < previous)
                        return true;
                    previous = index;
                    seen[index] = 1;
                    touched.push_back(index);
                }
            }
        for (int index : touched)
            seen[index] = 0;
    }
    return false;
}
// Count unique output coordinates first, then scatter each computed outer
// vector into the opposite storage order. Outer vectors arrive in increasing
// order, so one final conversion produces a canonical result without sorting.
template <class S> S product_scatter(const S &left, const S &right) {
    using T = typename S::Scalar;
    using Opposite = Eigen::SparseMatrix<T, S::IsRowMajor ? Eigen::ColMajor : Eigen::RowMajor, int>;
    const S &outer = S::IsRowMajor ? left : right, &inner = S::IsRowMajor ? right : left;
    const int width = int(S::IsRowMajor ? right.cols() : left.rows());
    Opposite temp(left.rows(), right.cols());
    std::vector<int> marks(width, -1);
    int64_t total = 0;
    for (int j = 0; j < outer.outerSize(); ++j) {
        for (typename S::InnerIterator a(outer, j); a; ++a) {
            for (typename S::InnerIterator b(inner, a.index()); b; ++b) {
                const int index = b.index();
                if (marks[index] != j) {
                    marks[index] = j;
                    temp.outerIndexPtr()[index + 1]++;
                    total++;
                }
            }
        }
    }
    if (total > std::numeric_limits<int>::max())
        throw std::overflow_error("Too many entries");
    for (int k = 0; k < width; ++k)
        temp.outerIndexPtr()[k + 1] += temp.outerIndexPtr()[k];
    temp.resizeNonZeros(total);
    std::vector<int> cursor(temp.outerIndexPtr(), temp.outerIndexPtr() + width), touched(width);
    std::vector<T> sums(width);
    std::fill(marks.begin(), marks.end(), -1);
    for (int j = 0; j < outer.outerSize(); ++j) {
        int count = 0;
        for (typename S::InnerIterator a(outer, j); a; ++a) {
            for (typename S::InnerIterator b(inner, a.index()); b; ++b) {
                const int index = b.index();
                const T value = S::IsRowMajor ? product_value(a.value(), b.value())
                                              : product_value(b.value(), a.value());
                if (marks[index] != j) {
                    marks[index] = j;
                    sums[index] = value;
                    touched[count++] = index;
                } else
                    sums[index] = sum_value(sums[index], value);
            }
        }
        for (int k = 0; k < count; ++k) {
            const int index = touched[k], position = cursor[index]++;
            temp.innerIndexPtr()[position] = j;
            temp.valuePtr()[position] = sums[index];
        }
    }
    return S(temp);
}
} // namespace sppy
