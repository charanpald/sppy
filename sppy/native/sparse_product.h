#pragma once
#include "bitset_product.h"
#include "scalar_arithmetic.h"
#include <Eigen/SparseCore>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace sppy {
// Gustavson accumulation in either CSR or CSC order. Results remain canonical
// and retain structurally touched zeros, including exact cancellation.
template <class S> S sparse_product(const S &left, const S &right) {
    using T = typename S::Scalar;
    if (left.cols() != right.rows())
        throw std::invalid_argument("Product shape mismatch");
    S result(left.rows(), right.cols());
    if (!left.nonZeros() || !right.nonZeros())
        return result;
    const S &outer = S::IsRowMajor ? left : right;
    const S &inner = S::IsRowMajor ? right : left;
    const Eigen::Index width = result.innerSize();
    if (width >= 64 && width <= (1 << 20))
        return bitset_product(left, right);
    const int64_t estimate = std::min<int64_t>(int64_t(left.nonZeros()) * right.nonZeros() /
                                                       std::max<Eigen::Index>(1, left.cols()) +
                                                   std::min(left.rows(), right.cols()),
                                               int64_t(1) << 20);
    result.resizeNonZeros(Eigen::Index(estimate));
    int64_t allocated = estimate;
    const bool dense = width <= (1 << 20);
    std::vector<uint8_t> marks(dense ? width : 0, 0);
    std::vector<int> touched(dense ? width : 0);
    std::vector<T> sums(dense ? width : 0);
    std::unordered_map<int, T> hash;
    int64_t written = 0;
    bool canonical = true;
    for (int j = 0; j < outer.outerSize(); ++j) {
        int touched_count = 0;
        if (!dense) {
            touched.clear();
            hash.clear();
        }
        for (typename S::InnerIterator a(outer, j); a; ++a) {
            for (typename S::InnerIterator b(inner, a.index()); b; ++b) {
                const int index = b.index();
                const T value = S::IsRowMajor ? product_value(a.value(), b.value())
                                              : product_value(b.value(), a.value());
                if (dense) {
                    if (!marks[index]) {
                        marks[index] = 1;
                        sums[index] = value;
                        touched[touched_count++] = index;
                    } else {
                        sums[index] = sum_value(sums[index], value);
                    }
                } else {
                    auto item = hash.try_emplace(index, value);
                    if (item.second)
                        touched.push_back(index);
                    else
                        item.first->second = sum_value(item.first->second, value);
                }
            }
        }
        if (!dense)
            touched_count = int(touched.size());
        const int64_t begin = written;
        written += touched_count;
        if (written > std::numeric_limits<int>::max())
            throw std::overflow_error("Too many stored product entries");
        if (!std::is_sorted(touched.begin(), touched.begin() + touched_count)) {
            if (dense)
                canonical = false;
            else
                std::sort(touched.begin(), touched.end());
        }
        if (written > allocated) {
            allocated = std::min<int64_t>(std::numeric_limits<int>::max(),
                                          std::max(written, allocated + allocated / 2 + 1));
            result.resizeNonZeros(Eigen::Index(allocated));
        }
        result.outerIndexPtr()[j] = int(begin);
        int position = int(begin);
        for (int k = 0; k < touched_count; ++k) {
            const int index = touched[k];
            result.innerIndexPtr()[position] = index;
            result.valuePtr()[position++] = dense ? sums[index] : hash.at(index);
            if (dense)
                marks[index] = 0;
        }
    }
    result.outerIndexPtr()[result.outerSize()] = int(written);
    result.resizeNonZeros(Eigen::Index(written));
    if (!canonical) {
        // Counting/scatter through the opposite storage order sorts all inner
        // indices in linear time instead of sorting every outer vector.
        using Opposite =
            Eigen::SparseMatrix<T, S::IsRowMajor ? Eigen::ColMajor : Eigen::RowMajor, int>;
        return S(Opposite(result));
    }
    return result;
}
} // namespace sppy
