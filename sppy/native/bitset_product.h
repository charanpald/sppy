#pragma once
#include "scalar_arithmetic.h"
#include <Eigen/SparseCore>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
namespace sppy {
// Three levels of occupancy bits enumerate only touched inner coordinates in
// ascending order. Arithmetic stays in traversal order; no output sort or
// storage-order conversion is required. Each occupied word is reset lazily.

inline int first_set_bit(uint64_t value) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(value);
#else
    int bit = 0;
    while (!(value & 1)) {
        value >>= 1;
        ++bit;
    }
    return bit;
#endif
}
template <class S> S bitset_product(const S &left, const S &right) {
    using T = typename S::Scalar;
    S result(left.rows(), right.cols());
    const S &outer = S::IsRowMajor ? left : right, &inner = S::IsRowMajor ? right : left;
    int width = int(result.innerSize());
    size_t words = (size_t(width) + 63) / 64, groups = (words + 63) / 64,
           topwords = (groups + 63) / 64;
    std::vector<uint64_t> bits(words, 0), summary(groups, 0), top(topwords, 0);
    std::vector<T> sums(width);
    int64_t allocated = std::min<int64_t>(int64_t(left.nonZeros()) * right.nonZeros() /
                                                  std::max<Eigen::Index>(1, left.cols()) +
                                              std::min(left.rows(), right.cols()),
                                          1 << 20);
    result.resizeNonZeros(allocated);
    int64_t written = 0;
    for (int j = 0; j < outer.outerSize(); ++j) {
        int count = 0;
        for (typename S::InnerIterator a(outer, j); a; ++a)
            for (typename S::InnerIterator b(inner, a.index()); b; ++b) {
                const int i = int(b.index());
                const size_t word = size_t(i) >> 6;
                const uint64_t flag = uint64_t(1) << (i & 63);
                T value = S::IsRowMajor ? product_value(a.value(), b.value())
                                        : product_value(b.value(), a.value());
                if (!(bits[word] & flag)) {
                    bits[word] |= flag;
                    summary[word >> 6] |= uint64_t(1) << (word & 63);
                    top[word >> 12] |= uint64_t(1) << ((word >> 6) & 63);
                    sums[i] = value;
                    count++;
                } else
                    sums[i] = sum_value(sums[i], value);
            }
        const int64_t begin = written;
        written += count;
        if (written > std::numeric_limits<int>::max())
            throw std::overflow_error("Too many entries");
        if (written > allocated) {
            allocated = std::min<int64_t>(std::numeric_limits<int>::max(),
                                          std::max(written, allocated + allocated / 2 + 1));
            result.resizeNonZeros(allocated);
        }
        result.outerIndexPtr()[j] = int(begin);
        int position = int(begin);
        for (size_t t = 0; t < topwords; ++t) {
            uint64_t tbits = top[t];
            while (tbits) {
                size_t group = t * 64 + first_set_bit(tbits);
                uint64_t sbits = summary[group];
                while (sbits) {
                    size_t word = group * 64 + first_set_bit(sbits);
                    uint64_t data = bits[word];
                    while (data) {
                        int i = int(word * 64) + first_set_bit(data);
                        result.innerIndexPtr()[position] = i;
                        result.valuePtr()[position++] = sums[i];
                        data &= data - 1;
                    }
                    bits[word] = 0;
                    sbits &= sbits - 1;
                }
                summary[group] = 0;
                tbits &= tbits - 1;
            }
            top[t] = 0;
        }
    }
    result.outerIndexPtr()[result.outerSize()] = int(written);
    result.resizeNonZeros(written);
    return result;
}
} // namespace sppy
