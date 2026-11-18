#pragma once
#include "common.h"
#include <algorithm>
#include <cstdint>

namespace sppy {
// Assemble directly in the requested storage order, without first building
// a column-major matrix and converting it a second time.
template <class S, class T>
S assemble_triplets(int rows, int cols, Array<int> ri, Array<int> ci, Array<T> values) {
    require(ri.ndim() == 1 && ci.ndim() == 1 && values.ndim() == 1 && ri.size() == ci.size() &&
                ri.size() == values.size(),
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
    S result(rows, cols);
    result.setFromTriplets(entries.begin(), entries.end());
    return result;
}

// Import canonical buffers directly. Unsorted or duplicate inner indices still
// take Eigen's summing triplet path, preserving the public assembly contract.
template <class S, class T>
S import_compressed(int rows, int cols, Array<T> values, Array<int> indices, Array<int> indptr) {
    require(rows >= 0 && cols >= 0, "Negative dimensions");
    const int outer = S::IsRowMajor ? rows : cols;
    const int inner = S::IsRowMajor ? cols : rows;
    require(values.ndim() == 1 && indices.ndim() == 1 && indptr.ndim() == 1 &&
                values.size() == indices.size() && indptr.size() == int64_t(outer) + 1,
            "Invalid compressed array lengths");
    require(values.size() <= std::numeric_limits<int>::max(), "Too many stored entries");
    require(indptr.data()[0] == 0 && indptr.data()[outer] == values.size(),
            "Invalid compressed pointers");
    bool canonical = true;
    for (int j = 0; j < outer; ++j) {
        const int begin = indptr.data()[j], end = indptr.data()[j + 1];
        require(begin >= 0 && begin <= end && end <= values.size(), "Invalid compressed pointers");
        int previous = -1;
        for (int k = begin; k < end; ++k) {
            const int index = indices.data()[k];
            require(index >= 0 && index < inner, "Invalid compressed inner indices");
            canonical = canonical && index > previous;
            previous = index;
        }
    }
    S result(rows, cols);
    if (canonical) {
        result.resizeNonZeros(values.size());
        std::copy_n(indptr.data(), int64_t(outer) + 1, result.outerIndexPtr());
        std::copy_n(indices.data(), values.size(), result.innerIndexPtr());
        std::copy_n(values.data(), values.size(), result.valuePtr());
    } else {
        std::vector<Eigen::Triplet<T>> entries;
        entries.reserve(values.size());
        for (int j = 0; j < outer; ++j)
            for (int k = indptr.data()[j]; k < indptr.data()[j + 1]; ++k)
                entries.emplace_back(S::IsRowMajor ? j : indices.data()[k],
                                     S::IsRowMajor ? indices.data()[k] : j, values.data()[k]);
        result.setFromTriplets(entries.begin(), entries.end());
    }
    return result;
}

struct SliceAxis {
    int64_t start, step;
    int count;
    void validate(int size) const {
        require(count >= 0 && step != 0 && step >= -(int64_t(size) + 1) &&
                    step <= int64_t(size) + 1,
                "Invalid slice descriptor");
        if (count) {
            require(start >= 0 && start < size, "Slice out of bounds");
            int64_t last = start + int64_t(count - 1) * step;
            require(last >= 0 && last < size, "Slice out of bounds");
        }
    }
    int position(int index) const {
        const int64_t delta = int64_t(index) - start;
        if (delta % step != 0)
            return -1;
        const int64_t position = delta / step;
        return position >= 0 && position < count ? int(position) : -1;
    }
};

template <class Result, class Source>
Result basic_slice(const Source &a, SliceAxis rows, SliceAxis cols, int mode) {
    using T = typename Source::Scalar;
    rows.validate(a.rows());
    cols.validate(a.cols());
    require(mode >= 0 && mode <= 2 && (mode != 1 || cols.count == 1) &&
                (mode != 2 || rows.count == 1),
            "Invalid slice dimensionality");
    const int result_rows = mode == 2 ? cols.count : rows.count;
    const int result_cols = mode == 0 ? cols.count : 1;
    Result result(result_rows, result_cols);
    if (!rows.count || !cols.count || !a.nonZeros())
        return result;
    std::vector<Eigen::Triplet<T>> entries;
    entries.reserve(std::min<int64_t>(a.nonZeros(), int64_t(rows.count) * cols.count));
    const SliceAxis &outer = Source::IsRowMajor ? rows : cols;
    const SliceAxis &inner = Source::IsRowMajor ? cols : rows;
    for (int j = 0; j < outer.count; ++j) {
        const int source_outer = int(outer.start + int64_t(j) * outer.step);
        for (typename Source::InnerIterator it(a, source_outer); it; ++it) {
            const int i = inner.position(it.index());
            if (i < 0)
                continue;
            const int row = Source::IsRowMajor ? j : i;
            const int col = Source::IsRowMajor ? i : j;
            entries.emplace_back(mode == 2 ? col : row, mode == 0 ? col : 0, it.value());
        }
    }
    auto less = [](const auto &a, const auto &b) {
        if constexpr (Result::IsRowMajor)
            return a.row() < b.row() || (a.row() == b.row() && a.col() < b.col());
        else
            return a.col() < b.col() || (a.col() == b.col() && a.row() < b.row());
    };
    if (!std::is_sorted(entries.begin(), entries.end(), less))
        std::sort(entries.begin(), entries.end(), less);
    result.setFromSortedTriplets(entries.begin(), entries.end());
    return result;
}
} // namespace sppy
