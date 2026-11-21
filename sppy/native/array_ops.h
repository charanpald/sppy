#pragma once
#include "common.h"
#include "ordered_triplets.h"
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
    const int count = int(values.size());
    unsigned invalid = 0, unsorted = 0;
    int outer_runs = 0;
    const int *rows_data = ri.data(), *cols_data = ci.data();
    for (int k = 0; k < count; ++k)
        invalid |=
            unsigned(rows_data[k]) >= unsigned(rows) || unsigned(cols_data[k]) >= unsigned(cols);
    require(invalid == 0, "Coordinate out of bounds");
    for (int k = 1; k < count; ++k) {
        const int outer = S::IsRowMajor ? rows_data[k] : cols_data[k];
        const int previous_outer = S::IsRowMajor ? rows_data[k - 1] : cols_data[k - 1];
        const int inner = S::IsRowMajor ? cols_data[k] : rows_data[k];
        const int previous_inner = S::IsRowMajor ? cols_data[k - 1] : rows_data[k - 1];
        outer_runs += outer < previous_outer;
        unsorted |= outer < previous_outer || (outer == previous_outer && inner < previous_inner);
    }
    if (unsorted) {
        auto ordered = ordered_triplet_scatter<S>(rows, cols, rows_data, cols_data, values.data(),
                                                  count, outer_runs);
        if (ordered)
            return std::move(*ordered);
    }
    // Eigen can consume a forward iterator over the original NumPy buffers;
    // avoid allocating and copying a separate vector of Triplet objects.
    struct Entry {
        const int *rows, *columns;
        const T *values;
        int position;
        int row() const { return rows[position]; }
        int col() const { return columns[position]; }
        const T &value() const { return values[position]; }
    };
    struct Iterator {
        Entry entry;
        const Entry *operator->() const { return &entry; }
        Iterator &operator++() {
            ++entry.position;
            return *this;
        }
        bool operator!=(const Iterator &other) const {
            return entry.position != other.entry.position;
        }
        bool operator==(const Iterator &other) const {
            return entry.position == other.entry.position;
        }
    };
    Iterator begin{{rows_data, cols_data, values.data(), 0}},
        end{{rows_data, cols_data, values.data(), count}};
    S result(rows, cols);
    if (!unsorted)
        result.setFromSortedTriplets(begin, end, sum_value<T>);
    else
        result.setFromTriplets(begin, end, sum_value<T>);
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
    const int count = int(values.size());
    const int *pointers = indptr.data();
    const int *inner_indices = indices.data();
    // Validate pointers before using them as offsets. Accumulated error flags
    // let the compiler vectorize the scans without a branch/exception per entry.
    unsigned bad_pointers = 0;
    for (int j = 0; j < outer; ++j) {
        bad_pointers |= unsigned(pointers[j]) > unsigned(count);
        bad_pointers |= unsigned(pointers[j + 1]) > unsigned(count);
        bad_pointers |= pointers[j] > pointers[j + 1];
    }
    require(bad_pointers == 0, "Invalid compressed pointers");
    S result(rows, cols);
    result.resizeNonZeros(count);
    int *copied_indices = result.innerIndexPtr();
    unsigned bad_indices = 0;
    int noncanonical = 0;
    if (count) {
        copied_indices[0] = inner_indices[0];
        bad_indices |= unsigned(inner_indices[0]) >= unsigned(inner);
    }
    // Copy, bounds-check and count order violations in one streaming scan.
    // The copied indices are never used as offsets until validation succeeds.
    for (int k = 1; k < count; ++k) {
        const int index = inner_indices[k];
        copied_indices[k] = index;
        bad_indices |= unsigned(index) >= unsigned(inner);
        noncanonical += index <= inner_indices[k - 1];
    }
    require(bad_indices == 0, "Invalid compressed inner indices");
    for (int j = 1; j < outer; ++j) {
        const int boundary = pointers[j];
        if (boundary != pointers[j - 1] && boundary > 0 && boundary < count)
            noncanonical -= inner_indices[boundary] <= inner_indices[boundary - 1];
    }
    const bool canonical = noncanonical == 0;
    if (canonical) {
        std::copy_n(indptr.data(), int64_t(outer) + 1, result.outerIndexPtr());
        std::copy_n(values.data(), values.size(), result.valuePtr());
    } else {
        std::vector<Eigen::Triplet<T>> entries;
        entries.reserve(values.size());
        for (int j = 0; j < outer; ++j)
            for (int k = indptr.data()[j]; k < indptr.data()[j + 1]; ++k)
                entries.emplace_back(S::IsRowMajor ? j : indices.data()[k],
                                     S::IsRowMajor ? indices.data()[k] : j, values.data()[k]);
        result.setFromTriplets(entries.begin(), entries.end(), sum_value<T>);
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
