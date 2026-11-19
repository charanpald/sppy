#pragma once
#include <Eigen/SparseCore>
#include <algorithm>
#include <optional>
#include <vector>
namespace sppy {
// Diagonal/run-based COO input is often already monotone within every output
// outer vector, even though the input as a whole is unsorted. Probe one vector
// before attempting a count/scatter; verify every vector before accepting it.
template <class S, class T>
std::optional<S> ordered_triplet_scatter(int rows, int columns, const int *ri, const int *ci,
                                         const T *values, int count) {
    const int outer_size = S::IsRowMajor ? rows : columns;
    if (count < 20000 || outer_size < 2048)
        return std::nullopt;
    const int *outer = S::IsRowMajor ? ri : ci;
    const int *inner = S::IsRowMajor ? ci : ri;
    const int sample = outer[count / 2];
    int last = -1, found = 0;
    bool ascending = true, descending = true;
    for (int k = 0; k < count; ++k) {
        if (outer[k] != sample)
            continue;
        if (found) {
            ascending &= inner[k] > last;
            descending &= inner[k] < last;
        }
        last = inner[k];
        ++found;
        if (!ascending && !descending)
            return std::nullopt;
    }
    if (found < 4)
        return std::nullopt;
    S result(rows, columns);
    int *pointers = result.outerIndexPtr();
    for (int k = 0; k < count; ++k)
        pointers[outer[k] + 1]++;
    for (int j = 0; j < outer_size; ++j)
        pointers[j + 1] += pointers[j];
    result.resizeNonZeros(count);
    std::vector<int> cursor(pointers, pointers + outer_size);
    int *indices = result.innerIndexPtr();
    T *data = result.valuePtr();
    for (int k = 0; k < count; ++k) {
        const int at = cursor[outer[k]]++;
        indices[at] = inner[k];
        data[at] = values[k];
    }
    int bad_ascending = 0, bad_descending = 0;
    for (int k = 1; k < count; ++k) {
        bad_ascending += indices[k] <= indices[k - 1];
        bad_descending += indices[k] >= indices[k - 1];
    }
    for (int j = 1; j < outer_size; ++j) {
        const int boundary = pointers[j];
        if (boundary != pointers[j - 1] && boundary > 0 && boundary < count) {
            bad_ascending -= indices[boundary] <= indices[boundary - 1];
            bad_descending -= indices[boundary] >= indices[boundary - 1];
        }
    }
    if (!bad_ascending)
        return result;
    if (!bad_descending) {
        for (int j = 0; j < outer_size; ++j) {
            std::reverse(indices + pointers[j], indices + pointers[j + 1]);
            std::reverse(data + pointers[j], data + pointers[j + 1]);
        }
        return result;
    }
    return std::nullopt;
}
} // namespace sppy
