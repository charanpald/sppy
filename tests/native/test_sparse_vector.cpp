#include "SparseVectorExt.h"
#include <cassert>
#include <cmath>
#include <cstdint>

void test_vector_dot() {
    // Arrange
    SparseVectorExt<double> vector(5);
    vector.insertVal(0, 1.0);
    vector.insertVal(1, 5.2);
    vector.insertVal(3, -0.2);

    // Act
    auto result = vector.dot(vector);

    // Assert
    assert(std::abs(result - 28.08) < 1e-12);
}

void test_indices_and_integer_values() {
    // Arrange
    SparseVectorExt<int64_t> vector(4);
    const int64_t large = (int64_t{1} << 60) + 1;
    vector.insertVal(1, large);
    int64_t indices[1];
    int64_t values[1];

    // Act
    vector.nonZeroInds(indices);
    vector.nonZeroVals(values);

    // Assert
    assert(indices[0] == 1);
    assert(values[0] == large);
}

int main() {
    test_vector_dot();
    test_indices_and_integer_values();
}
