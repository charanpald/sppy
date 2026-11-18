#include "SparseMatrixExt.h"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

void test_triplets_preserve_large_integers() {
    // Arrange
    SparseMatrixExt<int64_t> matrix(2, 2);
    int rows[] = {0, 1};
    int columns[] = {1, 0};
    int64_t values[] = {(int64_t{1} << 60) + 1, -3};

    // Act
    matrix.putUsingTriplets(rows, columns, values, 2);

    // Assert
    assert(matrix.coeff(0, 1) == values[0]);
    assert(matrix.coeff(1, 0) == -3);
}

void test_sparse_indices_and_slice() {
    // Arrange
    SparseMatrixExt<double, Eigen::RowMajor> matrix(3, 4);
    matrix.insertVal(0, 1, 2.0);
    matrix.insertVal(2, 3, -3.0);
    int rows[2];
    int columns[2];
    int selected_rows[] = {0, 2};
    int selected_columns[] = {1, 3};
    SparseMatrixExt<double, Eigen::RowMajor> slice(2, 2);

    // Act
    matrix.nonZeroInds(rows, columns);
    matrix.slice(selected_rows, 2, selected_columns, 2, &slice);
    auto row_indices = matrix.getIndsRow(0);
    auto column_indices = matrix.getIndsCol(3);

    // Assert
    assert(rows[0] == 0 && rows[1] == 2);
    assert(columns[0] == 1 && columns[1] == 3);
    assert(slice.coeff(0, 0) == 2.0 && slice.coeff(1, 1) == -3.0);
    assert(row_indices == std::vector<int>{1});
    assert(column_indices == std::vector<int>{2});
}

void test_dense_products() {
    // Arrange
    SparseMatrixExt<double, Eigen::RowMajor> matrix(2, 2);
    matrix.insertVal(0, 1, 2.0);
    matrix.insertVal(1, 0, -3.0);
    double operand[] = {1, 2, 3, 4};
    double product[4] = {};
    double partial[4] = {};

    // Act
    matrix.dot2d(operand, 2, product);
    matrix.dotSub2d(operand, 2, 0, 1, partial);

    // Assert
    assert(product[0] == 6 && product[1] == 8);
    assert(product[2] == -3 && product[3] == -6);
    assert(partial[0] == 6 && partial[1] == 8);
}

void test_reserve_handles_uncompressed_storage() {
    // Arrange
    SparseMatrixExt<double> matrix(2, 3);
    matrix.insertVal(0, 1, 2.0);

    // Act
    matrix.reserve(10);
    matrix.fill(1.0);

    // Assert
    assert(matrix.nonZeros() == 6);
    assert(matrix.sum() == 6);
}

void test_transpose_and_block() {
    // Arrange
    SparseMatrixExt<double> matrix(3, 4);
    matrix.insertVal(0, 1, 2.0);
    matrix.insertVal(2, 3, -3.0);

    // Act
    auto transposed = matrix.trans();
    auto block = matrix.submatrix(1, 2, 2, 2);

    // Assert
    assert(transposed.rows() == 4 && transposed.cols() == 3);
    assert(transposed.coeff(1, 0) == 2 && transposed.coeff(3, 2) == -3);
    assert(block.rows() == 2 && block.cols() == 2);
    assert(block.coeff(1, 1) == -3);
}

int main() {
    test_triplets_preserve_large_integers();
    test_sparse_indices_and_slice();
    test_dense_products();
    test_reserve_handles_uncompressed_storage();
    test_transpose_and_block();
}
