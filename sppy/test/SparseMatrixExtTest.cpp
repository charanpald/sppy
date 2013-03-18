
#include <iostream>
#include <eigen3/Eigen/Sparse>
#include "../include/SparseMatrixExt.h"
#include <vector> 

using Eigen::SparseMatrix;

void testNonZeroInds() { 
    int n = 5; 
    SparseMatrixExt<double, Eigen::RowMajor> A(7, 7); 
    A.reserve(10);


    A.coeffRef(0, 1) = 1;
    A.coeffRef(0, 4) = 1.5;
    A.coeffRef(1, 3) = 5.2;
    A.coeffRef(3, 3) = -0.2;

    long rowInds[4]; 
    long colInds[4];

    A.nonZeroInds(rowInds, colInds);

    for(int i=0;i<4;i++) { 
        std::cout << rowInds[i] << " " << colInds[i] << " " << A.coeffRef(rowInds[i], colInds[i]) << std::endl;         
        }
}

void testSlice() { 
    int n = 5; 
    SparseMatrixExt<double, Eigen::ColMajor> A(7, 7); 
    A.reserve(10);

    A.coeffRef(0, 1) = 1;
    A.coeffRef(0, 4) = 1.5;
    A.coeffRef(1, 3) = 5.2;
    A.coeffRef(3, 3) = -0.2;

    int rowInds[2] = {0, 1};
    int colInds[3] = {1, 3, 4}; 

    SparseMatrixExt<double, Eigen::ColMajor> B(2, 3);

    A.slice(rowInds, 2, colInds, 3, &B);

    B.printValues();
}

void testGetIndsRow() { 
    int n = 5; 
    SparseMatrixExt<double, Eigen::RowMajor> A(7, 7); 
    A.reserve(10);

    A.coeffRef(0, 1) = 1;
    A.coeffRef(0, 4) = 1.5;
    A.coeffRef(1, 3) = 5.2;
    A.coeffRef(3, 3) = -0.2;

    std::vector<long> inds = A.getIndsRow(1);
    
    for(int i = 0; i<inds.size();i++)
        std::cout << inds[i] << std::endl; 
    
}

void testGetIndsCol() { 
    int n = 5; 
    SparseMatrixExt<double, Eigen::ColMajor> A(7, 7); 
    A.reserve(10);

    A.coeffRef(0, 1) = 1;
    A.coeffRef(0, 4) = 1.5;
    A.coeffRef(1, 3) = 5.2;
    A.coeffRef(3, 3) = -0.2;

    std::vector<long> inds = A.getIndsCol(3);
    
    for(int i = 0; i<inds.size();i++)
        std::cout << inds[i] << std::endl; 
    
}

int main()
{
    testGetIndsCol();

}
