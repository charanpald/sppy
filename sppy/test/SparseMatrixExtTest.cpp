
#include <iostream>
#include <eigen3/Eigen/Sparse>
#include "../include/SparseMatrixExt.h"

using Eigen::SparseMatrix;

int main()
{
    int n = 5; 
    SparseMatrixExt<double> A(7, 7); 
    A.reserve(10);


    A.coeffRef(0, 1) = 1;
    A.coeffRef(1, 3) = 5.2;
    A.coeffRef(3, 3) = -0.2;

    SparseMatrixExt<double> B(7, 7); 
    B.reserve(10);

    B.coeffRef(0, 1) = 1;
    B.coeffRef(1, 3) = 5.2;
    B.coeffRef(3, 3) = -0.2;
    B.coeffRef(4, 4) = 12.2;
    B.coeffRef(0, 6) = -1.23;

    SparseMatrixExt<double> C(5, 5); 
    C.fill(1);
    C.printValues(); 
    //std::cout << C.norm(); 

}
