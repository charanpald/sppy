
#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET

#include <iostream>
#include <eigen3/Eigen/Sparse>
#include "../include/SparseMatrixExt.h"

using Eigen::SparseMatrix;

int main()
{
    int n = 5; 
    SparseMatrixExt<double> A(5, 7); 

    A.insertVal(0, 1, 1);
    A.insertVal(1, 3, 5.2);
    A.insertVal(3, 3, -0.2);
    //A.coeffRef(0, 1) = 1;
    //A.coeffRef(1, 3) = 5.2;
    //A.coeffRef(3, 3) = -0.2;
    std::cout << A.sum() << std::endl; 

    Eigen::SparseMatrix<double> B(5, 7); 

    B.coeffRef(0, 1) = 1;
    B.coeffRef(1, 3) = 5.2;
    B.coeffRef(3, 3) = -0.2;
    B.coeffRef(4, 4) = 12.2;
    B.coeffRef(0, 6) = -1.23;
    std::cout << B.sum() << std::endl; 

}
