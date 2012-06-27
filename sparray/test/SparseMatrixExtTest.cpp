
#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET

#include <iostream>
#include <eigen3/Eigen/Sparse>

using Eigen::SparseMatrix;

int main()
{
    int n = 5; 
    Eigen::SparseMatrix<double> A(n, n); 

    A.coeffRef(0, 0) = 23.1;
    A.coeffRef(2, 0) = -3.1;
    A.coeffRef(3, 0) = -10.0;
    A.coeffRef(2, 1) = -5;
    A.coeffRef(3, 1) = 5;
    std::cout << A.sum() << std::endl; 

    Eigen::SparseMatrix<double> B(5, 7); 

    B.coeffRef(0, 1) = 1;
    B.coeffRef(1, 3) = 5.2;
    B.coeffRef(3, 3) = -0.2;
    B.coeffRef(4, 4) = 12.2;
    B.coeffRef(0, 6) = -1.23;
    std::cout << B.sum() << std::endl; 

}
