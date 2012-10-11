
#include <iostream>
#include <eigen3/Eigen/Sparse>
#include "../include/SparseMatrixExt.h"
#include <vector> 

using Eigen::SparseMatrix;

int main()
{
    int n = 5; 
    SparseMatrixExt<double> A(7, 7); 
    A.reserve(10);


    A.coeffRef(0, 1) = 1;
    A.coeffRef(0, 4) = 1.5;
    A.coeffRef(1, 3) = 5.2;
    A.coeffRef(3, 3) = -0.2;

    //std::vector<long> inds = A.getIndsRow(2);
    std::vector<long> inds = A.getIndsCol(4);
 
    for (int j=0;j<inds.size();j++) { 
        std::cout << inds[j] << std::endl; 
        }    

}
