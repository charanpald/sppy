
#include <iostream>
#include <eigen3/Eigen/Sparse>
#include "../include/SparseVectorExt.h"

using Eigen::SparseVector;

int main()
{
    int n = 5; 
    SparseVectorExt<double> vec1(n); 
    vec1.reserve(5);


    vec1.coeffRef(0) = 1;
    vec1.coeffRef(1) = 5.2;
    vec1.coeffRef(3) = -0.2;

    std::cout << vec1.dot(vec1) << std::endl; 
}
