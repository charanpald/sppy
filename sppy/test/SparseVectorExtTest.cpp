
#include <iostream>
#include <eigen3/Eigen/Sparse>
#include "../include/SparseVectorExt.h"

using Eigen::SparseVector;

int main()
{
    int n = 5; 
    SparseVectorExt<double> vec1(n); 
    vec1.reserve(10);


    vec1.coeffRef(0) = 1;
    vec1.coeffRef(1) = 5.2;
    vec1.coeffRef(3) = -0.2;

    vec1.printValues();

    long nonZeroInds[3]; 
    vec1.nonZeroInds(nonZeroInds);
    
    for (int i=0;i<3;i++)
        std::cout << nonZeroInds[i] << std::endl; 
}
