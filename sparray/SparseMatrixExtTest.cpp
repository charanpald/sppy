
#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET

#include <iostream>
#include <Eigen/Sparse>
#include "include/DynamicSparseMatrixExt.h"

using Eigen::SparseMatrix;

int main()
{
    DynamicSparseMatrixExt<double> m(3,3);
    m.insert(0,0) = 3;
    m.insert(1,0) = 2.5;
    m.insert(0,1) = -1;
    m.insert(2,0) = -1;    
    m.insert(2,2) = 5.12;
    
    m.printValues();

    int array1[] = { 0, 2 };
    int size1 = 2;
    int array2[] = { 0, 2 };
    int size2 = 2; 

    DynamicSparseMatrixExt<double> *m2 = m.slice(array1, size1, array2, size2);

    std::cout << std::endl; 
    m2->printValues();

}